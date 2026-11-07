"""Student model: modern small char-level encoder.

6 x 384 bidirectional transformer with RoPE / RMSNorm / SwiGLU (2024-style recipe),
candidate-masked readout at every labeled polyphone position.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rope_table(max_len, dim, base=10000.0, device='cpu', dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float64)
    angles = torch.outer(t, inv)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).to(dtype)  # (T, D/2, 2)


def apply_rope(x, rope):
    """x: (B, T, H, Dh); rope: (T, Dh/2, 2)"""
    T = x.shape[1]
    r = rope[:T].to(x.dtype)  # (T, Dh/2, 2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = r[..., 0].unsqueeze(0).unsqueeze(2)  # (1, T, 1, Dh/2)
    sin = r[..., 1].unsqueeze(0).unsqueeze(2)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mask, rope):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_heads, self.dh).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.dh).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dh).transpose(1, 2)
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        # mask: (B, T) bool True=valid -> (B,1,1,T) additive
        att_mask = (~mask).unsqueeze(1).unsqueeze(2) * -1e4  # (B,1,1,T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(y)


class ConvStem(nn.Module):
    """depthwise conv k=3 with dilations 1/2/3 -> fused local n-gram features. ~3.5K params."""
    def __init__(self, d):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, padding=dl, dilation=dl, groups=d) for dl in (1, 2, 3)])
        self.norm = RMSNorm(d)

    def forward(self, x):
        # x: (B, T, D); padding-aware: zero pad chars already are 0-emb? mask conv beyond seq via padding zeros
        h = x.transpose(1, 2)  # (B, D, T)
        hs = [c(h) for c in self.convs]
        out = x + sum(hs).transpose(1, 2)
        return self.norm(out)

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, ffn_hidden):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d, ffn_hidden)

    def forward(self, x, mask, rope):
        x = x + self.attn(self.n1(x), mask, rope)
        x = x + self.ffn(self.n2(x))
        return x


class StudentG2P(nn.Module):
    def __init__(self, vocab_size, n_labels, n_poly_chars, d=384, n_layers=6,
                 n_heads=6, ffn_hidden=1024, max_len=128, dropout=0.1, use_conv_stem=True,
                 emb_dim=None, bias_rank=0, n_passes=1):
        self.n_passes = n_passes
        super().__init__()
        rope = rope_table(max_len, d // n_heads)
        self.register_buffer('rope', rope, persistent=False)
        emb_dim = emb_dim or d
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_proj = nn.Linear(emb_dim, d, bias=False) if emb_dim != d else nn.Identity()
        self.stem = ConvStem(d) if use_conv_stem else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, n_heads, ffn_hidden)
                                     for _ in range(n_layers)])
        self.n_final = RMSNorm(d)
        self.readout = nn.Linear(d, n_labels, bias=False)
        # tone-factored auxiliary heads (set via set_tone_heads)
        self.n_syl = 0
        # per (poly char, label) bias; low-rank option for tiny models
        if bias_rank > 0:
            self.char_bias = nn.Sequential(
                nn.Embedding(n_poly_chars, bias_rank),
                nn.Linear(bias_rank, n_labels, bias=False),
            )
        else:
            self.char_bias = nn.Embedding(n_poly_chars, n_labels)
        nn.init.zeros_(self.char_bias[-1].weight if bias_rank > 0 else self.char_bias.weight)
        self.pos_head = nn.Linear(d, 11)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def set_tone_heads(self, n_syl, n_tone=6):
        import torch.nn as nn
        self.n_syl = n_syl
        dev = self.readout.weight.device
        self.syl_head = nn.Linear(self.readout.in_features, n_syl).to(dev)
        self.tone_head = nn.Linear(self.readout.in_features, n_tone).to(dev)

    def init_bias_prior(self, counts):
        """counts: list (aligned with poly chars) of dict label_id -> count"""
        import numpy as np
        if isinstance(self.char_bias, nn.Sequential):
            n = self.char_bias[0].weight.shape[0]
            L = self.char_bias[-1].weight.shape[0]
        else:
            n = self.char_bias.weight.shape[0]
            L = self.char_bias.weight.shape[1]
        M = np.zeros((n, L), dtype=np.float32)
        for i, d in enumerate(counts):
            tot = sum(d.values()) + 1e-9
            for j, c in d.items():
                M[i, j] = math.log((c + 1.0) / (tot + len(d)))
        if isinstance(self.char_bias, nn.Sequential):
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            r = self.char_bias[0].weight.shape[1]
            emb = U[:, :r] * np.sqrt(S[:r])
            lin = (np.sqrt(S[:r])[:, None] * Vh[:r]).T
            with torch.no_grad():
                self.char_bias[0].weight.copy_(torch.from_numpy(emb))
                self.char_bias[-1].weight.copy_(torch.from_numpy(lin))
        else:
            with torch.no_grad():
                self.char_bias.weight.copy_(torch.from_numpy(M))

    def forward_with_ids(self, char_ids, poly_ids, positions, pos_mask, cand_mask):
        pad = char_ids.ne(0)
        x = self.emb_proj(self.emb(char_ids))
        x = self.drop(x)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x, pad, self.rope)
        h = self.n_final(x)
        B, T, D = h.shape
        P = positions.shape[1]
        idx = positions.clamp(min=0).unsqueeze(-1).expand(B, P, D)
        ph = torch.gather(h, 1, idx)
        poly_ids_s = poly_ids.clamp(min=0)
        logits = self.readout(ph) + self.char_bias(poly_ids_s)
        masked = logits.masked_fill(~cand_mask, -1e4)
        logp = F.log_softmax(masked.float(), dim=-1)
        pos_logits = self.pos_head(ph)
        extra = (self.syl_head(ph), self.tone_head(ph)) if self.n_syl else None
        return logp, pos_logits, ph, masked, extra
