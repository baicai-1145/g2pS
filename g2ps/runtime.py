"""g2pS numpy-only runtime. Deps: numpy (that's it).

Weights: models/<tier>/<tier>.npz (fp16 storage, fp32 compute) + meta.pkl.
"""
import os
import pickle

# set default thread count BEFORE numpy loads BLAS (bigger models benefit from threads)
_tier_hint = os.environ.get('G2PS_THREADS')
if _tier_hint is None:
    os.environ.setdefault('OMP_NUM_THREADS', '8')

import numpy as np

MAXLEN = 128
EPS = 1e-6

ARCH = {
    'small': dict(D=384, L=6, H=6, FFN=1024, emb=False, rank=0, passes=1),
    'tiny': dict(D=256, L=5, H=4, FFN=640, emb=True, rank=64, passes=1),
    'nano': dict(D=96, L=6, H=4, FFN=256, emb=True, rank=16, passes=2),
}


def _set_blas_threads(n):
    try:
        import glob
        import ctypes
        for lib in glob.glob(os.path.join(os.path.dirname(np.__file__), '..', 'numpy.libs', 'libopenblas*.so*')) \
                + glob.glob(os.path.join(os.path.dirname(np.__file__), '.libs', 'libopenblas*.so*')) \
                + glob.glob(os.path.join(os.path.dirname(np.__file__), '..', 'scipy.libs', 'libopenblas*.so*')):
            try:
                ctypes.CDLL(lib).openblas_set_num_threads(n)
            except OSError:
                pass
    except Exception:
        pass


class G2pS:
    def __init__(self, model_dir, tier=None):
        tier = tier or os.path.basename(os.path.normpath(model_dir))
        a = ARCH[tier]
        self.tier = tier
        self.D, self.L, self.H, self.FFN, self.PASSES = a['D'], a['L'], a['H'], a['FFN'], a['passes']
        self.DH = a['D'] // a['H']
        D, L = self.D, self.L
        _set_blas_threads(1 if D <= 96 else 8)

        npz = os.path.join(model_dir, tier + '.npz')
        z = np.load(npz)
        w = {k: z[k].astype(np.float32) for k in z.files}
        self.emb = w['emb.weight'] @ w['emb_proj.weight'].T if a['emb'] else w['emb.weight']
        self.w12 = [np.ascontiguousarray(w[f'blocks.{li}.n2.w'][:, None] * np.concatenate(
            [w[f'blocks.{li}.ffn.w1.weight'].T, w[f'blocks.{li}.ffn.w2.weight'].T], 1)) for li in range(L)]
        self.w3 = [np.ascontiguousarray(w[f'blocks.{li}.ffn.w3.weight'].T) for li in range(L)]
        self.qkv = [np.ascontiguousarray(w[f'blocks.{li}.n1.w'][:, None] * w[f'blocks.{li}.attn.qkv.weight'].T) for li in range(L)]
        self.o = [np.ascontiguousarray(w[f'blocks.{li}.attn.o.weight'].T) for li in range(L)]
        self.char_bias = (w['char_bias.0.weight'] @ w['char_bias.1.weight'].T) if a['rank'] else w['char_bias.weight']
        self.stem_w = [w[f'stem.convs.{i}.weight'][:, 0, :] for i in range(3)]
        self.stem_b = [w[f'stem.convs.{i}.bias'] for i in range(3)]
        self.stem_n = w['stem.norm.w']
        self.readout = np.ascontiguousarray(w['n_final.w'][:, None] * w['readout.weight'].T)
        inv = 1.0 / (10000.0 ** (np.arange(0, self.DH, 2, dtype=np.float64) / self.DH))
        ang = np.outer(np.arange(MAXLEN), inv)
        self.rope_cos = np.repeat(np.cos(ang)[:, None, :], self.H, 1).astype(np.float32)
        self.rope_sin = np.repeat(np.sin(ang)[:, None, :], self.H, 1).astype(np.float32)

        meta = pickle.load(open(os.path.join(model_dir, 'meta.pkl'), 'rb'))
        self.vocab = meta['vocab']
        self.labels = meta['labels']
        self.poly_id = {c: i for i, c in enumerate(meta['poly_chars'])}
        self.char_cands = meta['char_cands']

    def _rms(self, x):
        ms = (x * x).sum(-1, keepdims=True) * (1.0 / self.D) + EPS
        return x * (1.0 / np.sqrt(ms))

    def forward(self, char_ids, poly_ids, positions, cand):
        T = len(char_ids)
        D, H, DH = self.D, self.H, self.DH
        x = self.emb[char_ids]
        s = np.zeros_like(x)
        for i, dl in enumerate((1, 2, 3)):
            wk, b = self.stem_w[i], self.stem_b[i]
            xp = np.zeros((T + 2 * dl, D), dtype=np.float32)
            xp[dl:dl + T] = x
            s += b
            for k in range(3):
                s += wk[:, k] * xp[k * dl:k * dl + T]
        x = self._rms(x + s) * self.stem_n
        am = np.where(char_ids != 0, np.float32(0), np.float32(-1e4))[None, None, :]
        scale = DH ** -0.5
        for _ in range(self.PASSES):
            for li in range(self.L):
                qkv = self._rms(x) @ self.qkv[li]
                q = qkv[:, :D]; k = qkv[:, D:2 * D]
                q4 = q.reshape(T, H, DH // 2, 2)
                k4 = k.reshape(T, H, DH // 2, 2)
                cos, sin = self.rope_cos[:T], self.rope_sin[:T]
                qr = np.empty((T, H, DH // 2, 2), dtype=np.float32)
                kr = np.empty((T, H, DH // 2, 2), dtype=np.float32)
                qe, qo = q4[..., 0], q4[..., 1]
                ke, ko = k4[..., 0], k4[..., 1]
                qr[..., 0] = qe * cos - qo * sin
                qr[..., 1] = qe * sin + qo * cos
                kr[..., 0] = ke * cos - ko * sin
                kr[..., 1] = ke * sin + ko * cos
                qh = qr.reshape(T, H, DH).transpose(1, 0, 2)
                kh = kr.reshape(T, H, DH).transpose(1, 0, 2)
                vh = qkv[:, 2 * D:].reshape(T, H, DH).transpose(1, 0, 2)
                a = qh @ kh.transpose(0, 2, 1) * scale + am
                a -= a.max(-1, keepdims=True)
                np.exp(a, out=a)
                a /= a.sum(-1, keepdims=True)
                y = (a @ vh).transpose(1, 0, 2).reshape(T, D)
                x = x + y @ self.o[li]
                ab = self._rms(x) @ self.w12[li]
                A, B_ = ab[:, :self.FFN], ab[:, self.FFN:]
                A /= (1.0 + np.exp(-A))
                x = x + (A * B_) @ self.w3[li]
        ph = self._rms(x)[positions]
        logits = ph @ self.readout + self.char_bias[poly_ids]
        logits = np.where(cand, logits, np.float32(-1e4))
        return logits

    def predict(self, text):
        """Returns {position: pinyin} over the (lowercased, maybe truncated) text."""
        text = text.lower()
        positions = [i for i, c in enumerate(text) if c in self.poly_id]
        if not positions:
            return {}, text
        if len(text) > MAXLEN:
            first = positions[0]
            start = max(0, min(first - MAXLEN // 2, len(text) - MAXLEN))
            text = text[start:start + MAXLEN]
            positions = [p - start for p in positions if start <= p < start + MAXLEN]
            if not positions:
                return {}, text
        char_ids = np.array([self.vocab.get(c, 1) for c in text], dtype=np.int64)
        poly_ids = np.array([self.poly_id[text[p]] for p in positions])
        cand = np.zeros((len(positions), len(self.labels)), dtype=bool)
        for k, p in enumerate(positions):
            cand[k, list(self.char_cands[text[p]])] = True
        logits = self.forward(char_ids, poly_ids, np.array(positions), cand)
        pred = logits.argmax(-1)
        return {p: self.labels[int(pred[k])] for k, p in enumerate(positions)}, text

    def convert(self, text):
        """Returns (text_used, [(char, pinyin_or_None), ...]) — polyphones only; pair with
        pypinyin or any dictionary for monophonic chars."""
        res, t = self.predict(text)
        return t, [(c, res.get(i)) for i, c in enumerate(t)]


if __name__ == '__main__':
    import sys
    g = G2pS(sys.argv[1] if len(sys.argv) > 1 else 'models/nano')
    for s in ['他在银行工作，行踪不定。', '恢复汉阳区人民政府', '能把小孩哄睡是个技术活。']:
        _, pairs = g.convert(s)
        print(s, '->', [(c, p) for c, p in pairs if p])
