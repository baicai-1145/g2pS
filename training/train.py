"""Train the student model on sentence-level records.

Loss: masked-softmax CE with label smoothing (smoothed over the char's candidates)
      + 0.1 * POS auxiliary CE
      + R-Drop KL between two dropout passes
Recipe: AdamW, warmup + cosine, EMA of weights.
"""
import argparse
import copy
import math
import pickle
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from student.modules import StudentG2P
from student.data import POS_TAGS, MAX_LEN


class RecordDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


def collate(batch, n_labels, char_cand_mask, pair_w=None):
    """char_cand_mask: dict poly_char_id -> LongTensor (n_labels,) of 0/1;
    pair_w: dict (poly_id,label_id) -> weight, computed on CPU side"""
    B = len(batch)
    max_t = max(len(r[0]) for r in batch)
    max_p = max(len(r[1]) for r in batch)
    char_ids = torch.zeros(B, max_t, dtype=torch.long)
    positions = torch.zeros(B, max_p, dtype=torch.long)
    pos_mask = torch.zeros(B, max_p, dtype=torch.bool)
    cand_mask = torch.zeros(B, max_p, n_labels, dtype=torch.bool)
    label_ids = torch.zeros(B, max_p, dtype=torch.long)
    pos_ids = torch.zeros(B, max_p, dtype=torch.long)
    poly_ids = torch.zeros(B, max_p, dtype=torch.long)
    t_probs = torch.zeros(B, max_p, n_labels) if (batch and len(batch[0]) > 5) else None
    weights = torch.ones(B, max_p) if pair_w else None

    for b, r in enumerate(batch):
        cids, pp, ll, poss, poly = r[0], r[1], r[2], r[3], r[4]
        char_ids[b, :len(cids)] = torch.tensor(cids)
        for k, (pos, lab, pst, pol) in enumerate(zip(pp, ll, poss, poly)):
            positions[b, k] = pos
            label_ids[b, k] = lab
            pos_ids[b, k] = pst
            poly_ids[b, k] = pol
            pos_mask[b, k] = True
            cand_mask[b, k] = char_cand_mask[pol].bool()
            if pair_w:
                weights[b, k] = pair_w.get((pol, lab), 1.0)
            if t_probs is not None and len(r) > 5:
                kd = r[5]
                if isinstance(kd, tuple):  # sparse top8: (idx (P,8), val (P,8)) numpy
                    t_probs[b, k] = 0.0
                    t_probs[b, k, torch.from_numpy(kd[0][k].astype('int64'))] = torch.from_numpy(kd[1][k].astype('float32'))
                else:
                    t_probs[b, k] = kd[k]
    if pair_w:
        out = (char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids, weights)
    else:
        out = (char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids)
    if t_probs is not None:
        out = out + (t_probs,)
    return out
    return (char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(d).add_(pm.detach(), alpha=1 - d)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)

    def state_dict(self):
        return self.shadow.state_dict()


def masked_ce(logp, label_ids, pos_mask, cand_mask, smoothing=0.05, w=None):
    """logp (B,P,L); label (B,P); pos_mask (B,P); cand_mask (B,P,L); w (B,P) optional weights"""
    nll = -logp.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)  # (B,P)
    target_term = (1 - smoothing) * nll
    cand_logp = (logp * cand_mask).sum(-1) / cand_mask.sum(-1).clamp(min=1)
    smooth_term = -smoothing * cand_logp
    loss = (target_term + smooth_term) * pos_mask
    if w is not None:
        loss = loss * w
    return loss.sum() / (pos_mask * (w if w is not None else 1.0)).sum().clamp(min=1)


def kl_between(logp1, logp2, pos_mask):
    """symmetric KL over distributions at valid positions"""
    p1 = logp1.exp()
    p2 = logp2.exp()
    kl = (p1 * (logp1 - logp2)).sum(-1) + (p2 * (logp2 - logp1)).sum(-1)
    return (kl * pos_mask).sum() / pos_mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, char_cand_mask, n_labels, device):
    model.eval()
    per_char = defaultdict(lambda: [0, 0])
    per_pair = defaultdict(lambda: [0, 0])
    correct = total = 0
    for batch in loader:
        char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids = \
            [x.to(device) for x in batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logp, _, _, _, _ = model.forward_with_ids(char_ids, poly_ids, positions, pos_mask, cand_mask)
        logp = logp.float()
        pred = logp.argmax(-1)
        for b in range(pred.shape[0]):
            for k in range(pred.shape[1]):
                if not pos_mask[b, k]:
                    continue
                hit = int(pred[b, k] == label_ids[b, k])
                poly = int(poly_ids[b, k])
                lab = int(label_ids[b, k])
                correct += hit
                total += 1
                per_char[poly][0] += hit
                per_char[poly][1] += 1
                per_pair[(poly, lab)][0] += hit
                per_pair[(poly, lab)][1] += 1
    acc = correct / total
    avg_p = sum(h for h, _ in per_char.values()) / max(1, sum(n for _, n in per_char.values()))
    avg_pp = sum(v[0] / v[1] for v in per_pair.values()) / max(1, len(per_pair))
    model.train()
    return {'acc': acc, 'avg_p': avg_p, 'avg_pp': avg_pp, 'n': total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='datasets/student')
    ap.add_argument('--exp', default='student_v1')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=800)
    ap.add_argument('--d', type=int, default=384)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--ffn', type=int, default=1024)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--rdrop', type=float, default=1.0)
    ap.add_argument('--label_smooth', type=float, default=0.05)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--max_positions', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1313)
    ap.add_argument('--eval_every', type=int, default=2000)
    ap.add_argument('--pos_weight', type=float, default=0.1)
    ap.add_argument('--use_conv_stem', type=int, default=1)
    ap.add_argument('--rew_alpha', type=float, default=0.0)
    ap.add_argument('--emb_dim', type=int, default=0)
    ap.add_argument('--bias_rank', type=int, default=0)
    ap.add_argument('--n_passes', type=int, default=1)
    ap.add_argument('--compile', type=int, default=0)
    ap.add_argument('--tone_aux', type=int, default=0)
    ap.add_argument('--tone_w', type=float, default=0.3)
    ap.add_argument('--train_pkl', default='train.pkl')
    ap.add_argument('--kd_path', default='')
    ap.add_argument('--kd_T', type=float, default=2.0)
    ap.add_argument('--kd_alpha', type=float, default=0.5)
    ap.add_argument('--bf16', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    meta = pickle.load(open(f'{args.data_root}/meta.pkl', 'rb'))
    vocab, labels, poly_chars = meta['vocab'], meta['labels'], meta['poly_chars']
    char_cands = meta['char_cands']  # char -> set(label_id)
    n_labels = len(labels)
    label_id = {l: i for i, l in enumerate(labels)}
    poly_id = {c: i for i, c in enumerate(poly_chars)}

    train = pickle.load(open(f'{args.data_root}/{args.train_pkl}', 'rb'))
    dev = pickle.load(open(f'{args.data_root}/dev.pkl', 'rb'))
    test = pickle.load(open(f'{args.data_root}/test.pkl', 'rb'))
    print(f'train sentences: {len(train)}, dev: {len(dev)}, test: {len(test)}')

    if args.kd_path:
        tp = torch.load(args.kd_path)
        probs, offsets = tp['probs'], tp['offsets']
        if len(train) > 0 and len(train[0]) > 5:
            print('records already carry KD tensors; using them')
        else:
            assert offsets[-1].item() == sum(len(r[1]) for r in train), 'teacher probs do not align with train.pkl'
            train = [r + (probs[offsets[i]:offsets[i + 1]].float(),) for i, r in enumerate(train)]
        print(f'loaded teacher probs: {tuple(probs.shape)}')

    # candidate mask per poly char, plus log-freq prior counts
    char_cand_mask = {}
    counts = []
    for i, ch in enumerate(poly_chars):
        m = torch.zeros(n_labels, dtype=torch.bool)
        for j in char_cands[ch]:
            m[j] = True
        char_cand_mask[i] = m
        counts.append({})
    # truncate to max_positions per record
    def trim(records):
        out = []
        for r in records:
            if len(r[1]) > args.max_positions:
                r = (r[0], r[1][:args.max_positions], r[2][:args.max_positions],
                     r[3][:args.max_positions], r[4][:args.max_positions]) + \
                    ((r[5][:args.max_positions],) if len(r) > 5 else ())
            out.append(r)
        return out
    train = trim(train)

    for r in train:
        for pol, lab in zip(r[4], r[2]):
            counts[pol][lab] = counts[pol].get(lab, 0) + 1

    model = StudentG2P(len(vocab), n_labels, len(poly_chars),
                       d=args.d, n_layers=args.layers, n_heads=args.heads,
                       ffn_hidden=args.ffn, max_len=MAX_LEN, dropout=args.dropout,
                       use_conv_stem=bool(args.use_conv_stem),
                       emb_dim=(args.emb_dim or None),
                       bias_rank=args.bias_rank, n_passes=args.n_passes).to(device)
    if args.tone_aux:
        import re as _re
        def split_label(lb):
            m = _re.match(r'^(.*?)([1-5])$', lb)
            return (m.group(1), int(m.group(2))) if m else (lb, 0)
        syls = sorted({split_label(l)[0] for l in labels})
        syl_id = {s: i for i, s in enumerate(syls)}
        tone_of = {lid: split_label(l)[1] for lid, l in enumerate(labels)}
        syl_of = {lid: syl_id[split_label(l)[0]] for lid, l in enumerate(labels)}
        model.set_tone_heads(len(syls))
        print(f'tone aux: {len(syls)} syllables')
    if args.compile:
        model = torch.compile(model, dynamic=True)
        print('torch.compile enabled')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model params: {n_params/1e6:.1f}M')
    model.init_bias_prior(counts)

    train_loader = DataLoader(RecordDataset(train), batch_size=args.batch_size,
                              shuffle=True, collate_fn=lambda b: collate(b, n_labels, char_cand_mask, pair_w or None),
                              num_workers=4, drop_last=True, pin_memory=True)
    dev_loader = DataLoader(RecordDataset(dev), batch_size=256,
                            collate_fn=lambda b: collate(b, n_labels, char_cand_mask), num_workers=2)
    test_loader = DataLoader(RecordDataset(test), batch_size=256,
                             collate_fn=lambda b: collate(b, n_labels, char_cand_mask), num_workers=2)

    # per-(poly,label) mild inverse-freq weights
    pair_count = {}
    for r in train:
        for pol, lab in zip(r[4], r[2]):
            pair_count[(pol, lab)] = pair_count.get((pol, lab), 0) + 1
    import statistics as _st
    cnts = list(pair_count.values())
    med = _st.median(cnts)
    pair_w = {}
    if args.rew_alpha > 0:
        for k_, c_ in pair_count.items():
            pair_w[k_] = min(2.0, max(0.5, (med / max(1, c_)) ** args.rew_alpha))
    print(f'reweight alpha={args.rew_alpha}, weighted pairs: {len(pair_w)}')

    ema = EMA(model, decay=args.ema_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f'steps/epoch: {steps_per_epoch}, total steps: {total_steps}')

    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * t))

    step = 0
    best = 0.0
    t0 = time.time()
    log_path = f'{args.data_root}/../{args.exp}.log'
    logf = open(log_path, 'w')
    for epoch in range(args.epochs):
        for batch in train_loader:
            char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids = \
                [x.to(device, non_blocking=True) for x in batch[:7]]
            if len(batch) == 9:
                w = batch[7].to(device, non_blocking=True)
                t_probs = batch[8].to(device, non_blocking=True)
            elif len(batch) == 8:
                w = batch[7].to(device, non_blocking=True) if batch[7].dim() == 2 else None
                t_probs = batch[7].to(device, non_blocking=True) if batch[7].dim() == 3 else None
            else:
                w = None
                t_probs = None
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            # R-Drop: two forward passes
            import contextlib as _cl
            amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else _cl.nullcontext()
            with amp_ctx:
                logp1, pos_logits1, _, logits1, extra1 = model.forward_with_ids(char_ids, poly_ids, positions, pos_mask, cand_mask)
                if args.rdrop > 0:
                    logp2, pos_logits2, _, logits2, _ = model.forward_with_ids(char_ids, poly_ids, positions, pos_mask, cand_mask)
                    loss = masked_ce(logp1.float(), label_ids, pos_mask, cand_mask, args.label_smooth, w)
                    loss = loss + masked_ce(logp2.float(), label_ids, pos_mask, cand_mask, args.label_smooth, w)
                    loss = 0.5 * loss + args.rdrop * 0.5 * kl_between(logp1.float(), logp2.float(), pos_mask)
                    pos_loss = F.cross_entropy(pos_logits1.transpose(1, 2), pos_ids,
                                               reduction='none')[pos_mask].mean() \
                               + F.cross_entropy(pos_logits2.transpose(1, 2), pos_ids,
                                                 reduction='none')[pos_mask].mean()
                    loss = loss + args.pos_weight * 0.5 * pos_loss
                    kd_logits = (logits1, logits2)
                else:
                    loss = masked_ce(logp1.float(), label_ids, pos_mask, cand_mask, args.label_smooth, w)
                    pos_loss = F.cross_entropy(pos_logits1.transpose(1, 2), pos_ids,
                                               reduction='none')[pos_mask].mean()
                    loss = loss + args.pos_weight * pos_loss
                    kd_logits = (logits1,)
                if args.tone_aux and extra1 is not None:
                    syl_logits, tone_logits = extra1
                    syl_t = torch.tensor([syl_of.get(int(x), 0) for x in label_ids[pos_mask].tolist()], device=device)
                    tone_t = torch.tensor([tone_of.get(int(x), 0) for x in label_ids[pos_mask].tolist()], device=device)
                    sl = F.cross_entropy(syl_logits.transpose(1, 2)[:, :, :][pos_mask] if False else syl_logits[pos_mask], syl_t)
                    tl = F.cross_entropy(tone_logits[pos_mask], tone_t)
                    loss = loss + args.tone_w * (sl + tl)
                if t_probs is not None:
                    T = args.kd_T
                    qt = t_probs.clamp_min(1e-8).pow(1.0 / T) * cand_mask
                    qt = qt / qt.sum(-1, keepdim=True).clamp_min(1e-8)
                    kd = None
                    for logits in kd_logits:
                        logq = F.log_softmax(logits.float() / T, dim=-1)
                        term = (qt * (qt.clamp_min(1e-8).log() - logq)).sum(-1)
                        term = (term * pos_mask).sum() / pos_mask.sum().clamp(min=1)
                        kd = term if kd is None else kd + term
                    loss = loss + args.kd_alpha * (T * T) * 0.5 * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step += 1
            if step % 200 == 0:
                msg = (f'[step {step}/{total_steps}] loss={loss.item():.4f} '
                       f'lr={lr_at(step):.2e} elapsed={time.time()-t0:.0f}s')
                print(msg)
                logf.write(msg + '\n'); logf.flush()
            if step % args.eval_every == 0:
                m = evaluate(model, dev_loader, char_cand_mask, n_labels, device)
                me = evaluate(ema.shadow, dev_loader, char_cand_mask, n_labels, device)
                msg = (f'[eval {step}] dev: acc={m["acc"]:.4f}/{m["avg_p"]:.4f}/{m["avg_pp"]:.4f} | '
                       f'ema-dev: acc={me["acc"]:.4f}/{me["avg_p"]:.4f}/{me["avg_pp"]:.4f}')
                print(msg); logf.write(msg + '\n'); logf.flush()
                if me['acc'] > best:
                    best = me['acc']
                    torch.save(ema.shadow.state_dict(), f'{args.data_root}/{args.exp}_best.pt')

    me = evaluate(ema.shadow, test_loader, char_cand_mask, n_labels, device)
    mt = evaluate(model, test_loader, char_cand_mask, n_labels, device)
    msg = f'FINAL test: ema acc={me["acc"]:.4f}/{me["avg_p"]:.4f}/{me["avg_pp"]:.4f} | raw acc={mt["acc"]:.4f}'
    print(msg); logf.write(msg + '\n')
    torch.save(ema.shadow.state_dict(), f'{args.data_root}/{args.exp}_final.pt')
    logf.close()


if __name__ == '__main__':
    main()
