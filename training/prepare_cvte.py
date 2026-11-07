"""Expand CVTE-poly sentence-level annotations into g2pW per-(sentence, char) samples.

CVTE-poly raw format (one line = one sentence):
  train.sent:  all chars wrapped with ▁, e.g. '看▁重▁新技术'  (▁ marks ANY char)
  train.lb:    space-joined pinyin labels, only for SOME polyphonic anchors, in order
  train.pos:   space-joined POS tags, aligned 1:1 with the labels

g2pW expects one sample per (sentence, target char):
  train.sent:  sentence with a single ▁ right before the target char
  train.lb:    single pinyin
  train.pos:   single POS tag

Alignment rule: assign each label to the earliest not-yet-consumed polyphonic
anchor whose candidate set (from POLYPHONIC_CHARS.txt) contains the label.
Lines with no valid alignment are dropped as label noise.
"""
import argparse
import random
import re
from collections import Counter

ANCHOR = '\u2581'  # ▁


def load_candidates(path):
    poly = {}
    for line in open(path, encoding='utf-8'):
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 2:
            poly.setdefault(parts[0], set()).add(parts[1])
    return poly


def read_lines(path):
    return open(path, encoding='utf-8').read().rstrip('\n').split('\n')


def find_anchors(sent):
    """positions of the char right after each ▁ marker"""
    return [j + 1 for j, c in enumerate(sent) if c == ANCHOR and j + 1 < len(sent)]


def align(anchors_chars, labels, poly, truly_poly):
    """greedy monotone alignment with skips. Returns list of (anchor_idx, label) or None."""
    out = []
    ai = 0
    for lb in labels:
        found = False
        while ai < len(anchors_chars):
            c = anchors_chars[ai]
            ai += 1
            if c in truly_poly and lb in poly.get(c, ()):
                out.append((ai - 1, lb))  # ai-1 = index into anchors list
                found = True
                break
        if not found:
            return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/root/.tmp/polyphone/data/CVTE-poly')
    ap.add_argument('--poly_chars', default='/root/.tmp/polyphone/data/POLYPHONIC_CHARS.txt')
    ap.add_argument('--dst', default='/root/g2pW/datasets/cvte')
    ap.add_argument('--dev_ratio', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=1313)
    args = ap.parse_args()

    poly = load_candidates(args.poly_chars)
    truly_poly = {c for c, v in poly.items() if len(v) > 1}

    sents = read_lines(f'{args.src}/train.sent')
    lbs = read_lines(f'{args.src}/train.lb')
    poses = read_lines(f'{args.src}/train.pos')
    assert len(sents) == len(lbs) == len(poses)

    samples = []  # (sent_with_single_anchor, label, pos)
    stats = Counter()
    for sent, lb, pos in zip(sents, lbs, poses):
        anchor_positions = find_anchors(sent)
        anchor_chars = [sent[p] for p in anchor_positions]
        labels = lb.strip().split(' ') if lb.strip() else []
        pos_tags = re.split(r'[ \t]+', pos.strip()) if pos.strip() else []
        if len(labels) != len(pos_tags):
            stats['pos_count_mismatch'] += 1
            continue
        mapping = align(anchor_chars, labels, poly, truly_poly)
        if mapping is None:
            stats['unalignable'] += 1
            continue
        for k, (a_idx, label) in enumerate(mapping):
            # rebuild sentence with a single ▁ before the target anchor char
            # walk the original string, skipping ▁ chars, insert marker at the target char
            ap = anchor_positions[a_idx]
            out = []
            for j, ch in enumerate(sent):
                if ch == ANCHOR:
                    continue
                if j == ap:
                    out.append(ANCHOR)
                out.append(ch)
            samples.append((''.join(out), label, pos_tags[k]))
        stats['lines_used'] += 1

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_dev = int(len(samples) * args.dev_ratio)
    splits = {
        'train': samples[n_dev:],
        'dev': samples[:n_dev],
    }

    import os
    os.makedirs(args.dst, exist_ok=True)
    for split, items in splits.items():
        with open(f'{args.dst}/{split}.sent', 'w', encoding='utf-8') as fs, \
             open(f'{args.dst}/{split}.lb', 'w', encoding='utf-8') as fl, \
             open(f'{args.dst}/{split}.pos', 'w', encoding='utf-8') as fp:
            for sent, label, pos in items:
                fs.write(sent + '\n')
                fl.write(label + '\n')
                fp.write(pos + '\n')
        print(f'{split}: {len(items)} samples')

    print('stats:', dict(stats))
    print('total samples:', len(samples))
    # sanity: uniqueness of target char per sample and label validity
    bad = 0
    checked = 0
    for sent, label, _ in samples[:50000]:
        i = sent.find(ANCHOR)
        if i == -1 or i + 1 >= len(sent):
            bad += 1
            continue
        checked += 1
        c = sent[i + 1]
        if label not in poly.get(c, ()):
            bad += 1
    print(f'bad samples (first 50k): {bad} (checked {checked})')


if __name__ == '__main__':
    main()
