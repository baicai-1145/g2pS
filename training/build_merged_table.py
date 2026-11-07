"""Rebuild the polyphone candidate table: merge Xinhua dict + CVTE/CEDICT table.

Conventions:
- ü notation unified to 'u:' (Xinhua uses 'v')
- keep light tone '5' (CVTE convention; Xinhua lacks them)
- output sorted (char, pinyin) pairs
"""
import argparse

def load(path):
    d = {}
    for line in open(path, encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) == 2:
            py = p[1].strip().replace('v', 'u:')
            if py:
                d.setdefault(p[0], set()).add(py)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xinhua', default='新华字典/POLYPHONIC_CHARS.txt')
    ap.add_argument('--cvte', default='datasets/POLYPHONIC_CHARS.txt')
    ap.add_argument('--out', default='datasets/POLYPHONIC_CHARS_MERGED.txt')
    args = ap.parse_args()

    xh = load(args.xinhua)
    ct = load(args.cvte)

    merged = {}
    src = {}
    for ch, ps in ct.items():
        merged.setdefault(ch, set()).update(ps)
        src.setdefault(ch, set()).update('C' for _ in ps)
    for ch, ps in xh.items():
        merged.setdefault(ch, set()).update(ps)
        src.setdefault(ch, set()).update('X' for _ in ps)

    both = sum(1 for ch in merged if ch in xh and ch in ct)
    only_x = sum(1 for ch in merged if ch in xh and ch not in ct)
    only_c = sum(1 for ch in merged if ch in ct and ch not in xh)
    n_pairs = sum(len(v) for v in merged.values())

    with open(args.out, 'w', encoding='utf-8') as f:
        for ch in sorted(merged):
            for py in sorted(merged[ch]):
                f.write(f'{ch}\t{py}\n')

    labels = sorted({p for v in merged.values() for p in v})
    poly = {c for c, v in merged.items() if len(v) > 1}
    print(f'merged: {len(merged)} chars, {n_pairs} pairs, {len(poly)} true polyphones, {len(labels)} distinct pinyins')
    print(f'  both: {both} | xinhua-only chars: {only_x} | cvte-only chars: {only_c}')
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
