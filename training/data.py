"""Sentence-level dataset format for the student model.

One record = one sentence: all labeled polyphone positions in one sample.
Serialized to pickle: lists of (char_ids, positions, label_ids, pos_ids).
"""
import argparse
import pickle
from collections import Counter

ANCHOR = '\u2581'
POS_TAGS = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']
MAX_LEN = 128


def load_labels(path):
    """(char, pinyin) candidate table -> label space, char candidates, poly chars"""
    pairs = [l.rstrip('\n').split('\t') for l in open(path, encoding='utf-8')]
    labels = sorted({p for _, p in pairs})
    label_id = {l: i for i, l in enumerate(labels)}
    char_cands = {}
    for ch, p in pairs:
        char_cands.setdefault(ch, set()).add(label_id[p])
    poly_chars = sorted(char_cands.keys())
    poly_id = {c: i for i, c in enumerate(poly_chars)}
    return labels, label_id, char_cands, poly_chars, poly_id


def read_samples(sent_path, lb_path, pos_path):
    for s, l, p in zip(open(sent_path, encoding='utf-8'),
                       open(lb_path, encoding='utf-8'),
                       open(pos_path, encoding='utf-8')):
        s = s.rstrip('\n')
        if not s:
            continue
        i = s.find(ANCHOR)
        yield s.replace(ANCHOR, ''), i, l.rstrip('\n'), p.rstrip('\n')


def build_split(out_path, sent_path, lb_path, pos_path, char_to_id, label_id, char_cands, poly_id, max_len=MAX_LEN):
    """group single-anchor samples into sentence records"""
    by_sent = {}
    for text, i, lb, pos in read_samples(sent_path, lb_path, pos_path):
        rec = by_sent.setdefault(text, {})
        if i < 0 or i >= len(text):
            continue
        rec[i] = (lb, pos)

    records = []
    dropped_pos = 0
    for text, posmap in by_sent.items():
        if not posmap:
            continue
        # truncate long sentences: window centered on first labeled position
        if len(text) > max_len:
            first = sorted(posmap)[0]
            start = max(0, min(first - max_len // 2, len(text) - max_len))
            window = text[start:start + max_len]
            newmap = {i - start: v for i, v in posmap.items() if start <= i < start + max_len}
            dropped_pos += len(posmap) - len(newmap)
            if not newmap:
                continue
            text, posmap = window, newmap
        char_ids = [char_to_id.get(c, 1) for c in text]  # 1 = UNK
        positions, label_ids, pos_ids, poly_ids = [], [], [], []
        for i in sorted(posmap):
            lb, pos = posmap[i]
            if lb not in label_id:
                continue
            ch = text[i]
            if ch not in poly_id:
                continue
            positions.append(i)
            label_ids.append(label_id[lb])
            pos_ids.append(POS_TAGS.index(pos) if pos in POS_TAGS else 0)
            poly_ids.append(poly_id[ch])
        if positions:
            records.append((char_ids, positions, label_ids, pos_ids, poly_ids))
    pickle.dump(records, open(out_path, 'wb'))
    n_pos = sum(len(r[3 - 1]) for r in records)  # label_ids
    print(f'{out_path}: {len(records)} sentences, {n_pos} labeled positions (dropped {dropped_pos} out-of-window positions)')
    return records


def build_vocab(train_sent_path, extra_paths=()):
    chars = set()
    for path in [train_sent_path] + list(extra_paths):
        for s in open(path, encoding='utf-8'):
            chars.update(s.replace(ANCHOR, '').rstrip('\n'))
    vocab = ['<pad>', '<unk>'] + sorted(chars)
    return {c: i for i, c in enumerate(vocab)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets')
    ap.add_argument('--poly_chars', default='datasets/POLYPHONIC_CHARS.txt')
    ap.add_argument('--out', default='datasets/student')
    args = ap.parse_args()
    root = args.root

    labels, label_id, char_cands, poly_chars, poly_id = load_labels(args.poly_chars)
    print(f'label space: {len(labels)} pinyins; poly chars: {len(poly_chars)}')

    vocab = build_vocab(f'{root}/merged/train.sent')
    print(f'char vocab: {len(vocab)}')
    # sanity: every poly char + every train sentence char covered by design (UNK=1 fallback)
    missing = [c for c in poly_chars if c not in vocab]
    print('poly chars missing from vocab:', missing[:10] if missing else 'none')

    import os
    os.makedirs(args.out, exist_ok=True)
    pickle.dump({'vocab': vocab, 'labels': labels, 'poly_chars': poly_chars,
                 'char_cands': char_cands},
                open(f'{args.out}/meta.pkl', 'wb'))

    build_split(f'{args.out}/train.pkl', f'{root}/merged/train.sent',
                f'{root}/merged/train.lb', f'{root}/merged/train.pos',
                vocab, label_id, char_cands, poly_id)
    build_split(f'{args.out}/dev.pkl', f'{root}/cvte/dev.sent',
                f'{root}/cvte/dev.lb', f'{root}/cvte/dev.pos',
                vocab, label_id, char_cands, poly_id)
    build_split(f'{args.out}/test.pkl', f'{root}/cpp/test.sent',
                f'{root}/cpp/test.lb', f'{root}/cpp/test.pos',
                vocab, label_id, char_cands, poly_id)


if __name__ == '__main__':
    main()
