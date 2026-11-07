"""Build v5 dataset: cut vocab tail (A1) and remap all records.

Keep: 852 poly chars + context chars by frequency, total ~6000.
Output: datasets/student_v5/{meta.pkl, train_full.pkl, dev.pkl, test.pkl}
Records with KD 6-tuples (dense BERT probs) are preserved untouched.
"""
import array
import pickle
from collections import Counter

import numpy as np

SRC = 'datasets/student_v4'
DST = 'datasets/student_v5'
TARGET_VOCAB = 6000


def remap_records(records, old2new):
    out = []
    for r in records:
        cids = array.array('h', old2new[np.asarray(r[0], dtype=np.int16)])
        out.append((cids,) + tuple(r[1:]))
    return out


def main():
    import os
    os.makedirs(DST, exist_ok=True)
    meta = pickle.load(open(f'{SRC}/meta.pkl', 'rb'))
    vocab, labels, poly_chars, char_cands = meta['vocab'], meta['labels'], meta['poly_chars'], meta['char_cands']
    train = pickle.load(open(f'{SRC}/train_full.pkl', 'rb'))
    dev = pickle.load(open(f'{SRC}/dev.pkl', 'rb'))
    test = pickle.load(open(f'{SRC}/test.pkl', 'rb'))

    freq = Counter()
    for r in train:
        freq.update(np.asarray(r[0], dtype=np.int16).tolist())
    inv_vocab = {v: k for k, v in vocab.items()}
    poly_ids = {vocab[c] for c in poly_chars if c in vocab}
    # rank context chars by freq
    ctx = [(c, n) for c, n in freq.items() if c not in poly_ids and c > 1]
    ctx.sort(key=lambda x: -x[1])
    keep = [c for c, _ in ctx[:TARGET_VOCAB - 2 - len(poly_ids)]]
    keep_set = set(keep) | poly_ids | {0, 1}  # pad, unk always

    keep_chars = [inv_vocab[i] for i in sorted(keep_set - {0, 1})]
    new_vocab = {ch: i for i, ch in enumerate(['<pad>', '<unk>'] + keep_chars)}
    old2new = np.zeros(len(vocab), dtype=np.int16)
    for ch, new_i in new_vocab.items():
        old2new[vocab[ch]] = new_i
    print(f'vocab: {len(vocab)} -> {len(new_vocab)}')
    cov = sum(n for c, n in freq.items() if c in keep_set) / sum(freq.values())
    print(f'token coverage kept: {cov*100:.3f}%')

    meta2 = {'vocab': new_vocab, 'labels': labels, 'poly_chars': poly_chars, 'char_cands': char_cands}
    pickle.dump(meta2, open(f'{DST}/meta.pkl', 'wb'))
    pickle.dump(remap_records(train, old2new), open(f'{DST}/train_full.pkl', 'wb'), protocol=4)
    pickle.dump(remap_records(dev, old2new), open(f'{DST}/dev.pkl', 'wb'))
    pickle.dump(remap_records(test, old2new), open(f'{DST}/test.pkl', 'wb'))
    print('saved v5')


if __name__ == '__main__':
    main()
