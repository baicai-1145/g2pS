"""Generate teacher-assistant (v4_15m) top-8 soft labels for ALL v4 train_full positions."""
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '.')
from student.modules import StudentG2P
from student.train import collate, RecordDataset

SRC = 'datasets/student_v4'
OUT = 'datasets/student_v5/ta_top8.npz'
BATCH = 256


def main():
    meta = pickle.load(open(f'{SRC}/meta.pkl', 'rb'))
    vocab, labels, poly_chars, char_cands = meta['vocab'], meta['labels'], meta['poly_chars'], meta['char_cands']
    n_labels = len(labels)
    char_cand_mask = {i: torch.zeros(n_labels, dtype=torch.bool).index_fill_(0, torch.tensor(sorted(char_cands[ch])), True)
                      for i, ch in enumerate(poly_chars)}
    model = StudentG2P(len(vocab), n_labels, len(poly_chars))
    model.load_state_dict(torch.load('datasets/student_v4/v4_15m_best.pt', map_location='cuda'))
    model.to('cuda').eval()

    train = pickle.load(open(f'{SRC}/train_full.pkl', 'rb'))
    print('records:', len(train))
    dl = DataLoader(RecordDataset(train), batch_size=BATCH, shuffle=False,
                    collate_fn=lambda b: collate(b, n_labels, char_cand_mask), num_workers=6)
    n_pos = sum(len(r[1]) for r in train)
    idx_all = np.zeros((n_pos, 8), dtype=np.int16)
    val_all = np.zeros((n_pos, 8), dtype=np.float16)
    offsets = np.zeros(len(train) + 1, dtype=np.int64)
    for i, r in enumerate(train):
        offsets[i + 1] = offsets[i] + len(r[1])
    p = 0
    with torch.no_grad():
        for b in tqdm(dl, desc='ta'):
            char_ids, poly_ids, positions, pos_mask, cand_mask, label_ids, pos_ids = [x.to("cuda") for x in b[:7]]
            logp, _, _, _ = model.forward_with_ids(char_ids, poly_ids, positions, pos_mask, cand_mask)
            probs = logp.float().exp()
            v, ix = probs.topk(8, dim=-1)
            bs = pos_mask.sum(1)
            for bi in range(ix.shape[0]):
                k = int(bs[bi])
                if k:
                    idx_all[p:p + k] = ix[bi, :k].cpu().numpy()
                    val_all[p:p + k] = v[bi, :k].cpu().numpy().astype(np.float16)
                    p += k
    assert p == n_pos
    np.savez(OUT, offsets=offsets, idx=idx_all, val=val_all)
    print('saved', OUT, idx_all.shape)


if __name__ == '__main__':
    main()
