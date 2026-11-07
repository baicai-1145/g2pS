"""Bulk pseudo-label the external corpus with the BERT g2pw teacher.

For each sentence: find positions of chars in the TEACHER's table (552 chars),
run teacher (window 32 per position, batched), store argmax/prob/top2 in shards.
Chars only in the merged table (Xinhua-only) are NOT labeled here (LLM's job).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

sys.path.insert(0, './')
from g2pw.dataset import TextDataset, get_phoneme_labels
from g2pw.module import G2PW
from g2pw.utils import load_config

TEACHER_DIR = 'saved_models/OURS_BERT_M_DescWS-Sec-cLin-B_POSw01'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sent_path', default='data/external/sents.txt')
    ap.add_argument('--out_dir', default='data/external/teacher')
    ap.add_argument('--shard_size', type=int, default=1_000_000)
    ap.add_argument('--start_shard', type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(f'{TEACHER_DIR}/config.py', use_default=True)
    tok = BertTokenizer.from_pretrained(cfg.model_source)
    poly = [l.split('\t') for l in open(cfg.polyphonic_chars_path, encoding='utf-8').read().strip().split('\n')]
    labels, char2ph = get_phoneme_labels(poly)
    chars = sorted(char2ph.keys())
    teacher_chars = set(chars)
    os.makedirs(args.out_dir, exist_ok=True)

    # load sentences, find teacher-covered positions
    sents = open(args.sent_path, encoding='utf-8').read().rstrip('\n').split('\n')
    sents = [s.lower() for s in sents]  # TextDataset lowercases; pre-lower to keep positions aligned
    print('sentences:', len(sents))

    flat_texts, flat_qids, owner = [], [], []
    for si, s in enumerate(sents):
        for i, c in enumerate(s):
            if c in teacher_chars:
                flat_texts.append(s)
                flat_qids.append(i)
                owner.append(si)
    print('teacher-covered positions:', len(flat_texts))
    # defensive: drop any pair whose char is missing from the candidate dict (paranoia)
    keep = [k for k, (s, i) in enumerate(zip(flat_texts, flat_qids)) if s[i] in char2ph]
    if len(keep) != len(flat_texts):
        print('dropped misaligned pairs:', len(flat_texts) - len(keep))
        flat_texts = [flat_texts[k] for k in keep]
        flat_qids = [flat_qids[k] for k in keep]

    model = G2PW.from_pretrained(cfg.model_source, labels=labels, chars=chars,
                                 pos_tags=TextDataset.POS_TAGS,
                                 use_conditional=cfg.use_conditional,
                                 param_conditional=cfg.param_conditional,
                                 use_pos=cfg.use_pos, param_pos=cfg.param_pos)
    model.load_state_dict(torch.load(f'{TEACHER_DIR}/best_accuracy.pth', map_location='cuda'))
    model.to('cuda').eval()

    ds = TextDataset(tok, labels, char2ph, chars, flat_texts, flat_qids,
                     use_mask=True, use_pos=False, window_size=32, for_train=False)
    dl = DataLoader(ds, batch_size=512, collate_fn=ds.create_mini_batch, num_workers=8)

    n = len(flat_texts)
    argmax_all = np.zeros(n, dtype=np.int16)
    top2_all = np.zeros((n, 2), dtype=np.float16)
    with torch.no_grad():
        for bi, data in enumerate(tqdm(dl, desc='teacher-bulk')):
            x = {k: data[k].to('cuda') for k in
                 ('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids')}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                probs = model(**x)
            tp2, tam = probs.float().topk(2, dim=-1)
            argmax_all[bi * 512:bi * 512 + tam.shape[0]] = tam[:, 0].cpu().numpy()
            top2_all[bi * 512:bi * 512 + tam.shape[0]] = tp2.cpu().numpy().astype(np.float16)

    # group by sentence: offsets + per-sentence arrays
    counts = np.zeros(len(sents) + 1, dtype=np.int64)
    for si in owner:
        counts[si + 1] += 1
    offsets = counts.cumsum()
    # positions in flat order match sentence scan order
    pos_list = np.array(flat_qids, dtype=np.int16)
    np.savez_compressed(f'{args.out_dir}/bulk.npz',
                        offsets=offsets, positions=pos_list,
                        argmax=argmax_all, top2=top2_all,
                        labels=np.array(labels))
    print('saved', f'{args.out_dir}/bulk.npz')


if __name__ == '__main__':
    main()
