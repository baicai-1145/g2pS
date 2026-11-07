"""Generate BERT teacher logits for all train records (aligned to train.pkl order).

Output: fp16 probs (N_positions, n_labels) + offsets per record.
"""
import pickle
import sys

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
    cfg = load_config(f'{TEACHER_DIR}/config.py', use_default=True)
    tok = BertTokenizer.from_pretrained(cfg.model_source)
    poly = [l.split('\t') for l in open(cfg.polyphonic_chars_path, encoding='utf-8').read().strip().split('\n')]
    labels, char2ph = get_phoneme_labels(poly)
    labels_sorted = sorted(labels)
    assert labels == labels_sorted
    label_id = {l: i for i, l in enumerate(labels)}
    chars = sorted(char2ph.keys())
    n_labels = len(labels)
    print('teacher label space:', n_labels)

    # our records define (sentence, positions) pairs in order
    records = pickle.load(open('datasets/student/train.pkl', 'rb'))
    texts, qids = [], []
    for cids, positions, *_ in records:
        inv = {v: k for k, v in pickle.load(open('datasets/student/meta.pkl', 'rb'))['vocab'].items()} if False else None
        texts.append(None)  # placeholder, replaced below
        qids.append(None)

    # rebuild texts from vocab
    meta = pickle.load(open('datasets/student/meta.pkl', 'rb'))
    vocab = meta['vocab']
    inv_vocab = {v: k for k, v in vocab.items()}
    flat_texts, flat_qids, flat_owner = [], [], []
    for ri, (cids, positions, *_ ) in enumerate(records):
        text = ''.join(inv_vocab.get(c, '') for c in cids)
        for p in positions:
            flat_texts.append(text)
            flat_qids.append(p)
            flat_owner.append(ri)

    ds = TextDataset(tok, labels, char2ph, chars, flat_texts, flat_qids,
                     use_mask=True, use_pos=False, window_size=32, for_train=False)
    dl = DataLoader(ds, batch_size=512, collate_fn=ds.create_mini_batch, num_workers=8)

    model = G2PW.from_pretrained(cfg.model_source, labels=labels, chars=chars,
                                 pos_tags=TextDataset.POS_TAGS,
                                 use_conditional=cfg.use_conditional,
                                 param_conditional=cfg.param_conditional,
                                 use_pos=cfg.use_pos, param_pos=cfg.param_pos)
    model.load_state_dict(torch.load(f'{TEACHER_DIR}/best_accuracy.pth', map_location='cuda'))
    model.to('cuda').eval()

    all_probs = []
    with torch.no_grad():
        for data in tqdm(dl, desc='teacher'):
            x = {k: data[k].to('cuda') for k in
                 ('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids')}
            probs = model(**x)
            all_probs.append(probs.cpu().to(torch.float16))

    probs = torch.cat(all_probs)  # (N_pos, n_labels) fp16
    # offsets: count positions per record in order
    offsets = torch.zeros(len(records) + 1, dtype=torch.long)
    for ri in flat_owner:
        offsets[ri + 1] += 1
    offsets = offsets.cumsum(0)
    assert offsets[-1].item() == probs.shape[0]

    torch.save({'probs': probs, 'offsets': offsets}, 'datasets/student/teacher_probs.pt')
    print('saved', probs.shape)


if __name__ == '__main__':
    main()
