"""Assemble the v4 mega distillation dataset.

Sources:
  - gold:   datasets/student_v4/train.pkl (6-tuples w/ teacher KD probs, 887-label space)
  - wyw:    datasets/wyw/train.* classical gold (5-tuples)
  - bulk:   data/external + teacher/bulk.npz, top1>=TH hard labels (5-tuples)
  - llm:    data/external/llm_newchar.jsonl + pypinyin-default expansion (5-tuples)
Output: datasets/student_v4/train_full.pkl  (list of records)
"""
import array
import json
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, '.')
TH = 0.97
MAX_LEN = 128
BULK_FRAC = 0.4


def main():
    meta = pickle.load(open('datasets/student_v4/meta.pkl', 'rb'))
    vocab, labels, poly_chars, char_cands = meta['vocab'], meta['labels'], meta['poly_chars'], meta['char_cands']
    label_id = {l: i for i, l in enumerate(labels)}
    poly_id = {c: i for i, c in enumerate(poly_chars)}
    rng = random.Random(1313)

    # 1) gold (already 6-tuples with KD probs)
    gold = pickle.load(open('datasets/student_v4/train.pkl', 'rb'))
    n_gold_pos = sum(len(r[1]) for r in gold)
    print(f'gold: {len(gold)} sents, {n_gold_pos} positions')

    # 2) wyw
    wyw = []
    A = '\u2581'
    for s, l, p in zip(open('datasets/wyw/train.sent', encoding='utf-8'),
                       open('datasets/wyw/train.lb', encoding='utf-8'),
                       open('datasets/wyw/train.pos', encoding='utf-8')):
        s = s.rstrip('\n')
        i = s.find(A)
        if i < 0:
            continue
        text = s.replace(A, '')
        c = text[i]
        lb = l.strip()
        if c not in poly_id or lb not in label_id:
            continue
        wyw.append((text, i, lb))
    records = []

    def to_record(text, posmap):
        # posmap: {pos: label_id}; truncate>MAX_LEN centered on first pos
        if len(text) > MAX_LEN:
            first = sorted(posmap)[0]
            start = max(0, min(first - MAX_LEN // 2, len(text) - MAX_LEN))
            window = text[start:start + MAX_LEN]
            posmap = {i - start: v for i, v in posmap.items() if start <= i < start + MAX_LEN}
            if not posmap:
                return None
            text = window
        cids = array.array('h', [vocab.get(ch, 1) for ch in text])
        ps = sorted(posmap)
        return (cids, ps, [posmap[p] for p in ps], [0] * len(ps), [poly_id[text[p]] for p in ps])

    for text, i, lb in wyw:
        r = to_record(text, {i: label_id[lb]})
        if r:
            records.append(r)
    n_wyw_pos = sum(len(r[1]) for r in records)
    print(f'wyw: {len(records)} sents, {n_wyw_pos} positions')

    # 3) bulk teacher labels
    bulk = np.load('data/external/teacher/bulk.npz', allow_pickle=True)
    offsets, positions = bulk['offsets'], bulk['positions']
    argmax, top2 = bulk['argmax'], bulk['top2']
    t_labels = [str(x) for x in bulk['labels']]
    sents = open('data/external/sents.txt', encoding='utf-8').read().rstrip('\n').split('\n')
    sents = [s.lower() for s in sents]
    assert len(offsets) - 1 == len(sents)

    n_bulk_pos = 0
    n_bulk_sents = 0
    for si, s in enumerate(sents):
        a, b = offsets[si], offsets[si + 1]
        if a == b:
            continue
        conf = top2[a:b, 0].astype(np.float32)
        keep = (conf >= TH) & (rng.random() < BULK_FRAC)
        if not keep.any():
            continue
        posmap = {}
        for k in np.nonzero(keep)[0]:
            gi = a + k
            py = t_labels[int(argmax[gi])]
            if py not in label_id:
                continue
            posmap[int(positions[gi])] = label_id[py]
        if not posmap:
            continue
        r = to_record(s, posmap)
        if r:
            records.append(r)
            n_bulk_pos += len(r[1])
            n_bulk_sents += 1
    print(f'bulk: {n_bulk_sents} sents, {n_bulk_pos} positions (TH={TH})')

    # 4) LLM new-char + pypinyin expansion
    try:
        rows = [json.loads(l) for l in open('data/external/llm_newchar.jsonl', encoding='utf-8')]
    except FileNotFoundError:
        rows = []
    import pypinyin
    n_llm_pos = n_pyp_pos = 0
    if rows:
        poly_t = set(l.split('\t')[0] for l in open('datasets/POLYPHONIC_CHARS.txt', encoding='utf-8'))
        new_chars = {c for c in poly_id if c not in poly_t}
        from collections import defaultdict, Counter
        by_char_llm = defaultdict(Counter)
        seen = set()
        for r in rows:
            ch = r['char']
            if ch in new_chars:
                by_char_llm[ch][r['pinyin']] += 1
                seen.add((r['sent'], r['pos']))
        # LLM rows -> records (sentence may hold several targets; group)
        per_sent = defaultdict(dict)
        for r in rows:
            if r['char'] in new_chars:
                per_sent[r['sent']][r['pos']] = label_id.get(r['pinyin'])
        for s, pm in per_sent.items():
            pm = {p: v for p, v in pm.items() if v is not None}
            if pm:
                rec = to_record(s, pm)
                if rec:
                    records.append(rec)
                    n_llm_pos += len(rec[1])
        # pypinyin default expansion for high-agreement chars
        def py_default(ch):
            out = pypinyin.pinyin(ch, style=pypinyin.Style.TONE3, heteronym=False)
            if not out or not out[0]:
                return None
            s = out[0][0].replace('v', 'u:')
            # strip 'ü' style
            s = s.replace('ü', 'u:')
            return s

        agree_stats = {}
        for ch, cnt in by_char_llm.items():
            total = sum(cnt.values())
            d = py_default(ch)
            if d and total >= 30:
                agree_stats[ch] = cnt[d] / total
        expand_chars = {c for c, a in agree_stats.items() if a >= 0.98}
        print(f'llm rows: {len(rows)} | chars sampled>=30: {len(agree_stats)} | pypinyin-agree>=98%: {len(expand_chars)}')
        # extra pypinyin rows for expand_chars (cap 2000/char), skip LLM-seen
        from collections import defaultdict as dd
        pool = dd(list)
        for si, s in enumerate(sents):
            for i, c in enumerate(s):
                if c in expand_chars:
                    pool[c].append((s, i))
        for ch in expand_chars:
            d = py_default(ch)
            lid = label_id.get(d)
            if lid is None:
                continue
            extra = 0
            rng.shuffle(pool[ch])
            per_sent_local = dd(dict)
            for s, i in pool[ch]:
                if (s, i) in seen or extra >= 2000:
                    continue
                per_sent_local[s][i] = lid
                extra += 1
            for s, pm in per_sent_local.items():
                rec = to_record(s, pm)
                if rec:
                    records.append(rec)
                    n_pyp_pos += len(rec[1])
    print(f'llm-labeled: {n_llm_pos} positions | pypinyin-expanded: {n_pyp_pos} positions')

    rng.shuffle(records)
    all_records = gold + records  # gold keep KD 6-tuples; rest hard 5-tuples
    rng.shuffle(all_records)
    pickle.dump(all_records, open('datasets/student_v4/train_full.pkl', 'wb'), protocol=4)
    n_pos = sum(len(r[1]) for r in all_records)
    print(f'TOTAL: {len(all_records)} records, {n_pos} positions -> datasets/student_v4/train_full.pkl')


if __name__ == '__main__':
    main()
