"""Calibrate LLM labeler on cvte/dev gold: accuracy overall / by position count / parse stats."""
import asyncio
import json
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, 'student')
from llm_label import LLMLabeler

def main(n_limit=0):
    meta = pickle.load(open('datasets/student/meta.pkl', 'rb'))
    vocab, labels, poly_chars = meta['vocab'], meta['labels'], meta['poly_chars']
    inv_vocab = {v: k for k, v in vocab.items()}
    cands = {}
    for line in open('datasets/POLYPHONIC_CHARS_MERGED.txt', encoding='utf-8'):
        c, p = line.rstrip('\n').split('\t')
        cands.setdefault(c, set()).add(p.replace('v', 'u:'))

    dev = pickle.load(open('datasets/student/dev.pkl', 'rb'))
    items = []
    for cids, positions, label_ids, pos_ids, poly_ids in dev:
        text = ''.join(inv_vocab.get(c, '') for c in cids)
        targets = []
        for p, lid, pid in zip(positions, label_ids, poly_ids):
            ch = poly_chars[pid]
            targets.append((p, ch, sorted(cands.get(ch, {labels[lid]}))))
        gold = {p: labels[lid] for p, lid in zip(positions, label_ids)}
        items.append((text, targets, gold))
    if n_limit:
        items = items[:n_limit]

    async def run():
        lab = LLMLabeler(concurrency=96)
        t0 = time.time()
        results = await asyncio.gather(*[
            lab.label_sentence(t, tg) for t, tg, _ in items
        ])
        dt = time.time() - t0
        n_ok = n_err = n_fail = 0
        correct = 0
        by_char = defaultdict(lambda: [0, 0])
        for (text, targets, gold), res in zip(items, results):
            if res is None:
                n_err += 1
                continue
            if res == {}:
                n_fail += 1
                continue
            for p, ch, c in targets:
                if p in res:
                    hit = res[p] == gold.get(p)
                    correct += hit
                    n_ok += 1
                    by_char[ch][0] += hit
                    by_char[ch][1] += 1
        n_pos = sum(len(tg) for _, tg, _ in items)
        print(f'calls={lab.n_call} ok={lab.n_ok} parse_fail={lab.n_parse_fail} error={lab.n_error} | wall={dt:.0f}s ({lab.n_call/max(dt,1):.1f} call/s)')
        print(f'answered positions: {n_ok}/{n_pos} ({n_ok/n_pos*100:.1f}%), LLM acc on answered = {correct}/{n_ok} = {correct/max(1,n_ok)*100:.2f}%')
        worst = sorted(by_char.items(), key=lambda kv: kv[1][0]/max(1,kv[1][1]))[:10]
        print('worst chars:', [(c, f'{h}/{n}') for c, (h, n) in worst if n >= 3])

    asyncio.run(run())

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
