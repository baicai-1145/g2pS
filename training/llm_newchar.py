"""LLM labeling of Xinhua-only char positions (stratified sample) with checkpoint/resume.

Output: data/external/llm_newchar.jsonl  {"sent": s, "pos": i, "char": c, "pinyin": p}
Run long in background; resume-safe.
"""
import asyncio
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, 'student')
sys.path.insert(0, '.')
from llm_label import LLMLabeler

PER_CHAR = 500
BATCH_SENTS = 4  # sentences per LLM call? keep 1 for simplicity of position bookkeeping -> use 1
OUT = 'data/external/llm_newchar.jsonl'


def build_tasks():
    poly_t = set(l.split('\t')[0] for l in open('datasets/POLYPHONIC_CHARS.txt', encoding='utf-8'))
    cands = {}
    for line in open('datasets/POLYPHONIC_CHARS_MERGED.txt', encoding='utf-8'):
        c, p = line.rstrip('\n').split('\t')
        cands.setdefault(c, set()).add(p.replace('v', 'u:'))
    new_chars = {c: sorted(ps) for c, ps in cands.items() if c not in poly_t}

    by_char = defaultdict(list)
    for si, s in enumerate(open('data/external/sents.txt', encoding='utf-8')):
        s = s.rstrip('\n')
        for i, c in enumerate(s):
            if c in new_chars:
                by_char[c].append((s, i))
    rng = random.Random(1313)
    tasks = []  # (sent, [(pos, char, cands)])
    for c, items in by_char.items():
        rng.shuffle(items)
        for s, i in items[:PER_CHAR]:
            tasks.append((s, [(i, c, new_chars[c])]))
    rng.shuffle(tasks)
    return tasks


async def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'):
            try:
                r = json.loads(l)
                done.add((r['sent'], r['pos']))
            except json.JSONDecodeError:
                pass
    tasks = build_tasks()
    tasks = [t for t in tasks if (t[0], t[1][0][0]) not in done]
    print(f'tasks: {len(tasks)} (skip {len(done)} done)', flush=True)

    lab = LLMLabeler(concurrency=256)
    lock = asyncio.Lock()
    fout = open(OUT, 'a', encoding='utf-8')
    n = 0

    async def one(sent, targets):
        nonlocal n
        r = await lab.label_sentence(sent, targets)
        async with lock:
            if r:
                for p, py in r.items():
                    ch = sent[p]
                    fout.write(json.dumps({'sent': sent, 'pos': p, 'char': ch, 'pinyin': py}, ensure_ascii=False) + '\n')
                n += 1
            if n % 500 == 0:
                fout.flush()
                print(f'progress {n}/{len(tasks)} calls={lab.n_call} err={lab.n_error} fail={lab.n_parse_fail}', flush=True)

    BATCH = 2000
    for k in range(0, len(tasks), BATCH):
        chunk = tasks[k:k + BATCH]
        await asyncio.gather(*[one(s, t) for s, t in chunk])
        fout.flush()
    fout.close()
    print('ALL DONE', lab.n_call, lab.n_ok, lab.n_parse_fail, lab.n_error, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
