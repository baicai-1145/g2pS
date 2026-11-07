"""Stream diverse Chinese corpora, extract sentences containing merged-table polyphone chars.

Sources (diverse registers): wikipedia-zh, zhihu QA, medical QA, news(LCSTS if available).
Output: data/external/sents.txt (one sentence per line, 8-64 chars, deduped, test-leak-free).
"""
import hashlib
import os
import re
import sys

POLY = set()
for line in open('datasets/POLYPHONIC_CHARS_MERGED.txt', encoding='utf-8'):
    c, p = line.rstrip('\n').split('\t')
    POLY.add(c)
print('poly chars:', len(POLY), file=sys.stderr)

SENT_SPLIT = re.compile(r'[^。！？!?;\n]{8,64}[。！？!?;]?')

def leaks():
    s = set()
    for path in ['datasets/cpp/test.sent', 'datasets/cpp/dev.sent', 'datasets/cvte/dev.sent',
                 'datasets/cpp/train.sent', 'datasets/cvte/train.sent']:
        for l in open(path, encoding='utf-8'):
            s.add(l.replace('\u2581', '').rstrip('\n'))
    return s

def sent_ok(s):
    return 8 <= len(s) <= 64 and any(c in POLY for c in s)

def main():
    os.makedirs('data/external', exist_ok=True)
    leak = leaks()
    seen = set()
    out = open('data/external/sents.txt', 'w', encoding='utf-8')
    n_total = 0

    from datasets import load_dataset

    def harvest(iterable, cap, tag):
        nonlocal n_total
        n = 0
        for doc in iterable:
            text = doc if isinstance(doc, str) else (doc.get('text') or doc.get('content') or doc.get('summary') or '')
            if not text:
                continue
            for s in SENT_SPLIT.findall(text):
                s = s.strip()
                if not sent_ok(s):
                    continue
                h = hashlib.md5(s.encode()).digest()[:12]
                if h in seen or s in leak:
                    continue
                seen.add(h)
                out.write(s + '\n')
                n += 1
                n_total += 1
                if n >= cap:
                    print(f'[{tag}] hit cap {cap}', file=sys.stderr)
                    return
        print(f'[{tag}] exhausted, got {n}', file=sys.stderr)

    SOURCES = [
        ('wiki', 2_500_000, lambda: (d['text'] for d in load_dataset('wikimedia/wikipedia', '20231101.zh', split='train', streaming=True))),
        ('zhihu', 1_200_000, lambda: (d['content'] for d in load_dataset('wangrui6/Zhihu-KOL', split='train', streaming=True))),
        ('medical', 800_000, lambda: (d['question'] + ' ' + (d.get('answer') or '') for d in load_dataset('FreedomIntelligence/Huatuo26M', 'QA', split='train', streaming=True) if d.get('question'))),
        ('news', 800_000, lambda: (d['summary'] for d in load_dataset('haisenpkg/lcsts', split='train', streaming=True))),
    ]

    for tag, cap, gen in SOURCES:
        try:
            harvest(gen(), cap, tag)
            print(f'total so far: {n_total}', file=sys.stderr)
        except Exception as e:
            print(f'[{tag}] FAILED: {type(e).__name__}: {e}', file=sys.stderr)
    out.close()
    print('FINAL total sentences:', n_total, file=sys.stderr)

if __name__ == '__main__':
    main()
