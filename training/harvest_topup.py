"""Top-up harvest: append diverse sources (news/lcsts, zhihu, medical, dialogue) to existing sents.txt."""
import hashlib
import os
import re
import sys

POLY = set()
for line in open('datasets/POLYPHONIC_CHARS_MERGED.txt', encoding='utf-8'):
    POLY.add(line.rstrip('\n').split('\t')[0])

SENT_SPLIT = re.compile(r'[^。！？!?;\n]{8,64}[。！？!?;]?')

def sent_ok(s):
    return 8 <= len(s) <= 64 and any(c in POLY for c in s)

def main():
    leak = set()
    for path in ['datasets/cpp/test.sent', 'datasets/cpp/dev.sent', 'datasets/cvte/dev.sent',
                 'datasets/cpp/train.sent', 'datasets/cvte/train.sent']:
        for l in open(path, encoding='utf-8'):
            leak.add(l.replace('\u2581', '').rstrip('\n'))

    seen = set()
    out_path = 'data/external/sents.txt'
    for l in open(out_path, encoding='utf-8'):
        seen.add(hashlib.md5(l.rstrip('\n').encode()).digest()[:12])
    print('existing:', len(seen), file=sys.stderr)

    from datasets import load_dataset

    out = open(out_path, 'a', encoding='utf-8')
    n_total = 0

    def harvest(iterable, cap, tag):
        nonlocal n_total
        n = 0
        for doc in iterable:
            if isinstance(doc, str):
                text = doc
            else:
                text = ''
                for k in ('text', 'content', 'question', 'answer', 'summary', 'title'):
                    v = doc.get(k) if hasattr(doc, 'get') else None
                    if v:
                        text += str(v) + ' '
            if not text.strip():
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
                    print(f'[{tag}] cap {cap}', file=sys.stderr)
                    return
        print(f'[{tag}] exhausted n={n}', file=sys.stderr)

    SOURCES = [
        ('news-lcsts', 900_000, lambda: load_dataset('hugcyp/lcsts', split='train', streaming=True)),
        ('thucnews', 900_000, lambda: load_dataset('seamew/THUCNewsText', split='train', streaming=True)),
        ('zhihu', 900_000, lambda: load_dataset('wangrui6/Zhihu-KOL', split='train', streaming=True)),
        ('medical', 500_000, lambda: load_dataset('FreedomIntelligence/huatuo_consultation_qa', split='train', streaming=True)),
        ('dialog-lccc', 500_000, lambda: load_dataset('silver/lccc', 'base', split='train', streaming=True)),
    ]
    for tag, cap, gen in SOURCES:
        try:
            harvest(iter(gen()), cap, tag)
            out.flush()
            print(f'total added: {n_total}', file=sys.stderr)
        except Exception as e:
            print(f'[{tag}] FAILED: {type(e).__name__}: {e}', file=sys.stderr)
    out.close()
    print('DONE, added', n_total, file=sys.stderr)

if __name__ == '__main__':
    main()
