# g2pS — Lightweight Mandarin Polyphone Disambiguation

Context-aware pronunciation (G2P) for Mandarin polyphonic characters, distilled from a fine-tuned BERT teacher into compact transformer students that run **anywhere** — including Raspberry Pi and mobile CPUs.

Three tiers, ONNX int8, **no torch / transformers at inference** (only `onnxruntime` + `numpy`):

| Model | Params | Weights (fp16) | Accuracy* | Latency (numpy, x86) | Best for |
|---|---|---|---|---|---|
| **small** | 16.0M | 32.0 MB | 98.49 | ~10 ms | servers, best long-tail (avg.pp 95.11) |
| **tiny** | 4.9M | 9.8 MB | **98.51** | ~5 ms | best accuracy-per-byte |
| **nano** | 1.1M | **2.2 MB** | 98.41 | ~5 ms | minimal footprint |

\* sentence-level accuracy on the CPP test set (8,896 sentences, 8,935 labeled positions). Teacher (BERT-base, 102M) = 98.70 on the same protocol. Latency: ONNX Runtime CPU, batch 1, single ~32-char sentence; your mileage varies.

## Quick start

```python
from g2ps import G2pS

g = G2pS("models/tiny")           # numpy-only; needs <tier>.npz + meta.pkl
text_used, pairs = g.convert("他在银行工作，行踪不定。")
print([(c, p) for c, p in pairs if p])
# [('行', 'hang2'), ('作', 'zuo4'), ('行', 'xing2'), ('不', 'bu4')]
```

The model returns pinyin for **polyphonic characters only**. For full-sentence pinyin, fill the `None` slots with your favorite dictionary (e.g. `pypinyin`) — g2pS is designed to be the disambiguator, not another syllabary.

CLI:

```bash
python -m g2ps.runtime models/tiny
```

PyTorch training checkpoints (for further fine-tuning) live on the weight hosts under `torch_weights/`; the architecture is `training/modules.py`.

## How it works

- **Student architecture**: char-level transformer (RoPE, conv stem, factorized embedding, per-character label-bias table), trained with logit distillation from the BERT teacher + hard labels, R-Drop, POS auxiliary task, and tone/syllable factorized readout (tiny/nano).
- **Data**: ~1M gold-labeled positions (CPP + CVTE corpora, ~916k sentences) + ~9.5M teacher-pseudo-labeled positions over a 4.3M-sentence multi-domain corpus (Wikipedia-zh, LCSTS news, medical QA, dialogue, classical Chinese), + LLM-verified labels for dictionary-extension characters.
- **852 polyphonic characters** candidate table (merged from CVTE/CEDICT and Xinhua dictionaries, ü-normalized, light-tone convention).
- Small = 6-layer/384d; tiny = 5-layer/256d + low-rank char bias; nano = 6-layer/96d with weight sharing (12 effective layers) + tone-factored readout.

See `MODEL_CARD.md` for training details, error analysis, and known limitations.

## Repository layout

```
g2ps/runtime.py     # the only inference file you need (numpy)
models/<tier>/      # <tier>.npz (fp16 weights) + meta.pkl (vocab/labels/candidates)
examples/quickstart.py   # g2pS + pypinyin full-sentence pinyin
training/           # full torch training pipeline (TRAINING.md)
```

## Benchmarks

Evaluated on the CPP test split (standard protocol). `avg.pp` = macro-averaged per-character accuracy over all polyphones (long-tail sensitivity).

| Model | Params | Acc | avg.pp |
|---|---|---|---|
| published BERT baseline | 102M | 86.09* | 76.58 |
| our BERT teacher | 102M | 98.70 | 96.13 |
| g2pS-small | 16.0M | 98.49 | 95.11 |
| g2pS-tiny | 4.9M | 98.51 | 94.90 |
| g2pS-nano | 1.1M | 98.41 | 94.30 |

\* the published v2 onnx checkpoint of that baseline, evaluated zero-shot on this split (domain-shifted; it was not trained on CPP/CVTE).

## Pure-numpy inference

The runtime re-implements the forward pass in ~150 lines of numpy with load-time weight surgery: embedding×projection fused into one lookup table, RMSNorm scales folded into downstream matrices, FFN up-projections concatenated into single GEMMs, RoPE tables precomputed. BLAS thread count auto-tunes per tier (small matrices favor 1 thread, larger ones 8). Measured accuracy is identical to the fp32 torch model (99.2/99.1/99.1 on a 2k test slice; full-test figures above).

## Known limitations

~60% of remaining errors require world knowledge (transliterated proper nouns, classical readings, rare terminology) that is not inferable from context — e.g. 斧镐 hào, 使持节都 dū. A dictionary-override post-processing layer (CEDICT + 审音表) is the intended remedy; see MODEL_CARD.md.

## Attribution & license

- Training corpora: [CPP](https://github.com/NeRFox/CPP) (Apache-2.0), CVTE polyphone corpus (public research release), Wikipedia-zh (CC BY-SA), LCSTS, Huatuo-26M, LCCC, classical Chinese dictionary sentences — **models only, no data is redistributed here**.
- License: Apache-2.0.
