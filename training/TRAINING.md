# Training pipeline (g2pS)

Full recipe from raw corpora to the three released tiers. ~1 GPU-day on an RTX 3090 for everything after the teacher.

## 0. Environment

```
torch, transformers, datasets (HF), pypinyin, openai (any OpenAI-compatible endpoint)
```
Base framework: clone [g2pW](https://github.com/GitYCC/g2pW) (Apache-2.0) — the teacher trains inside its `train_g2p_bert.py`; students are self-contained in `training/`.

## 1. Data

1. **Gold**: CPP (GitHub NeRFox/CPP) + CVTE polyphone corpus → `training/prepare_cvte.py` converts CVTE to g2pW's `.sent/.lb/.pos` format; merge both into `datasets/merged/`.
2. **Candidate table**: `training/build_merged_table.py` — merges the CVTE/CEDICT table with a Xinhua dictionary table (ü→`u:`), 852 chars / 887 pinyins.
3. **Student records**: `training/data.py --poly_chars POLYPHONIC_CHARS_MERGED.txt --out datasets/student_v4` — sentence-level records (char ids, positions, labels, POS, poly ids), max 128 chars, ≤16 positions.
4. **External corpus** (pseudo-label substrate): `training/harvest_corpus.py` + `harvest_topup.py` — stream zh-Wikipedia / LCSTS / medical QA / LCCC / classical-Chinese sentences containing ≥1 polyphone char; dedup + eval-leak filter → 4.3M sentences.
5. **Classical gold**: a per-character annotated classical dictionary → 2.5k labeled positions (`datasets/wyw/`).

## 2. Teacher

Fine-tune BERT-base in the g2pW framework on merged gold (see repo's `configs/config_ours.py` recipe: masked-softmax over candidates, per-char label bias). Test acc 98.70.

## 3. Distillation data

| Script | Purpose |
|---|---|
| `gen_teacher_logits.py` | BERT teacher soft labels for all gold positions (full 887-dim, T-tempered at train time) |
| `teacher_bulk.py` | teacher inference over 25.1M external positions, keep top-1 ≥ 0.97 (purity 99.7% on gold calibration) |
| `llm_label.py` + `calibrate_llm.py` | OpenAI-compatible LLM labeler (system-prompt constraints, JSON user prompt, forced JSON output, streaming, low reasoning effort). Calibrate on gold first — ours measured 92.9% |
| `llm_newchar.py` | stratified LLM sampling for dictionary-extension chars; chars with ≥98% LLM-vs-pypinyin agreement get pypinyin-default expansion |
| `assemble_v4.py` | gold-KD + wyw + filtered bulk + LLM rows → `train_full.pkl` (~10.6M positions) |
| `build_v5_data.py` | vocab tail cut (11156→6000, 99.93% coverage) for tiny/nano |
| `ta_gen.py` | teacher-assistant pass: g2pS-small soft labels (top-8) for ALL positions |

## 4. Student training

`training/train.py` — one script, all tiers. Key flags: `--d/--layers/--heads/--ffn/--emb_dim/--bias_rank/--n_passes`, KD (`--kd_path/--kd_T/--kd_alpha`), `--rdrop`, `--tone_aux` (tone/syllable factorized readout), `--bf16`, `--compile`, `--pos_weight`, `--rew_alpha` (long-tail).

Released configurations:

| | small | tiny | nano |
|---|---|---|---|
| arch | 384/6L/6H/ffn1024 | 256/5L/4H/ffn640, emb128, bias-rank64 | 96/6L/4H/ffn256 ×2 shared, emb48, bias-rank16 |
| data | student_v4 (full vocab) | student_v5 (6k vocab) | student_v5 |
| KD | BERT (gold) | BERT (gold) + TA (bulk) | BERT (gold) + TA (bulk) |
| reg | R-Drop 0.3 | R-Drop 0.1 | no R-Drop, dropout 0.05 |
| aux | POS | POS + tone | POS + tone |
| epochs / lr | 3 / 5e-4 | 4 / 6e-4 | 6 / 5e-4 |

Example (nano):

```bash
python training/train.py --data_root datasets/student_v5 --train_pkl train_full.pkl \
  --exp nano --epochs 6 --batch_size 256 --lr 5e-4 \
  --d 96 --layers 6 --heads 4 --ffn 256 --emb_dim 48 --bias_rank 16 --n_passes 2 \
  --dropout 0.05 --rdrop 0 --pos_weight 0.3 --rew_alpha 0.25 \
  --kd_path datasets/student_v4/teacher_probs.pt --kd_T 2.0 --kd_alpha 0.5 \
  --bf16 1 --compile 1 --tone_aux 1 --eval_every 4000
```

## 5. Export

`training/export_npz.py checkpoint.pt meta.pkl models/<tier>/<tier>.npz` → fp16-storage numpy weights for the pure-numpy runtime (`g2ps/runtime.py`). PyTorch checkpoints stay fp32 for further training.

## Known gotchas

- Lowercase all text before locating polyphone positions (some Unicode chars expand under `.lower()`, silently shifting indices).
- `--rdrop 0` still disables only the loss; the double forward is skipped automatically.
- EMA checkpoints from a compiled model carry `_orig_mod.` prefixes and tone-head keys — strip before export (export script handles it).
- DataLoader workers copy the big record lists; don't run diagnostics that re-load the pickle alongside training (OOM).
