# g2pS Model Card

## Summary

Mandarin polyphone (多音字) pronunciation disambiguation students, distilled from a fine-tuned BERT teacher. Input: raw Chinese sentence. Output: pinyin (with tone digits, `u:` for ü, `5` = neutral tone) for each polyphonic character.

## Training pipeline

1. **Teacher**: BERT-base-chinese fine-tuned (candidate-masked softmax) on merged CPP+CVTE gold data (976k sentences / 973k positions). Test acc 98.70 (sentence) / 96.13 (macro per-char).
2. **Candidate table**: 852 polyphonic chars / 1,816 (char, pinyin) pairs / 887 distinct pinyins — union of the CVTE/CEDICT table and a Xinhua dictionary table (ü notation normalized to `u:`, neutral tones kept as `5`).
3. **Corpus**: 4.3M sentences (8–64 chars, each containing ≥1 polyphone char), deduplicated and leak-filtered against all eval splits:
   - zh Wikipedia (2.5M), LCSTS news (0.9M), Huatuo medical QA, LCCC dialogue, classical Chinese dictionary example sentences (8.4k).
4. **Pseudo-labeling**: teacher inference over 25.1M positions; positions with top-1 confidence ≥ 0.97 (98.2% coverage, 99.71% purity on a gold calibration set) kept → 9.48M hard-labeled positions (40% subsample).
5. **LLM verification** (deepseek-v4.1-flash, structured JSON, streaming, low reasoning effort): stratified sampling of 300 dictionary-extension chars (46,584 calls) → 40k direct labels; chars with ≥98% LLM-vs-pypinyin agreement were expanded with pypinyin defaults (+110k positions, capped 2k/char).
6. **Student training**: logit KD from BERT teacher (gold positions, T=2) + teacher-assistant KD from g2pS-small (bulk positions) + hard CE, EMA, bf16, R-Drop (small/tiny), POS auxiliary head, long-tail reweighting, tone/syllable factorized readout (tiny/nano).

## Architecture

Shared: char embedding (factorized) → depthwise conv stem → pre-norm transformer blocks (RoPE) → per-position readout over 887 labels + per-(char,label) bias table (frequency-prior initialized) → candidate-masked softmax.

| | small | tiny | nano |
|---|---|---|---|
| d / layers / heads | 384 / 6 / 6 | 256 / 5 / 4 | 96 / 6 / 4 (×2 weight sharing = 12 effective) |
| emb dim | 384 | 128 | 48 (vocab 6,000) |
| char bias | full table | rank 64 | rank 16 |
| extra | — | tone-factored readout | tone-factored readout |

## Evaluation (CPP test, 8,896 sentences)

| | acc | avg.pp (macro) | int8 size |
|---|---|---|---|
| small | 98.49 | 95.11 | 16.3 MB |
| tiny | 98.51 | 94.90 | 5.2 MB |
| nano | 98.41 | 94.30 | 1.4 MB |

int8 dynamic quantization: accuracy delta ≤ 0.03 across all tiers.

## Error analysis (tiny, 132/8,935 errors)

- ~60% **world knowledge**: transliterated names (塞尔维亚, 牟定县), classical/official readings (使持节都 dū, 征辟 bì), tool names (斧镐 hào). Not inferable from context; needs dictionary override.
- ~20% **tone-only confusions** within correct syllable (sa1↔sa3) — reduced by the tone-factored readout.
- ~15% **annotation convention** (neutral tone markup, gold inconsistencies).

## Intended use / limitations

- Mandarin simplified Chinese text. Classical Chinese partially supported (dedicated 2.5k gold classical positions in training).
- Returns predictions only for the 852 polyphone chars in the candidate table.
- Not a full G2P system: monophonic chars need a dictionary fallback.
- Input is lowercased internally (latin chars), max 128 chars (centered truncation on the first polyphone).

## Ethics

Built for TTS/front-end and text processing research. Pronunciation predictions can be wrong on rare names and classical texts; do not use as an authoritative source for language instruction.
