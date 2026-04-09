# Innovation Analysis: EBR and HAT

This document is intentionally organized as a continuation of the baseline comparison workflow:
1. What baseline analysis revealed as shortcomings.
2. What innovations were implemented to address them.
3. How innovations compare against baseline methods on the same metrics.
4. What was actually improved.

## 1. Shortcomings observed after baseline analysis

From the baseline strategy comparison (rule-based + subword):
- Rule-based tokenizers delivered high boundary fidelity, but recurring local errors remained around contractions, repeated punctuation, and hyphenated forms. This happens because regex-class tokenizers split unconditionally on every punctuation character — they have no awareness of linguistic attachment conventions (e.g., UD treats `n't` as one token, but regex splits on the apostrophe).
- Subword tokenizers delivered very low OOV but reduced agreement with UD word boundaries. Subword algorithms optimize for vocabulary compression and coverage, not for matching gold word boundaries — so a single word like `playing` is split into `play` + `##ing`, creating spurious boundary mismatches.
- No baseline strategy simultaneously optimized boundary F1, speed, and robustness. This trade-off is fundamental: high-fidelity rules require per-language engineering overhead, while fast/robust subword methods sacrifice boundary alignment by design.

These shortcomings motivated two innovation tracks:
- Local repair without retraining a large model: EBR.
- Adaptive routing instead of one global policy: HAT.

## 2. Implemented innovations

### 2.1 Error-Aware Boundary Repair (EBR)
- Strategy name: `error_aware_repair`
- File: `src/tokenization_project/strategies.py`
- Base tokenizer: regex word-punctuation segmentation.
- Repair rules implemented:
  - `don | ' | t` -> `do | n't`  (UD-aligned n't contraction)
  - `I | ' | m` -> `I | 'm`  (and other clitics: 's, 're, 've, 'd, 'll)
  - Handles both ASCII `'` and Unicode `'` (U+2019)
  - `- | -` -> `--`
  - `. . .` and `. . . .` collapse into grouped ellipsis tokens
  - `U | . | S | .` -> `U.S.`  (abbreviation collapse, ≥ 2 letter-period pairs)
  - `e | - | mail` -> `e-mail`  (expanded: left side up to 4 chars, covering self-, well-, long-, anti-, etc.)

### 2.2 Hybrid Adaptive Tokenizer (HAT)
- Strategy name: `hybrid_adaptive_hat`
- File: `src/tokenization_project/strategies.py`
- Primary tokenizer: `error_aware_repair` (EBR) — NOT spaCy.  A spaCy instance exists only as a safety fallback for span alignment failures.
- Fallback tokenizer: `wordpiece`
- Gating signals: unseen → subword; rare (freq ≤ threshold) → subword; for frequent tokens, require ≥ 2 of: long (≥15 chars), long hyphenated (≥10 chars), mixed-case/alphanumeric.
- Subword cache is bounded (LRU, max 8192 entries) to prevent memory growth.

## 3. Comparison protocol (same metrics as baseline)

Innovations were evaluated with the same core metrics used for baseline strategies:
- `boundary_precision`, `boundary_recall`, `boundary_f1`
- `tokens_per_second`
- `oov_rate_test`

Supplementary strict agreement metric:
- sentence-level exact-match and mismatch rate (`results/error_analysis_summary_test.csv`)

Primary source files:
- `results/dev/results.csv`
- `results/test/results.csv`

## 4. Innovation vs baseline on the same metrics

### 4.1 EBR compared to regex baseline (`regex_wordpunct`)

| Split | Delta F1 | Delta Precision | Delta Recall | Delta Speed (tok/s) | Delta OOV |
|---|---:|---:|---:|---:|---:|
| dev | +0.0136 | +0.0251 | -0.0001 | -191,983.51 | +0.0035 |
| test | +0.0135 | +0.0259 | -0.0013 | -181,298.63 | +0.0039 |

Strict sentence agreement on test:
- Exact-match: 1,641 vs 1,415 (delta +226)
- Mismatch rate: 0.2099 vs 0.3187 (improvement 0.1088)

Statistical significance: delta F1 = +0.0135, p = 0.0000 (paired permutation test, significant at α=0.05)

### 4.2 HAT compared to subword baseline (`wordpiece`)

| Split | Delta F1 | Delta Precision | Delta Recall | Delta Speed (tok/s) | Delta OOV |
|---|---:|---:|---:|---:|---:|
| dev | +0.0233 | +0.0346 | +0.0035 | -12,735.06 | +0.0091 |
| test | +0.0216 | +0.0323 | +0.0021 | -133,608.62 | +0.0107 |

Strict sentence agreement on test:
- Exact-match: 787 vs 86 (delta +701)
- Mismatch rate: 0.6211 vs 0.9586 (improvement 0.3375)

Statistical significance: delta F1 = +0.0216, p = 0.0000 (paired permutation test, significant at α=0.05)

Innovation-improvement plots:
- ![Innovation delta F1](../results/analysis/innovation_delta_f1.png)
- ![Innovation delta speed](../results/analysis/innovation_delta_speed.png)
- ![Innovation delta OOV](../results/analysis/innovation_delta_oov.png)
- ![Innovation exact-match gain](../results/analysis/innovation_exact_match_gain_test.png)

## 5. What improved and what did not

### 5.1 Verified improvements
- EBR improved boundary F1 on both dev and test over its baseline comparator (statistically significant, p < 0.001).
- EBR also improved sentence-level exact-match behavior.
- HAT improved boundary F1, precision, and strict sentence-level exact agreement over WordPiece (statistically significant, p < 0.001).

### 5.2 Non-improvements / trade-offs
- HAT still has higher OOV than WordPiece (0.013 vs 0.002), though it was reduced by 85% from the untuned baseline (0.086) through rare-token subword routing. The residual gap exists because tokens kept whole by the rule-based path may not appear in training vocabulary, whereas WordPiece can decompose any input into known subword pieces by construction.
- HAT is slower than WordPiece in the current configuration. The overhead comes from HAT's two-pass architecture: EBR tokenization first, then per-token gating decisions with selective WordPiece fallback and LRU cache lookup.

## 6. Reproducibility commands

```bash
.venv/bin/python scripts/run_experiments.py \
  --use-ud-splits \
  --corpus data/ud/en/en_ewt-ud-train.conllu

.venv/bin/python scripts/generate_analysis_plots.py
.venv/bin/python scripts/generate_error_analysis_report.py
```
