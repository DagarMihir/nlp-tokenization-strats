# nlp-tokenization-strats

Course project for CS458 NLP: survey, implement, compare, and analyze tokenization strategies on a UD English corpus.

## Project goals
- Build a concise survey of major tokenization families.
- Implement multiple tokenization strategies in a reproducible pipeline.
- Compare quality and efficiency with shared metrics.
- Perform quantitative and qualitative error analysis.

## Implemented strategies
- `whitespace` (rule-based baseline)
- `regex_wordpunct` (regex split into words/punctuation)
- `nltk_treebank` (NLTK Treebank tokenizer)
- `spacy_en` (spaCy blank English tokenizer)
- `bytelevel_bpe` (trainable subword tokenizer)
- `wordpiece` (trainable subword tokenizer)
- `sentencepiece_bpe` (trainable SentencePiece BPE)
- `sentencepiece_unigram` (trainable SentencePiece Unigram LM)
- `tiktoken_cl100k` (GPT-style tiktoken byte-level tokenizer)
- `error_aware_repair` (error-aware repair over regex boundaries)
- `hybrid_adaptive_hat` (hybrid adaptive rule + subword tokenizer)

## Metrics
- Boundary Precision/Recall/F1 against UD gold token boundaries
- Speed in tokens/second
- Average tokens per sentence
- Train vocabulary size
- Test OOV rate using train vocabulary

## Latest validated findings (UD official splits)
- Best overall boundary quality: `spacy_en` (dev 0.9812, test 0.9849 boundary F1). spaCy's exception-based tokenizer matches UD conventions closely because it encodes linguistically-curated rules for clitics, abbreviations, and punctuation attachment.
- EBR provides a real improvement over its intended baseline (`regex_wordpunct`):
	- Boundary F1 gain: +0.0136 (dev), +0.0135 (test)
	- Strict sentence exact-match gain on test: +226 sentences (mismatch rate reduction 0.1088)
	- Repairs: n't contractions (UD-aligned), clitics, Unicode apostrophes, abbreviation collapse, expanded hyphen joins.
- HAT shows validated gains over `wordpiece`:
	- Boundary F1 gain: +0.0233 (dev), +0.0216 (test)
	- OOV reduced by 85% from untuned baseline (0.0857 → 0.0126 on test) through rare-token subword routing.
	- Exact-match gain on test: +701 sentences (mismatch rate reduction 0.3375)
	- Remaining trade-off: OOV is still higher than WordPiece (0.013 vs 0.002) because tokens kept whole by the rule-based path may not appear in training vocabulary, whereas WordPiece can always decompose into known subwords.
- Tiktoken (`tiktoken_cl100k`) fixed:
	- Was broken (F1 ≈ 0.06) because byte-level encoding returned byte offsets that were incorrectly mapped to character offsets for UD boundary comparison. After fixing the alignment, tiktoken achieves F1 ≈ 0.91.
- Statistical rigor:
	- Both EBR vs regex and HAT vs WordPiece improvements are statistically significant (p = 0.0000, paired permutation test).
	- 95% bootstrap confidence intervals are reported for all strategies.
- Practical conclusion:
	- `error_aware_repair` is the validated innovation win for this project.
	- `hybrid_adaptive_hat` shows validated quality gains vs WordPiece, with dramatically reduced OOV as the key improvement.

## Repository structure
```
nlp-tokenization-strats/
	data/
		ud/
			en/
				en_ewt-ud-train.conllu
				en_ewt-ud-dev.conllu
				en_ewt-ud-test.conllu
	docs/
		survey.md
		report_template.md
		innovation_ebr_hat.md
		benchmark_en_ewt_ud_splits.md
		operations_guide.md
	scripts/
		run_experiments.py
		error_analysis.py
		generate_analysis_plots.py
		generate_error_analysis_report.py
	src/tokenization_project/
		data.py
		utils.py
		strategies.py
		metrics.py
		experiment.py
	tests/
		test_strategies.py
		test_metrics.py
	results/
		.gitkeep
	requirements.txt
	pyproject.toml
```

## Setup
From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run experiments
Default dataset path is now auto-detected from:
- `data/ud/en/en_ewt-ud-train.conllu`
- `data/ud/en/en_ewt_ud_train.conllu`

```bash
python scripts/run_experiments.py
```

Use official UD train/dev/test splits (recommended when available):

```bash
python scripts/run_experiments.py --use-ud-splits --corpus data/ud/en/en_ewt-ud-train.conllu
```

Or pass split files explicitly:

```bash
python scripts/run_experiments.py \
	--train-corpus data/ud/en/en_ewt-ud-train.conllu \
	--dev-corpus data/ud/en/en_ewt-ud-dev.conllu \
	--test-corpus data/ud/en/en_ewt-ud-test.conllu
```

Run multilingual comparisons (separate result folder per corpus):

```bash
python scripts/run_experiments.py \
	--corpora data/ud/en/en_ewt-ud-train.conllu data/ud/de/de_hdt-ud-train.conllu data/ud/es/es_ancora-ud-train.conllu
```

Useful options:

```bash
python scripts/run_experiments.py \
	--corpus data/ud/en/en_ewt-ud-train.conllu \
	--max-sentences 3000 \
	--test-ratio 0.2 \
	--seed 42 \
	--subword-vocab-size 8000
```

Generated files (in `results/`):
- `results.csv` and `results.md`
- `summary.md`
- `boundary_f1.png`
- `speed_tokens_per_second.png`

Generated files in UD split mode:
- `results/dev/*` and `results/test/*` (separate metrics for each split)
- `results/split_summary.md`

## Notes
- Subword tokenizers optimize robustness and vocabulary efficiency, not necessarily word-boundary matching.
- If optional dependencies are missing, the pipeline skips those strategies and reports the reason.
