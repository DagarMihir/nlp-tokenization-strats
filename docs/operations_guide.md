# Operations Guide

This document explains how to install dependencies, run experiments, generate analysis artifacts, and extend the project with new tokenizer strategies.

Innovation workflow documents:

- `docs/innovation_ebr_hat.md` - details of Error-Aware Boundary Repair (EBR) and Hybrid Adaptive Tokenizer (HAT)

## 1. Prerequisites

- macOS, Linux, or Windows with Python 3.10+
- A terminal shell
- Internet access for installing Python packages

## 2. Project Layout

Core folders and files:

- `data/ud/en/` - UD English train/dev/test CoNLL-U files
- `src/tokenization_project/` - implementation code
- `scripts/` - runnable scripts for experiments and analysis
- `results/` - generated outputs (CSV, markdown summaries, plots)
- `docs/` - survey, report, and documentation

## 3. Installation and Environment Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Optional verification:

```bash
.venv/bin/python --version
.venv/bin/python -c "import matplotlib, nltk, pandas, spacy, tokenizers, tiktoken; print('ok')"
```

## 4. Dataset Placement

Expected UD file layout:

```text
data/
  ud/
    en/
      en_ewt-ud-train.conllu
      en_ewt-ud-dev.conllu
      en_ewt-ud-test.conllu
```

If you use other languages, keep the same pattern, for example:

```text
data/ud/de/de_hdt-ud-train.conllu
```

## 5. Run Main Experiments

### 5.1 Recommended: official UD split mode

```bash
.venv/bin/python scripts/run_experiments.py \
  --use-ud-splits \
  --corpus data/ud/en/en_ewt-ud-train.conllu
```

This auto-discovers dev/test files from the train filename when available.

### 5.2 Explicit split file mode

```bash
.venv/bin/python scripts/run_experiments.py \
  --train-corpus data/ud/en/en_ewt-ud-train.conllu \
  --dev-corpus data/ud/en/en_ewt-ud-dev.conllu \
  --test-corpus data/ud/en/en_ewt-ud-test.conllu
```

### 5.3 Single-corpus random split mode

```bash
.venv/bin/python scripts/run_experiments.py \
  --corpus data/ud/en/en_ewt-ud-train.conllu \
  --test-ratio 0.2 \
  --seed 42 \
  --max-sentences 3000
```

### 5.4 Multilingual mode

```bash
.venv/bin/python scripts/run_experiments.py \
  --corpora data/ud/en/en_ewt-ud-train.conllu data/ud/de/de_hdt-ud-train.conllu
```

## 6. Generate Analysis Outputs

### 6.1 Core comparison plots

```bash
.venv/bin/python scripts/generate_analysis_plots.py
```

Generates plots in `results/analysis/`, including:

- boundary F1 (dev vs test)
- speed (dev vs test)
- OOV/quality trade-off plots
- confusion heatmap (if confusion CSV exists)

### 6.2 Qualitative disagreement samples

```bash
.venv/bin/python scripts/error_analysis.py \
  --corpus data/ud/en/en_ewt-ud-test.conllu \
  --max-sentences 700 \
  --output results/error_analysis_test.md
```

### 6.3 Enriched error analysis (csv + plots + curated examples)

```bash
.venv/bin/python scripts/generate_error_analysis_report.py
```

Produces:

- `results/error_analysis_summary_test.csv`
- `results/error_category_distribution_test.csv`
- `results/error_confusion_examples.csv`
- `results/error_analysis_enriched_test.md`
- additional error plots in `results/analysis/`

## 7. Key Output Files

Most frequently used artifacts:

- `results/dev/results.csv`
- `results/test/results.csv`
- `results/split_summary.md`
- `results/analysis/*.png`
- `results/error_analysis_test.md`
- `results/error_analysis_enriched_test.md`

## 8. How the Implemented Code Is Organized

Main implementation modules:

- `src/tokenization_project/data.py`
  - parses CoNLL-U files and split helpers
- `src/tokenization_project/utils.py`
  - shared utilities: `Span` type, `tokens_to_spans`, `resolve_default_corpus`
- `src/tokenization_project/strategies.py`
  - tokenizer strategy classes and registry
- `src/tokenization_project/metrics.py`
  - precision/recall/F1, vocab size, OOV, bootstrap CI, paired permutation test
- `src/tokenization_project/experiment.py`
  - experiment execution, evaluation, summary writing, plotting, significance testing

## 9. How To Add a New Tokenizer Strategy

1. Add a new class in `src/tokenization_project/strategies.py` extending `TokenizationStrategy`.
2. Implement:
   - `tokenize(self, text) -> List[str]`
   - `span_tokenize(self, text) -> List[(start, end)]`
   - Optionally override `tokenize_with_spans(self, text) -> Tuple[List[str], List[Span]]` for efficiency.
3. If training is needed, set `requires_fit = True` and implement `fit`.
4. Register the class in `available_strategies(...)`.
5. Re-run:

```bash
python3 -m compileall src scripts
.venv/bin/python scripts/run_experiments.py --use-ud-splits --corpus data/ud/en/en_ewt-ud-train.conllu
.venv/bin/python scripts/generate_analysis_plots.py
```

## 10. Quick End-to-End Command Sequence

```bash
# 1) install
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) run benchmark
.venv/bin/python scripts/run_experiments.py --use-ud-splits --corpus data/ud/en/en_ewt-ud-train.conllu

# 3) generate plots and error analysis
.venv/bin/python scripts/generate_analysis_plots.py
.venv/bin/python scripts/error_analysis.py --corpus data/ud/en/en_ewt-ud-test.conllu --max-sentences 700 --output results/error_analysis_test.md
.venv/bin/python scripts/generate_error_analysis_report.py
```

## 11. Troubleshooting

- Error: missing package
  - Reinstall dependencies with `pip install -r requirements.txt`
- Error: corpus not found
  - Confirm file paths under `data/ud/...`
- Plot not updated
  - Re-run `scripts/generate_analysis_plots.py`
- Strategy skipped during run
  - Check missing dependency or constructor failure in console output and summary markdown
