# Benchmark Snapshot: UD English EWT Train (Early Random Split)

This snapshot is from an early development run using a random 80/20 train/test split on the train corpus only (2000 sentences). It predates the innovation strategies (EBR, HAT) and the tiktoken fix.

For current results using official UD splits and all strategies, see `benchmark_en_ewt_ud_splits.md`.

## Run configuration
- Corpus: data/ud/en/en_ewt-ud-train.conllu
- Sentences used: 2000
- Train/Test split: 1600/400 (random)
- Metric for tokenization quality: boundary precision, recall, and F1 against UD gold token boundaries

## Results (sorted by boundary F1)

| Strategy | Family | Boundary F1 | Precision | Recall | Tokens/sec | Avg tokens/sentence | OOV rate |
|---|---|---:|---:|---:|---:|---:|---:|
| spacy_en | rule-based + language-specific | 0.9939 | 0.9904 | 0.9974 | 169106.99 | 19.71 | 0.1133 |
| nltk_treebank | statistical/rule-hybrid | 0.9873 | 0.9945 | 0.9802 | 228435.93 | 19.31 | 0.1209 |
| regex_wordpunct | rule-based | 0.9660 | 0.9367 | 0.9972 | 988177.90 | 20.77 | 0.1095 |
| whitespace | rule-based | 0.9246 | 1.0000 | 0.8598 | 900275.16 | 16.97 | 0.1961 |
| wordpiece | subword | 0.8589 | 0.7537 | 0.9981 | 250373.41 | 25.60 | 0.0117 |
| bytelevel_bpe | subword | 0.8330 | 0.7177 | 0.9923 | 290061.33 | 26.68 | 0.0126 |
| sentencepiece_unigram | subword | 0.7995 | 0.6774 | 0.9754 | 272075.21 | 27.75 | 0.0078 |
| sentencepiece_bpe | subword | 0.7987 | 0.7000 | 0.9297 | 336771.68 | 25.67 | 0.0162 |

Note: This run did not include `error_aware_repair`, `hybrid_adaptive_hat`, or a working `tiktoken_cl100k`.
