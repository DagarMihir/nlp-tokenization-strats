# Survey: Historical Progression of Tokenization in NLP

## Motivation
Tokenization has evolved in response to recurring bottlenecks: punctuation handling, language variability, out-of-vocabulary (OOV) words, and scaling to very large neural language models. The key pattern is that each generation solved one major weakness but introduced new trade-offs.

## Historical timeline: what broke, and what improved

### Stage 1: Whitespace tokenization (early baseline era)
- Core idea: split on spaces.
- Why it was used: minimal engineering cost and very fast throughput.
- What it lacked:
	- Fails on punctuation and contractions (for example, "don't", "U.S.").
	- Assumes word boundaries are marked by spaces, which is false for many languages.
	- Produces unstable vocabularies with many rare forms.
- How next methods improved it: rule-based tokenizers introduced punctuation-aware and language-aware boundary rules.

### Stage 2: Regex and handcrafted rule tokenization
- Core idea: split words, punctuation, and symbols with explicit patterns.
- Improvement over Stage 1:
	- Better segmentation around punctuation and special characters.
	- More consistent token boundaries for English-style text.
- What it still lacked:
	- Rules become brittle as domains change (social text, code-mixed text, biomedical text).
	- Hard to generalize to multilingual settings with one rule set.
- How next methods improved it: linguistically engineered tokenizers added richer exception handling and language-specific behavior.

### Stage 3: Treebank and linguistic rule systems (Penn Treebank style, NLTK, spaCy)
- Core idea: codify linguistic conventions for token boundaries (quotes, clitics, abbreviations).
- Improvement over Stage 2:
	- Higher boundary fidelity for formal corpora.
	- More robust sentence and word segmentation in production NLP pipelines.
- What it still lacked:
	- Vocabulary explosion and OOV issues remained for neural models.
	- Rare names, misspellings, and morphology-rich forms still fragmented model performance.
- How next methods improved it: subword tokenization learned reusable pieces to reduce OOV and compress vocabulary.

### Stage 4: Subword BPE for neural MT (Sennrich et al., 2016)
- Core idea: start from characters and iteratively merge frequent symbol pairs.
- Improvement over Stage 3:
	- Strong reduction in OOV by representing rare words as subword compositions.
	- Smaller fixed vocabularies with better generalization to unseen forms.
- What it still lacked:
	- Often relied on pre-tokenized input (language-dependent preprocessing).
	- Learned segments may not align with linguistic words.
- How next methods improved it: WordPiece and later SentencePiece refined objective functions and removed pre-tokenization dependency.

### Stage 5: WordPiece (industrial ASR to BERT era)
- Core idea: learn subword units with likelihood-based selection and greedy longest-match decoding.
- Improvement over Stage 4:
	- Better balance between vocabulary size and segmentation efficiency.
	- Became a standard in pretrained transformer encoders (for example BERT).
- What it still lacked:
	- Still commonly used language-dependent pre-tokenization.
	- Word boundaries were no longer the optimization target.
- How next methods improved it: SentencePiece made tokenization language-agnostic at the raw-text level.

### Stage 6: SentencePiece BPE and Unigram (Kudo and Richardson, 2018)
- Core idea: train directly from raw text; either BPE merges or Unigram language-model pruning.
- Improvement over Stage 5:
	- No mandatory whitespace pre-tokenization; better multilingual portability.
	- Unigram provides a probabilistic alternative to deterministic merge growth.
- What it still lacked:
	- Subword splits can be less interpretable than word-level boundaries.
	- Trade-off between compact vocabulary and longer token sequences remains.
- How next methods improved it: byte-level tokenization addressed Unicode robustness and deployment simplicity.

### Stage 7: Byte-level BPE and high-performance LLM tokenizers (GPT-2, tiktoken family)
- Core idea: tokenize bytes before merges, ensuring every string is representable without unknown tokens.
- Improvement over Stage 6:
	- Near-zero unknown-token behavior across arbitrary Unicode text.
	- Efficient and stable token accounting for LLM training/inference pipelines.
- Remaining limitations:
	- Boundaries can diverge from linguistic words.
	- Tokenization is model-specific, reducing cross-model comparability.

## Where our project sits in this history
This project implements representatives from each phase:
- Early/rule era: whitespace, regex, Treebank-like, spaCy.
- Subword transition: WordPiece, BPE variants.
- Modern multilingual/LLM era: SentencePiece BPE, SentencePiece Unigram, byte-level BPE, GPT-style `tiktoken` (`cl100k_base`).
- Innovation strategies: Error-Aware Boundary Repair (EBR) for local error correction over regex boundaries, and Hybrid Adaptive Tokenizer (HAT) for adaptive rule-plus-subword routing.

This enables a direct empirical test of the historical trade-off:
- Rule-based tokenizers often maximize word-boundary fidelity.
- Subword tokenizers often maximize OOV robustness and model compatibility.
- Hybrid/repair approaches (EBR, HAT) attempt to combine the strengths of both families.

## Evaluation framework used in this project
We compare strategies on UD English data using fixed train/dev/test splits and report:
- Boundary precision, recall, and F1 against gold UD word boundaries.
- Throughput (tokens/second).
- Average tokens per sentence.
- Train vocabulary size.
- Test OOV rate with respect to train vocabulary.

### Boundary quality metric
Let gold boundaries be $B_g$ and predicted boundaries be $B_p$.

$$
P = \frac{|B_p \cap B_g|}{|B_p|}, \quad
R = \frac{|B_p \cap B_g|}{|B_g|}, \quad
F_1 = \frac{2PR}{P + R}
$$

This measures how closely a tokenizer reproduces linguistically annotated word boundaries.

## Key references in chronological context
- Schuster and Nakajima (2012), Japanese and Korean Voice Search, WordPiece-style segmentation in production.
- Sennrich, Haddow, and Birch (2016), Neural Machine Translation of Rare Words with Subword Units, BPE for NMT.
- Kudo and Richardson (2018), SentencePiece, raw-text subword training and Unigram model.
- Devlin et al. (2019), BERT, large-scale WordPiece usage in encoder pretraining.
- Radford et al. (2019), Language Models are Unsupervised Multitask Learners, GPT-2 byte-level BPE.
- Raffel et al. (2020), Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, SentencePiece in T5.
- OpenAI tiktoken documentation: https://github.com/openai/tiktoken
- Hugging Face Tokenizers documentation: https://huggingface.co/docs/tokenizers
