from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import re
from typing import List, Sequence, Tuple

from .utils import Span, tokens_to_spans


class TokenizationStrategy(ABC):
    name: str
    family: str
    requires_fit: bool = False

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def span_tokenize(self, text: str) -> List[Span]:
        raise NotImplementedError

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        """Return both tokens and spans in a single call.

        Override in subclasses where this can be done more efficiently
        than calling tokenize() and span_tokenize() separately.
        """
        return self.tokenize(text), self.span_tokenize(text)

    def fit(self, train_texts: Sequence[str]) -> None:
        return


class WhitespaceStrategy(TokenizationStrategy):
    name = "whitespace"
    family = "rule-based"

    _pattern = re.compile(r"\S+")

    def tokenize(self, text: str) -> List[str]:
        return [m.group(0) for m in self._pattern.finditer(text)]

    def span_tokenize(self, text: str) -> List[Span]:
        return [m.span() for m in self._pattern.finditer(text)]


class RegexWordPunctStrategy(TokenizationStrategy):
    name = "regex_wordpunct"
    family = "rule-based"

    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        return [m.group(0) for m in self._pattern.finditer(text)]

    def span_tokenize(self, text: str) -> List[Span]:
        return [m.span() for m in self._pattern.finditer(text)]


class ErrorAwareBoundaryRepairStrategy(TokenizationStrategy):
    """Error-aware boundary repair over regex word/punctuation segmentation.

    Repair rules implemented:
      - n't contractions:  ``don ' t`` → ``do n't``  (UD-aligned)
      - Apostrophe clitics: ``I ' m`` → ``I 'm``
      - Paired dashes: ``- -`` → ``--``
      - Ellipsis collapse: ``. . .`` → ``...``
      - Hyphenated compounds: ``e - mail`` → ``e-mail``  (left ≤ 4 chars)
      - Abbreviation sequences: ``U . S .`` → ``U.S.``

    Both ASCII ``'`` and Unicode RIGHT SINGLE QUOTATION MARK ``'`` (U+2019)
    are handled.
    """

    name = "error_aware_repair"
    family = "rule-based + repair"

    _clitics = {"s", "m", "re", "ve", "d", "ll"}
    _apostrophes = frozenset({"'", "\u2019"})  # ASCII ' and Unicode '

    def __init__(self) -> None:
        self._base = RegexWordPunctStrategy()

    def _repair(self, tokens: Sequence[str]) -> List[str]:
        repaired: List[str] = []
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            # ── n't contraction repair (UD-aligned) ──────────────────────
            # don ' t  →  do  n't   |   can ' t  →  ca  n't
            if tok in self._apostrophes and i + 1 < len(tokens):
                nxt = tokens[i + 1]

                if (
                    nxt.lower() == "t"
                    and repaired
                    and repaired[-1].lower().endswith("n")
                    and len(repaired[-1]) > 1
                ):
                    prev = repaired.pop()
                    repaired.append(prev[:-1])          # "do" from "don"
                    apos = tok                           # preserve original character
                    repaired.append("n" + apos + "t")    # n't  or  n't
                    i += 2
                    continue

                # ── General apostrophe clitics ───────────────────────────
                # I ' m → I 'm  |  Google ' s → Google 's
                if nxt.lower() in self._clitics:
                    repaired.append(tok + nxt)
                    i += 2
                    continue

            # ── Join paired dashes: - - → -- ─────────────────────────────
            if tok == "-" and i + 1 < len(tokens) and tokens[i + 1] == "-":
                repaired.append("--")
                i += 2
                continue

            # ── Collapse repeated dots into ellipsis: . . . → ... ────────
            if tok == ".":
                j = i
                while j < len(tokens) and tokens[j] == ".":
                    j += 1
                run = j - i
                if run >= 3:
                    repaired.append("." * run)
                    i = j
                    continue

            # ── Abbreviation collapse: U . S . → U.S. ───────────────────
            if (
                tok.isalpha()
                and len(tok) == 1
                and i + 1 < len(tokens)
                and tokens[i + 1] == "."
            ):
                j = i
                parts: List[str] = []
                while (
                    j + 1 < len(tokens)
                    and tokens[j].isalpha()
                    and len(tokens[j]) == 1
                    and tokens[j + 1] == "."
                ):
                    parts.append(tokens[j] + ".")
                    j += 2
                if len(parts) >= 2:
                    repaired.append("".join(parts))
                    i = j
                    continue

            # ── Hyphenated compounds: e - mail → e-mail ──────────────────
            # Expanded from len(left) ≤ 2 to ≤ 4 to cover self-aware,
            # well-known, long-term, anti-war, etc.
            if i + 2 < len(tokens) and tokens[i + 1] == "-":
                left = tokens[i]
                right = tokens[i + 2]
                if (
                    left.isalpha()
                    and right.isalpha()
                    and len(left) <= 4
                    and len(right) >= 2
                ):
                    repaired.append(left + "-" + right)
                    i += 3
                    continue

            repaired.append(tok)
            i += 1

        return repaired

    def tokenize(self, text: str) -> List[str]:
        return self._repair(self._base.tokenize(text))

    def span_tokenize(self, text: str) -> List[Span]:
        repaired = self.tokenize(text)
        spans = tokens_to_spans(text, repaired)
        if spans:
            return spans
        return self._base.span_tokenize(text)

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        repaired = self.tokenize(text)
        spans = tokens_to_spans(text, repaired)
        if spans:
            return repaired, spans
        return self._base.tokenize(text), self._base.span_tokenize(text)


class NLTKTreebankStrategy(TokenizationStrategy):
    name = "nltk_treebank"
    family = "statistical/rule-hybrid"

    def __init__(self) -> None:
        from nltk.tokenize import TreebankWordTokenizer

        self._tokenizer = TreebankWordTokenizer()

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(text)

    def span_tokenize(self, text: str) -> List[Span]:
        return list(self._tokenizer.span_tokenize(text))


class SpacyStrategy(TokenizationStrategy):
    name = "spacy_en"
    family = "rule-based + language-specific"

    def __init__(self) -> None:
        import spacy

        self._nlp = spacy.blank("en")

    def tokenize(self, text: str) -> List[str]:
        return [tok.text for tok in self._nlp(text)]

    def span_tokenize(self, text: str) -> List[Span]:
        doc = self._nlp(text)
        return [(tok.idx, tok.idx + len(tok.text)) for tok in doc]


class ByteLevelBPEStrategy(TokenizationStrategy):
    name = "bytelevel_bpe"
    family = "subword"
    requires_fit = True

    def __init__(self, vocab_size: int = 8000, min_frequency: int = 2) -> None:
        from tokenizers import ByteLevelBPETokenizer

        self._tokenizer = ByteLevelBPETokenizer()
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._is_fitted = False

    def fit(self, train_texts: Sequence[str]) -> None:
        self._tokenizer.train_from_iterator(
            train_texts,
            vocab_size=self._vocab_size,
            min_frequency=self._min_frequency,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
        )
        self._is_fitted = True

    def _encode(self, text: str):
        if not self._is_fitted:
            raise RuntimeError("ByteLevelBPEStrategy must be fit before use")
        return self._tokenizer.encode(text)

    def tokenize(self, text: str) -> List[str]:
        return self._encode(text).tokens

    def span_tokenize(self, text: str) -> List[Span]:
        return self._encode(text).offsets

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        enc = self._encode(text)
        return enc.tokens, enc.offsets


class WordPieceStrategy(TokenizationStrategy):
    name = "wordpiece"
    family = "subword"
    requires_fit = True

    def __init__(self, vocab_size: int = 8000, min_frequency: int = 2) -> None:
        from tokenizers import BertWordPieceTokenizer

        self._tokenizer = BertWordPieceTokenizer(lowercase=True)
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._is_fitted = False

    def fit(self, train_texts: Sequence[str]) -> None:
        self._tokenizer.train_from_iterator(
            train_texts,
            vocab_size=self._vocab_size,
            min_frequency=self._min_frequency,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )
        self._is_fitted = True

    def _encode(self, text: str):
        if not self._is_fitted:
            raise RuntimeError("WordPieceStrategy must be fit before use")
        return self._tokenizer.encode(text)

    def tokenize(self, text: str) -> List[str]:
        return self._encode(text).tokens

    def span_tokenize(self, text: str) -> List[Span]:
        return self._encode(text).offsets

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        enc = self._encode(text)
        return enc.tokens, enc.offsets


class SentencePieceBPEStrategy(TokenizationStrategy):
    name = "sentencepiece_bpe"
    family = "subword"
    requires_fit = True

    def __init__(self, vocab_size: int = 8000, min_frequency: int = 2) -> None:
        from tokenizers import SentencePieceBPETokenizer

        self._tokenizer = SentencePieceBPETokenizer()
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._is_fitted = False

    def fit(self, train_texts: Sequence[str]) -> None:
        self._tokenizer.train_from_iterator(
            train_texts,
            vocab_size=self._vocab_size,
            min_frequency=self._min_frequency,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        )
        self._is_fitted = True

    def _encode(self, text: str):
        if not self._is_fitted:
            raise RuntimeError("SentencePieceBPEStrategy must be fit before use")
        return self._tokenizer.encode(text)

    def tokenize(self, text: str) -> List[str]:
        return self._encode(text).tokens

    def span_tokenize(self, text: str) -> List[Span]:
        return self._encode(text).offsets

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        enc = self._encode(text)
        return enc.tokens, enc.offsets


class SentencePieceUnigramStrategy(TokenizationStrategy):
    name = "sentencepiece_unigram"
    family = "subword"
    requires_fit = True

    def __init__(self, vocab_size: int = 8000) -> None:
        from tokenizers import SentencePieceUnigramTokenizer

        self._tokenizer = SentencePieceUnigramTokenizer()
        self._vocab_size = vocab_size
        self._is_fitted = False

    def fit(self, train_texts: Sequence[str]) -> None:
        self._tokenizer.train_from_iterator(
            train_texts,
            vocab_size=self._vocab_size,
            unk_token="<unk>",
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        )
        self._is_fitted = True

    def _encode(self, text: str):
        if not self._is_fitted:
            raise RuntimeError("SentencePieceUnigramStrategy must be fit before use")
        return self._tokenizer.encode(text)

    def tokenize(self, text: str) -> List[str]:
        return self._encode(text).tokens

    def span_tokenize(self, text: str) -> List[Span]:
        return self._encode(text).offsets

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        enc = self._encode(text)
        return enc.tokens, enc.offsets


class TiktokenCl100kStrategy(TokenizationStrategy):
    """GPT-style byte-level tokenizer using tiktoken cl100k_base encoding.

    Previous implementation mapped byte offsets to character offsets, which
    produced severely misaligned spans (boundary F1 ≈ 0.06).  This version
    decodes each token back to text and then uses character-level alignment,
    which correctly handles the leading-space convention used by GPT
    tokenizers.
    """

    name = "tiktoken_cl100k"
    family = "subword + byte-level"

    def __init__(self) -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding("cl100k_base")

    def tokenize(self, text: str) -> List[str]:
        token_ids = self._enc.encode_ordinary(text)
        tokens: List[str] = []
        for tid in token_ids:
            piece = self._enc.decode([tid])
            cleaned = piece.strip()
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def span_tokenize(self, text: str) -> List[Span]:
        tokens = self.tokenize(text)
        return tokens_to_spans(text, tokens)

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        tokens = self.tokenize(text)
        spans = tokens_to_spans(text, tokens)
        return tokens, spans


# ---------------------------------------------------------------------------
# Bounded LRU cache for HAT subword fragment lookups
# ---------------------------------------------------------------------------

class _BoundedCache:
    """Simple insertion-order bounded cache (FIFO eviction)."""

    def __init__(self, maxsize: int = 8192) -> None:
        self._store: OrderedDict[str, Tuple[List[str], List[Span]]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Tuple[List[str], List[Span]] | None:
        val = self._store.get(key)
        if val is not None:
            self._store.move_to_end(key)
        return val

    def put(self, key: str, value: Tuple[List[str], List[Span]]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


class HybridAdaptiveTokenizerStrategy(TokenizationStrategy):
    """Hybrid adaptive tokenizer: EBR coarse segmentation + WordPiece subword fallback.

    Routing logic:
      - Punctuation tokens → kept as-is.
      - Unseen tokens (freq = 0) → subword.
      - Very short tokens (≤ 3 chars) → kept as-is.
      - Rare tokens (freq ≤ rare_threshold, len > 3) → subword for OOV reduction.
      - Frequent tokens → require ≥ 2 risk signals (long, hyphenated, mixed-case).

    NOTE: The primary coarse tokenizer is **EBR** (``ErrorAwareBoundaryRepairStrategy``),
    not spaCy.  A spaCy instance is only a safety fallback for span alignment failures.
    """

    name = "hybrid_adaptive_hat"
    family = "hybrid (rule + subword)"
    requires_fit = True

    _mixed_pattern = re.compile(r"[A-Za-z]+[0-9]|[0-9]+[A-Za-z]|[a-z][A-Z]|[A-Z]{2,}[a-z]")
    _punct_or_symbol = re.compile(r"^[^\w\s]+$", re.UNICODE)

    def __init__(
        self,
        vocab_size: int = 8000,
        min_frequency: int = 2,
        rare_threshold: int = 2,
    ) -> None:
        # EBR pre-pass provides stronger coarse boundaries before adaptive subword routing.
        self._base_rule = ErrorAwareBoundaryRepairStrategy()
        self._base_spacy = SpacyStrategy()
        self._subword = WordPieceStrategy(vocab_size=vocab_size, min_frequency=min_frequency)
        self._rare_threshold = rare_threshold
        self._freq: dict[str, int] = {}
        self._subword_cache = _BoundedCache(maxsize=8192)
        self._last_text: str | None = None
        self._last_tokens: List[str] = []
        self._last_spans: List[Span] = []

    def fit(self, train_texts: Sequence[str]) -> None:
        self._subword.fit(train_texts)

        freq: dict[str, int] = {}
        for text in train_texts:
            for tok in self._base_rule.tokenize(text):
                key = tok.lower()
                freq[key] = freq.get(key, 0) + 1
        self._freq = freq
        self._subword_cache.clear()
        self._last_text = None
        self._last_tokens = []
        self._last_spans = []

    def _needs_subword(self, token: str) -> bool:
        if not token:
            return False

        if self._punct_or_symbol.fullmatch(token):
            return False

        key = token.lower()
        token_freq = self._freq.get(key, 0)

        # Unseen tokens are always routed to subword fallback for OOV robustness.
        if token_freq == 0:
            return True

        # Very short seen tokens are typically stable as whole words.
        if len(token) <= 3:
            return False

        # Route rare tokens directly to subword for OOV reduction.
        # This is the key change that addresses HAT's high OOV rate:
        # tokens seen only 1-2 times in training are unlikely to be robustly
        # handled by either the rule-based or subword vocab alone.
        if token_freq <= self._rare_threshold:
            return True

        # For more frequent tokens, use a multi-signal confidence gate.
        signals = 0

        if len(token) >= 15:
            signals += 1

        if "-" in token and len(token) >= 10:
            signals += 1

        if self._mixed_pattern.search(token):
            signals += 1

        return signals >= 2

    def _hybrid(self, text: str) -> Tuple[List[str], List[Span]]:
        if text == self._last_text:
            return self._last_tokens.copy(), self._last_spans.copy()

        base_tokens = self._base_rule.tokenize(text)
        base_spans = self._base_rule.span_tokenize(text)

        # Safety fallback in case repaired token spans cannot be aligned.
        if len(base_tokens) != len(base_spans):
            base_tokens = self._base_spacy.tokenize(text)
            base_spans = self._base_spacy.span_tokenize(text)

        tokens: List[str] = []
        spans: List[Span] = []

        for token, (start, end) in zip(base_tokens, base_spans):
            if not self._needs_subword(token):
                tokens.append(token)
                spans.append((start, end))
                continue

            fragment = text[start:end]
            cached = self._subword_cache.get(fragment)
            if cached is None:
                sub_tokens = self._subword.tokenize(fragment)
                sub_spans = self._subword.span_tokenize(fragment)
                self._subword_cache.put(fragment, (sub_tokens.copy(), list(sub_spans)))
            else:
                sub_tokens, sub_spans = cached
                sub_tokens = sub_tokens.copy()
                sub_spans = list(sub_spans)

            if not sub_tokens or len(sub_tokens) != len(sub_spans):
                tokens.append(token)
                spans.append((start, end))
                continue

            if len(sub_tokens) == 1:
                tokens.append(token)
                spans.append((start, end))
                continue

            for sub_token, (sub_start, sub_end) in zip(sub_tokens, sub_spans):
                abs_start = start + sub_start
                abs_end = start + sub_end
                if abs_end < abs_start:
                    abs_end = abs_start
                tokens.append(sub_token)
                spans.append((abs_start, abs_end))

        self._last_text = text
        self._last_tokens = tokens.copy()
        self._last_spans = spans.copy()
        return tokens, spans

    def tokenize(self, text: str) -> List[str]:
        tokens, _ = self._hybrid(text)
        return tokens

    def span_tokenize(self, text: str) -> List[Span]:
        _, spans = self._hybrid(text)
        return spans

    def tokenize_with_spans(self, text: str) -> Tuple[List[str], List[Span]]:
        return self._hybrid(text)


def available_strategies(vocab_size: int = 8000) -> Tuple[List[TokenizationStrategy], List[str]]:
    strategies: List[TokenizationStrategy] = []
    skipped: List[str] = []

    candidates = [
        (WhitespaceStrategy, {}),
        (RegexWordPunctStrategy, {}),
        (ErrorAwareBoundaryRepairStrategy, {}),
        (NLTKTreebankStrategy, {}),
        (SpacyStrategy, {}),
        (ByteLevelBPEStrategy, {"vocab_size": vocab_size}),
        (WordPieceStrategy, {"vocab_size": vocab_size}),
        (SentencePieceBPEStrategy, {"vocab_size": vocab_size}),
        (SentencePieceUnigramStrategy, {"vocab_size": vocab_size}),
        (TiktokenCl100kStrategy, {}),
        (HybridAdaptiveTokenizerStrategy, {"vocab_size": vocab_size}),
    ]

    for cls, kwargs in candidates:
        try:
            strategies.append(cls(**kwargs))
        except Exception as exc:  # pragma: no cover - environment dependent.
            skipped.append(f"{cls.__name__}: {exc}")

    return strategies, skipped
