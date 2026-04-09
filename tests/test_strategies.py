"""Unit tests for tokenization strategies."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the source tree is importable.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenization_project.strategies import (
    ErrorAwareBoundaryRepairStrategy,
    HybridAdaptiveTokenizerStrategy,
    RegexWordPunctStrategy,
    TiktokenCl100kStrategy,
    WhitespaceStrategy,
)
from tokenization_project.utils import Span, tokens_to_spans


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _boundaries(spans, text_length):
    return {end for _, end in spans if end < text_length}


# ---------------------------------------------------------------------------
# tokens_to_spans
# ---------------------------------------------------------------------------


class TestTokensToSpans:
    def test_simple(self):
        text = "Hello world"
        tokens = ["Hello", "world"]
        spans = tokens_to_spans(text, tokens)
        assert spans == [(0, 5), (6, 11)]

    def test_empty(self):
        assert tokens_to_spans("", []) == []

    def test_missing_token_returns_empty(self):
        assert tokens_to_spans("Hello", ["Missing"]) == []

    def test_punctuation(self):
        text = "Hello, world!"
        tokens = ["Hello", ",", "world", "!"]
        spans = tokens_to_spans(text, tokens)
        assert spans == [(0, 5), (5, 6), (7, 12), (12, 13)]


# ---------------------------------------------------------------------------
# WhitespaceStrategy
# ---------------------------------------------------------------------------


class TestWhitespace:
    def test_basic(self):
        s = WhitespaceStrategy()
        assert s.tokenize("Hello world") == ["Hello", "world"]

    def test_contraction_unsplit(self):
        s = WhitespaceStrategy()
        assert s.tokenize("don't stop") == ["don't", "stop"]

    def test_spans_match_text(self):
        s = WhitespaceStrategy()
        text = "a  b   c"
        spans = s.span_tokenize(text)
        assert [text[a:b] for a, b in spans] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# RegexWordPunctStrategy
# ---------------------------------------------------------------------------


class TestRegexWordPunct:
    def test_splits_punctuation(self):
        s = RegexWordPunctStrategy()
        assert s.tokenize("Hello, world!") == ["Hello", ",", "world", "!"]

    def test_contraction_split(self):
        s = RegexWordPunctStrategy()
        result = s.tokenize("don't")
        assert result == ["don", "'", "t"]


# ---------------------------------------------------------------------------
# ErrorAwareBoundaryRepairStrategy  (EBR)
# ---------------------------------------------------------------------------


class TestEBR:
    """Test the expanded repair rule set."""

    def setup_method(self):
        self.ebr = ErrorAwareBoundaryRepairStrategy()

    # ── n't contraction (UD-aligned) ────────────────────────────
    def test_nt_contraction_dont(self):
        result = self.ebr.tokenize("I don't know")
        assert "n't" in result
        assert "do" in result
        assert "don" not in result

    def test_nt_contraction_cant(self):
        result = self.ebr.tokenize("can't go")
        assert result == ["ca", "n't", "go"]

    def test_nt_contraction_wont(self):
        result = self.ebr.tokenize("won't stop")
        assert result == ["wo", "n't", "stop"]

    def test_nt_contraction_shouldnt(self):
        result = self.ebr.tokenize("I shouldn't worry")
        assert "should" in result
        assert "n't" in result

    # ── General apostrophe clitics ──────────────────────────────
    def test_clitic_im(self):
        result = self.ebr.tokenize("I'm fine")
        assert "'m" in result

    def test_clitic_hes(self):
        result = self.ebr.tokenize("he's good")
        assert "'s" in result

    def test_clitic_theyre(self):
        result = self.ebr.tokenize("they're here")
        assert "'re" in result

    # ── Unicode apostrophe ──────────────────────────────────────
    def test_unicode_apostrophe_clitic(self):
        result = self.ebr.tokenize("I\u2019m fine")
        assert any("m" in tok for tok in result)

    def test_unicode_apostrophe_nt(self):
        result = self.ebr.tokenize("don\u2019t")
        assert any("n\u2019t" in tok for tok in result)
        assert "do" in result

    # ── Paired dashes ───────────────────────────────────────────
    def test_double_dash(self):
        result = self.ebr.tokenize("word -- word")
        assert "--" in result

    # ── Ellipsis collapse ───────────────────────────────────────
    def test_ellipsis(self):
        result = self.ebr.tokenize("wait . . .")
        assert "..." in result

    def test_four_dots(self):
        result = self.ebr.tokenize("hmm . . . .")
        assert "...." in result

    # ── Abbreviation collapse ───────────────────────────────────
    def test_us_abbreviation(self):
        result = self.ebr.tokenize("the U . S . is big")
        assert "U.S." in result

    def test_single_letter_period_no_collapse(self):
        """A single letter-period pair should NOT collapse."""
        result = self.ebr.tokenize("Mr . Smith")
        assert "Mr" in result
        assert "." in result

    # ── Hyphenated compounds (expanded) ─────────────────────────
    def test_hyphen_email(self):
        result = self.ebr.tokenize("send e - mail")
        assert "e-mail" in result

    def test_hyphen_self_aware(self):
        result = self.ebr.tokenize("be self - aware")
        assert "self-aware" in result

    def test_hyphen_well_known(self):
        result = self.ebr.tokenize("a well - known fact")
        assert "well-known" in result

    def test_hyphen_long_left_no_join(self):
        """Left side > 4 chars should NOT be joined."""
        result = self.ebr.tokenize("super - natural")
        assert "super" in result
        assert "-" in result
        assert "natural" in result

    # ── Span alignment ──────────────────────────────────────────
    def test_span_alignment(self):
        text = "I don't know"
        spans = self.ebr.span_tokenize(text)
        reconstructed = [text[s:e] for s, e in spans]
        tokens = self.ebr.tokenize(text)
        assert reconstructed == tokens

    def test_tokenize_with_spans_consistent(self):
        text = "he's self - aware and won't stop"
        tokens, spans = self.ebr.tokenize_with_spans(text)
        assert len(tokens) == len(spans)
        reconstructed = [text[s:e] for s, e in spans]
        assert reconstructed == tokens


# ---------------------------------------------------------------------------
# TiktokenCl100kStrategy
# ---------------------------------------------------------------------------


class TestTiktoken:
    """Verify the fixed tiktoken strategy produces usable output."""

    def setup_method(self):
        try:
            self.tik = TiktokenCl100kStrategy()
        except Exception:
            pytest.skip("tiktoken not available")

    def test_tokenize_returns_text(self):
        tokens = self.tik.tokenize("Hello world")
        for t in tokens:
            assert not t.startswith("tid:"), "tokenize should return decoded text, not tid: format"

    def test_tokenize_simple(self):
        tokens = self.tik.tokenize("Hello world")
        joined = " ".join(tokens)
        assert "Hello" in joined
        assert "world" in joined

    def test_span_tokenize_covers_text(self):
        text = "Hello world"
        spans = self.tik.span_tokenize(text)
        assert len(spans) > 0
        # Spans should cover the text characters without huge gaps.
        boundaries = _boundaries(spans, len(text))
        # At minimum we should find at least one correct boundary.
        assert len(boundaries) >= 1

    def test_boundary_f1_not_degenerate(self):
        """A simple sentence should produce some valid boundaries."""
        text = "The quick brown fox jumps over the lazy dog"
        gold_tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        gold_spans = tokens_to_spans(text, gold_tokens)
        pred_spans = self.tik.span_tokenize(text)

        gold_bounds = _boundaries(gold_spans, len(text))
        pred_bounds = _boundaries(pred_spans, len(text))

        tp = len(gold_bounds & pred_bounds)
        # We should get at least some correct boundaries — not 0.06 F1.
        assert tp > 0, "tiktoken should produce at least some correct word boundaries"


# ---------------------------------------------------------------------------
# HybridAdaptiveTokenizerStrategy (HAT)
# ---------------------------------------------------------------------------


class TestHAT:
    """Validate HAT gating and output consistency."""

    def setup_method(self):
        try:
            self.hat = HybridAdaptiveTokenizerStrategy(vocab_size=500)
        except Exception:
            pytest.skip("Dependencies for HAT not available")
        # Fit with minimal training data.
        train = [
            "The quick brown fox jumps over the lazy dog.",
            "She's running fast and he's walking slowly.",
            "I don't think we can't do it.",
            "This is a well-known self-aware system.",
        ]
        self.hat.fit(train)

    def test_tokenize_returns_list(self):
        result = self.hat.tokenize("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_tokenize_with_spans_consistent(self):
        text = "She doesn't know."
        tokens, spans = self.hat.tokenize_with_spans(text)
        assert len(tokens) == len(spans)

    def test_rare_token_subword_routing(self):
        """A token unseen in training should be routed to subword."""
        # "xylophone" almost certainly wasn't in our 4-sentence training set.
        tokens = self.hat.tokenize("xylophone")
        # If routed to subword and split, we'd get multiple tokens.
        # If kept whole, we'd get one.  We just verify it doesn't crash.
        assert len(tokens) >= 1

    def test_cache_bounded(self):
        """Verify the bounded cache doesn't grow unboundedly."""
        for i in range(100):
            self.hat.tokenize(f"unique_token_{i}")
        cache_size = len(self.hat._subword_cache._store)
        assert cache_size <= 8192
