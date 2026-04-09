"""Unit tests for metrics functions."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenization_project.metrics import (
    bootstrap_f1_ci,
    boundaries_from_spans,
    oov_rate,
    paired_permutation_test,
    prf,
    vocab_size,
)


# ---------------------------------------------------------------------------
# prf
# ---------------------------------------------------------------------------


class TestPRF:
    def test_perfect(self):
        p, r, f1 = prf(10, 0, 0)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_no_tp(self):
        p, r, f1 = prf(0, 5, 5)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_fp_only(self):
        p, r, f1 = prf(0, 5, 0)
        assert p == 0.0

    def test_fn_only(self):
        p, r, f1 = prf(0, 0, 5)
        assert r == 0.0

    def test_all_zero(self):
        p, r, f1 = prf(0, 0, 0)
        assert f1 == 0.0

    def test_partial(self):
        p, r, f1 = prf(5, 5, 0)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# boundaries_from_spans
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_basic(self):
        spans = [(0, 5), (6, 11)]
        result = boundaries_from_spans(spans, 11)
        assert result == {5}  # 11 is excluded as text_length

    def test_single_token(self):
        spans = [(0, 5)]
        result = boundaries_from_spans(spans, 5)
        assert result == set()

    def test_multiple(self):
        spans = [(0, 3), (4, 7), (8, 12)]
        result = boundaries_from_spans(spans, 12)
        assert result == {3, 7}


# ---------------------------------------------------------------------------
# vocab_size
# ---------------------------------------------------------------------------


class TestVocabSize:
    def test_basic(self):
        data = [["a", "b", "c"], ["b", "c", "d"]]
        assert vocab_size(data) == 4

    def test_empty(self):
        assert vocab_size([]) == 0


# ---------------------------------------------------------------------------
# oov_rate
# ---------------------------------------------------------------------------


class TestOOVRate:
    def test_no_oov(self):
        train = [["a", "b", "c"]]
        test = [["a", "b"]]
        assert oov_rate(train, test) == 0.0

    def test_full_oov(self):
        train = [["a"]]
        test = [["x", "y"]]
        assert oov_rate(train, test) == 1.0

    def test_partial(self):
        train = [["a", "b"]]
        test = [["a", "c"]]
        assert oov_rate(train, test) == pytest.approx(0.5)

    def test_empty_test(self):
        assert oov_rate([["a"]], []) == 0.0


# ---------------------------------------------------------------------------
# bootstrap_f1_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_perfect_scores(self):
        data = [(10, 0, 0)] * 50
        mean, lower, upper = bootstrap_f1_ci(data, n_bootstrap=500, seed=42)
        assert mean == pytest.approx(1.0)
        assert lower == pytest.approx(1.0)
        assert upper == pytest.approx(1.0)

    def test_ci_width_positive(self):
        data = [(5, 2, 3)] * 30 + [(8, 1, 1)] * 20
        mean, lower, upper = bootstrap_f1_ci(data, n_bootstrap=500, seed=42)
        assert lower <= mean <= upper
        assert upper - lower > 0  # CI should have non-zero width

    def test_empty(self):
        mean, lower, upper = bootstrap_f1_ci([], n_bootstrap=100)
        assert mean == 0.0

    def test_reproducible(self):
        data = [(5, 2, 3)] * 50
        r1 = bootstrap_f1_ci(data, seed=42)
        r2 = bootstrap_f1_ci(data, seed=42)
        assert r1 == r2


# ---------------------------------------------------------------------------
# paired_permutation_test
# ---------------------------------------------------------------------------


class TestPairedPermutation:
    def test_identical_systems(self):
        data = [(5, 2, 3)] * 50
        diff, p = paired_permutation_test(data, data, n_permutations=500, seed=42)
        assert diff == pytest.approx(0.0)
        assert p >= 0.5  # Should be very non-significant

    def test_clearly_different(self):
        system_a = [(10, 0, 0)] * 50  # Perfect
        system_b = [(0, 5, 5)] * 50  # Terrible
        diff, p = paired_permutation_test(system_a, system_b, n_permutations=500, seed=42)
        assert diff > 0
        assert p < 0.05, f"Expected significant difference, got p={p}"

    def test_same_length_required(self):
        with pytest.raises(AssertionError):
            paired_permutation_test([(1, 0, 0)], [(1, 0, 0), (1, 0, 0)])

    def test_reproducible(self):
        a = [(5, 2, 3)] * 50
        b = [(3, 3, 4)] * 50
        r1 = paired_permutation_test(a, b, seed=42)
        r2 = paired_permutation_test(a, b, seed=42)
        assert r1 == r2
