from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Set, Tuple

from .utils import Span


def boundaries_from_spans(spans: Sequence[Span], text_length: int) -> Set[int]:
    # We ignore the final sentence boundary because it is always present.
    return {end for _, end in spans if end < text_length}


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def vocab_size(tokenized: Iterable[Sequence[str]]) -> int:
    vocab = set()
    for sent in tokenized:
        vocab.update(sent)
    return len(vocab)


def oov_rate(train_tokens: Iterable[Sequence[str]], test_tokens: Iterable[Sequence[str]]) -> float:
    train_vocab = set()
    for sent in train_tokens:
        train_vocab.update(sent)

    total = 0
    oov = 0
    for sent in test_tokens:
        for tok in sent:
            total += 1
            if tok not in train_vocab:
                oov += 1

    return oov / total if total else 0.0


# ---------------------------------------------------------------------------
# Statistical significance and confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_f1_ci(
    sentence_tpfpfn: List[Tuple[int, int, int]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for micro-averaged boundary F1.

    Parameters
    ----------
    sentence_tpfpfn : list of (tp, fp, fn) tuples, one per sentence.
    n_bootstrap : number of bootstrap resamples.
    alpha : significance level (default 0.05 for 95 % CI).
    seed : RNG seed for reproducibility.

    Returns
    -------
    (mean_f1, ci_lower, ci_upper)
    """
    rng = random.Random(seed)
    n = len(sentence_tpfpfn)
    if n == 0:
        return 0.0, 0.0, 0.0

    f1_samples: List[float] = []
    for _ in range(n_bootstrap):
        tp_sum = fp_sum = fn_sum = 0
        for _ in range(n):
            j = rng.randint(0, n - 1)
            tp_sum += sentence_tpfpfn[j][0]
            fp_sum += sentence_tpfpfn[j][1]
            fn_sum += sentence_tpfpfn[j][2]
        _, _, f1 = prf(tp_sum, fp_sum, fn_sum)
        f1_samples.append(f1)

    f1_samples.sort()
    lower = f1_samples[int(n_bootstrap * alpha / 2)]
    upper = f1_samples[min(int(n_bootstrap * (1 - alpha / 2)), n_bootstrap - 1)]
    mean_f1 = sum(f1_samples) / n_bootstrap
    return mean_f1, lower, upper


def paired_permutation_test(
    tpfpfn_a: List[Tuple[int, int, int]],
    tpfpfn_b: List[Tuple[int, int, int]],
    n_permutations: int = 5000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Two-sided paired permutation test for boundary F1 difference.

    Each sentence's (tp, fp, fn) is swapped between systems A and B with
    probability 0.5.  The proportion of permuted differences whose absolute
    value is ≥ the observed |diff| is the p-value.

    Parameters
    ----------
    tpfpfn_a, tpfpfn_b : per-sentence (tp, fp, fn) for the two systems.
    n_permutations : number of random permutations.
    seed : RNG seed.

    Returns
    -------
    (observed_diff, p_value) where observed_diff = F1_A − F1_B.
    """
    assert len(tpfpfn_a) == len(tpfpfn_b), "Systems must be evaluated on the same sentences."
    n = len(tpfpfn_a)

    def _micro_f1(lst: List[Tuple[int, int, int]]) -> float:
        tp = sum(x[0] for x in lst)
        fp = sum(x[1] for x in lst)
        fn = sum(x[2] for x in lst)
        _, _, f1 = prf(tp, fp, fn)
        return f1

    observed_diff = _micro_f1(tpfpfn_a) - _micro_f1(tpfpfn_b)

    rng = random.Random(seed)
    count = 0

    for _ in range(n_permutations):
        perm_a: List[Tuple[int, int, int]] = []
        perm_b: List[Tuple[int, int, int]] = []
        for i in range(n):
            if rng.random() < 0.5:
                perm_a.append(tpfpfn_a[i])
                perm_b.append(tpfpfn_b[i])
            else:
                perm_a.append(tpfpfn_b[i])
                perm_b.append(tpfpfn_a[i])

        perm_diff = _micro_f1(perm_a) - _micro_f1(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return observed_diff, count / n_permutations
