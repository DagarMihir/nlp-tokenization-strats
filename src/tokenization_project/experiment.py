from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .data import SentenceExample, parse_conllu, texts, train_test_split
from .metrics import (
    bootstrap_f1_ci,
    boundaries_from_spans,
    oov_rate,
    paired_permutation_test,
    prf,
    vocab_size,
)
from .strategies import TokenizationStrategy, available_strategies
from .utils import Span, tokens_to_spans


@dataclass
class StrategyResult:
    strategy: str
    family: str
    fit_time_seconds: float
    eval_time_seconds: float
    tokens_per_second: float
    avg_tokens_per_sentence: float
    train_vocab_size: int
    oov_rate_test: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    boundary_f1_ci_lower: float = 0.0
    boundary_f1_ci_upper: float = 0.0


def _gold_spans_from_text_tokens(text: str, tokens: Sequence[str]) -> List[Span]:
    """Align gold tokens back to character spans in *text*."""
    return tokens_to_spans(text, tokens)


def _evaluate_boundaries(
    text_value: str, pred_spans: Sequence[Span], gold_spans: Sequence[Span]
) -> Tuple[int, int, int]:
    pred = boundaries_from_spans(pred_spans, len(text_value))
    gold = boundaries_from_spans(gold_spans, len(text_value))

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def _evaluate_strategy(
    strategy: TokenizationStrategy,
    train_data: Sequence[SentenceExample],
    test_data: Sequence[SentenceExample],
) -> StrategyResult:
    train_texts = texts(train_data)

    fit_start = time.perf_counter()
    if strategy.requires_fit:
        strategy.fit(train_texts)
    fit_seconds = time.perf_counter() - fit_start

    train_tokens = [strategy.tokenize(ex.text) for ex in train_data]

    tp_total = 0
    fp_total = 0
    fn_total = 0
    token_count = 0
    test_tokens: List[List[str]] = []
    per_sentence_tpfpfn: List[Tuple[int, int, int]] = []

    eval_start = time.perf_counter()
    for ex in test_data:
        # Single combined call avoids redundant encode/decode in subword strategies.
        pred_toks, pred_spans = strategy.tokenize_with_spans(ex.text)
        gold_spans = _gold_spans_from_text_tokens(ex.text, ex.gold_tokens)

        if gold_spans:
            tp, fp, fn = _evaluate_boundaries(ex.text, pred_spans, gold_spans)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            per_sentence_tpfpfn.append((tp, fp, fn))

        token_count += len(pred_toks)
        test_tokens.append(pred_toks)

    eval_seconds = time.perf_counter() - eval_start

    precision, recall, f1 = prf(tp_total, fp_total, fn_total)
    avg_tokens = token_count / len(test_data) if test_data else 0.0
    tps = token_count / eval_seconds if eval_seconds > 0 else 0.0

    # Bootstrap 95 % confidence interval for boundary F1.
    _, ci_lower, ci_upper = bootstrap_f1_ci(per_sentence_tpfpfn, n_bootstrap=1000)

    return StrategyResult(
        strategy=strategy.name,
        family=strategy.family,
        fit_time_seconds=fit_seconds,
        eval_time_seconds=eval_seconds,
        tokens_per_second=tps,
        avg_tokens_per_sentence=avg_tokens,
        train_vocab_size=vocab_size(train_tokens),
        oov_rate_test=oov_rate(train_tokens, test_tokens),
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=f1,
        boundary_f1_ci_lower=ci_lower,
        boundary_f1_ci_upper=ci_upper,
    )


def _save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    f1_sorted = df.sort_values("boundary_f1", ascending=False)
    speed_sorted = df.sort_values("tokens_per_second", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(f1_sorted["strategy"], f1_sorted["boundary_f1"])
    plt.title("Boundary F1 by Strategy")
    plt.ylabel("F1")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "boundary_f1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(speed_sorted["strategy"], speed_sorted["tokens_per_second"])
    plt.title("Speed (tokens/sec) by Strategy")
    plt.ylabel("tokens/sec")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "speed_tokens_per_second.png", dpi=150)
    plt.close()


def _run_significance_tests(
    strategies: List[TokenizationStrategy],
    test_data: Sequence[SentenceExample],
) -> Dict[str, Dict[str, object]]:
    """Run paired permutation tests for key innovation pairs."""
    pairs = [
        ("error_aware_repair", "regex_wordpunct"),
        ("hybrid_adaptive_hat", "wordpiece"),
    ]
    name_to_strategy = {s.name: s for s in strategies}

    # Collect per-sentence (tp, fp, fn) for each strategy involved in a pair.
    needed = set()
    for a, b in pairs:
        needed.add(a)
        needed.add(b)

    per_sentence: Dict[str, List[Tuple[int, int, int]]] = {}
    for name in needed:
        strat = name_to_strategy.get(name)
        if strat is None:
            continue
        results: List[Tuple[int, int, int]] = []
        for ex in test_data:
            _, pred_spans = strat.tokenize_with_spans(ex.text)
            gold_spans = _gold_spans_from_text_tokens(ex.text, ex.gold_tokens)
            if gold_spans:
                results.append(_evaluate_boundaries(ex.text, pred_spans, gold_spans))
        per_sentence[name] = results

    sig_results: Dict[str, Dict[str, object]] = {}
    for a_name, b_name in pairs:
        if a_name not in per_sentence or b_name not in per_sentence:
            continue
        if len(per_sentence[a_name]) != len(per_sentence[b_name]):
            continue
        delta, p_val = paired_permutation_test(
            per_sentence[a_name], per_sentence[b_name], n_permutations=5000
        )
        sig_results[f"{a_name}_vs_{b_name}"] = {
            "delta_f1": delta,
            "p_value": p_val,
            "significant_at_005": p_val < 0.05,
        }

    return sig_results


def _run_on_fixed_sets(
    train_data: Sequence[SentenceExample],
    eval_data: Sequence[SentenceExample],
    output_dir: Path,
    summary_meta_lines: Sequence[str],
    vocab_size_for_subword: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies, skipped = available_strategies(vocab_size=vocab_size_for_subword)

    results: List[StrategyResult] = []
    errors: Dict[str, str] = {}

    for strategy in strategies:
        try:
            results.append(_evaluate_strategy(strategy, train_data, eval_data))
        except Exception as exc:
            errors[strategy.name] = str(exc)

    if not results:
        raise RuntimeError("No strategy could be evaluated. Check installed dependencies.")

    df = pd.DataFrame(asdict(r) for r in results)
    df = df.sort_values("boundary_f1", ascending=False)

    csv_path = output_dir / "results.csv"
    md_path = output_dir / "results.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    _save_plots(df, output_dir)

    # Run significance tests for innovation pairs.
    sig_results = _run_significance_tests(strategies, eval_data)
    if sig_results:
        sig_lines = ["# Significance Tests", ""]
        for pair_name, result in sig_results.items():
            sig_lines.append(f"## {pair_name}")
            sig_lines.append(f"- Delta F1: {result['delta_f1']:+.6f}")
            sig_lines.append(f"- p-value: {result['p_value']:.4f}")
            sig_lines.append(
                f"- Significant at α=0.05: {'Yes' if result['significant_at_005'] else 'No'}"
            )
            sig_lines.append("")
        (output_dir / "significance_tests.md").write_text("\n".join(sig_lines), encoding="utf-8")

    summary_lines = [
        "# Experiment Summary",
        "",
        *summary_meta_lines,
        "",
        "## Top Strategy by Boundary F1",
    ]

    top = df.iloc[0]
    summary_lines.extend(
        [
            f"- Strategy: {top['strategy']}",
            f"- Boundary F1: {top['boundary_f1']:.4f}"
            + (
                f" (95% CI: [{top['boundary_f1_ci_lower']:.4f}, {top['boundary_f1_ci_upper']:.4f}])"
                if "boundary_f1_ci_lower" in top.index
                else ""
            ),
            f"- Speed (tokens/sec): {top['tokens_per_second']:.2f}",
            "",
            "## Notes",
            "- Boundary F1 evaluates agreement with gold token boundaries from UD annotations.",
            "- Subword methods generally produce more splits and can trade boundary precision for vocabulary efficiency.",
            "- Confidence intervals are 95% bootstrap CIs (1000 resamples).",
        ]
    )

    if sig_results:
        summary_lines.extend(["", "## Statistical Significance"])
        for pair_name, result in sig_results.items():
            sig_str = "significant" if result["significant_at_005"] else "NOT significant"
            summary_lines.append(
                f"- {pair_name}: delta F1 = {result['delta_f1']:+.4f}, "
                f"p = {result['p_value']:.4f} ({sig_str} at α=0.05)"
            )

    if skipped:
        summary_lines.extend(["", "## Skipped Strategies", *[f"- {msg}" for msg in skipped]])

    if errors:
        summary_lines.extend(["", "## Strategy Errors", *[f"- {k}: {v}" for k, v in errors.items()]])

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "results_csv": str(csv_path),
        "results_md": str(md_path),
        "summary_md": str(summary_path),
        "top_strategy": str(top["strategy"]),
        "top_boundary_f1": float(top["boundary_f1"]),
        "skipped": skipped,
        "errors": errors,
        "significance": sig_results,
    }


def run_experiment(
    corpus_path: Path,
    output_dir: Path,
    test_ratio: float = 0.2,
    seed: int = 42,
    max_sentences: int | None = None,
    vocab_size_for_subword: int = 8000,
) -> Dict[str, object]:
    data = parse_conllu(corpus_path, max_sentences=max_sentences)
    if len(data) < 10:
        raise ValueError("Not enough sentences in dataset to run comparison")

    train_data, test_data = train_test_split(data, test_ratio=test_ratio, seed=seed)
    summary_meta = [
        f"- Corpus: {corpus_path}",
        f"- Sentences used: {len(data)}",
        f"- Train/Test: {len(train_data)}/{len(test_data)} (random split)",
    ]

    return _run_on_fixed_sets(
        train_data=train_data,
        eval_data=test_data,
        output_dir=output_dir,
        summary_meta_lines=summary_meta,
        vocab_size_for_subword=vocab_size_for_subword,
    )


def run_experiment_with_ud_splits(
    train_corpus_path: Path,
    output_dir: Path,
    dev_corpus_path: Path | None = None,
    test_corpus_path: Path | None = None,
    max_sentences: int | None = None,
    vocab_size_for_subword: int = 8000,
) -> Dict[str, object]:
    train_data = parse_conllu(train_corpus_path, max_sentences=max_sentences)
    if len(train_data) < 10:
        raise ValueError("Not enough sentences in train dataset to run comparison")

    runs: Dict[str, Dict[str, object]] = {}
    split_summary_lines = [
        "# UD Split Evaluation Summary",
        "",
        f"- Train corpus: {train_corpus_path}",
    ]

    if dev_corpus_path is not None:
        dev_data = parse_conllu(dev_corpus_path, max_sentences=max_sentences)
        if len(dev_data) < 5:
            raise ValueError("Not enough sentences in dev dataset to run comparison")

        dev_report = _run_on_fixed_sets(
            train_data=train_data,
            eval_data=dev_data,
            output_dir=output_dir / "dev",
            summary_meta_lines=[
                f"- Train corpus: {train_corpus_path}",
                f"- Eval corpus: {dev_corpus_path}",
                f"- Train/Eval: {len(train_data)}/{len(dev_data)} (fixed UD split)",
            ],
            vocab_size_for_subword=vocab_size_for_subword,
        )
        runs["dev"] = dev_report
        split_summary_lines.extend(
            [
                f"- Dev corpus: {dev_corpus_path}",
                f"- Dev top strategy: {dev_report['top_strategy']} (F1={dev_report['top_boundary_f1']:.4f})",
            ]
        )

    if test_corpus_path is not None:
        test_data = parse_conllu(test_corpus_path, max_sentences=max_sentences)
        if len(test_data) < 5:
            raise ValueError("Not enough sentences in test dataset to run comparison")

        test_report = _run_on_fixed_sets(
            train_data=train_data,
            eval_data=test_data,
            output_dir=output_dir / "test",
            summary_meta_lines=[
                f"- Train corpus: {train_corpus_path}",
                f"- Eval corpus: {test_corpus_path}",
                f"- Train/Eval: {len(train_data)}/{len(test_data)} (fixed UD split)",
            ],
            vocab_size_for_subword=vocab_size_for_subword,
        )
        runs["test"] = test_report
        split_summary_lines.extend(
            [
                f"- Test corpus: {test_corpus_path}",
                f"- Test top strategy: {test_report['top_strategy']} (F1={test_report['top_boundary_f1']:.4f})",
            ]
        )

    if not runs:
        raise ValueError("Provide at least one of dev_corpus_path or test_corpus_path")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "split_summary.md"
    summary_path.write_text("\n".join(split_summary_lines), encoding="utf-8")

    return {
        "mode": "ud_splits",
        "summary_md": str(summary_path),
        "runs": runs,
    }
