#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenization_project.data import SentenceExample, parse_conllu
from tokenization_project.strategies import TokenizationStrategy, available_strategies
from tokenization_project.utils import resolve_default_corpus


@dataclass
class ExampleRow:
    text: str
    cause: str
    gold: List[str]
    whitespace: List[str]
    spacy_en: List[str]
    wordpiece: List[str]


def classify_error_type(text: str) -> str:
    if "..." in text:
        return "ellipsis/repeated-punctuation"
    if "'" in text or "’" in text:
        return "apostrophe/contraction"
    if "-" in text:
        return "hyphenation"
    if any(ch in text for ch in "[](){}\"“”"):
        return "brackets/quotes"
    if re.search(r"[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+|[a-z][A-Z]", text):
        return "alnum-or-camelcase"
    if re.search(r"[.,;:!?]", text):
        return "punctuation-attachment"
    return "other"


def likely_cause(text: str) -> str:
    category = classify_error_type(text)
    mapping = {
        "ellipsis/repeated-punctuation": "Tokenizers differ in whether repeated dots are one token or multiple punctuation tokens.",
        "apostrophe/contraction": "Contraction and possessive splitting rules differ (for example, keeping 'm/'s attached vs splitting).",
        "hyphenation": "Hyphenated words can be kept whole or split around the hyphen depending on tokenizer rules.",
        "brackets/quotes": "Boundary behavior around brackets/quotes differs, especially when punctuation is adjacent.",
        "alnum-or-camelcase": "Mixed-case or alphanumeric strings trigger different segmentation heuristics and subword splitting.",
        "punctuation-attachment": "Some tokenizers detach punctuation while others keep it attached to neighboring words.",
        "other": "General segmentation policy differences between word-level and subword tokenizers.",
    }
    return mapping[category]


def compute_outputs(train_data: Sequence[SentenceExample], test_data: Sequence[SentenceExample]) -> None:
    results_dir = ROOT / "results"
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    strategies, skipped = available_strategies(vocab_size=8000)
    if not strategies:
        raise RuntimeError("No strategies available for error analysis")

    train_texts = [x.text for x in train_data]
    for strategy in strategies:
        if strategy.requires_fit:
            strategy.fit(train_texts)

    strategy_names = [s.name for s in strategies]
    totals = {name: 0 for name in strategy_names}
    exact = {name: 0 for name in strategy_names}
    category_counts: Dict[str, int] = {}

    selected: List[ExampleRow] = []
    seen_cause = set()

    for ex in test_data:
        outputs: Dict[str, List[str]] = {}
        any_mismatch = False

        for strategy in strategies:
            pred = strategy.tokenize(ex.text)
            outputs[strategy.name] = pred
            totals[strategy.name] += 1
            if pred == ex.gold_tokens:
                exact[strategy.name] += 1
            else:
                any_mismatch = True

        if not any_mismatch:
            continue

        err_type = classify_error_type(ex.text)
        category_counts[err_type] = category_counts.get(err_type, 0) + 1

        cause = likely_cause(ex.text)
        if len(selected) < 12:
            # Prioritize diversity of error causes first.
            if err_type not in seen_cause or len(selected) < 6:
                selected.append(
                    ExampleRow(
                        text=ex.text,
                        cause=cause,
                        gold=ex.gold_tokens,
                        whitespace=outputs.get("whitespace", []),
                        spacy_en=outputs.get("spacy_en", []),
                        wordpiece=outputs.get("wordpiece", []),
                    )
                )
                seen_cause.add(err_type)

    mismatch_rows = []
    for name in strategy_names:
        total = totals[name]
        exact_count = exact[name]
        mismatch = total - exact_count
        mismatch_rows.append(
            {
                "strategy": name,
                "total_sentences": total,
                "exact_match_sentences": exact_count,
                "mismatch_sentences": mismatch,
                "mismatch_rate": mismatch / total if total else 0.0,
            }
        )

    mismatch_df = pd.DataFrame(mismatch_rows).sort_values("mismatch_rate", ascending=False)
    mismatch_csv = results_dir / "error_analysis_summary_test.csv"
    mismatch_df.to_csv(mismatch_csv, index=False)

    category_df = pd.DataFrame(
        [{"error_type": k, "count": v} for k, v in category_counts.items()]
    ).sort_values("count", ascending=False)
    category_csv = results_dir / "error_category_distribution_test.csv"
    category_df.to_csv(category_csv, index=False)

    # Plot 1: sentence mismatch rate by tokenizer.
    plt.figure(figsize=(11, 5.5))
    plt.bar(mismatch_df["strategy"], mismatch_df["mismatch_rate"])
    plt.title("Sentence Mismatch Rate by Strategy (test)")
    plt.ylabel("mismatch_rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(analysis_dir / "error_mismatch_rate_test.png", dpi=150)
    plt.close()

    # Plot 2: dominant error-type frequency in mismatched sentences.
    plt.figure(figsize=(11, 5.5))
    plt.bar(category_df["error_type"], category_df["count"])
    plt.title("Dominant Error Type Frequency (test mismatches)")
    plt.ylabel("count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(analysis_dir / "error_type_distribution_test.png", dpi=150)
    plt.close()

    md_lines = [
        "# Enriched Error Analysis (test)",
        "",
        "## Files",
        f"- mismatch summary csv: {mismatch_csv}",
        f"- error type distribution csv: {category_csv}",
        "",
        "## Notes",
        f"- skipped strategy initializations: {skipped}",
        f"- total test sentences scanned: {len(test_data)}",
        "",
        "## Curated Divergence Examples",
    ]

    for idx, ex in enumerate(selected[:8], start=1):
        md_lines.extend(
            [
                "",
                f"### Example {idx}",
                f"- Text: {ex.text}",
                f"- Likely cause: {ex.cause}",
                f"- Gold: {' | '.join(ex.gold)}",
                f"- whitespace: {' | '.join(ex.whitespace)}",
                f"- spacy_en: {' | '.join(ex.spacy_en)}",
                f"- wordpiece: {' | '.join(ex.wordpiece)}",
            ]
        )

    (results_dir / "error_analysis_enriched_test.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    train_path = resolve_default_corpus(ROOT)
    # Infer test path from train path.
    test_name = train_path.name.replace("-train.", "-test.").replace("_train.", "_test.")
    test_path = train_path.with_name(test_name)

    train = parse_conllu(train_path)
    test = parse_conllu(test_path)
    compute_outputs(train, test)
    print("Wrote enriched error analysis artifacts to results/ and results/analysis/")


if __name__ == "__main__":
    main()