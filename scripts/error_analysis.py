#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenization_project.data import parse_conllu
from tokenization_project.strategies import available_strategies
from tokenization_project.utils import resolve_default_corpus as _resolve_corpus


def resolve_default_corpus() -> Path:
    return _resolve_corpus(ROOT)


def parse_args() -> argparse.Namespace:
    default_corpus = resolve_default_corpus()

    parser = argparse.ArgumentParser(description="Collect qualitative tokenizer disagreements.")
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument("--max-sentences", type=int, default=400)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "error_analysis.md")
    return parser.parse_args()


def format_tokens(tokens: List[str], limit: int = 40) -> str:
    if len(tokens) <= limit:
        return " | ".join(tokens)
    return " | ".join(tokens[:limit]) + " | ..."


def main() -> None:
    args = parse_args()

    data = parse_conllu(args.corpus, max_sentences=args.max_sentences)
    strategies, skipped = available_strategies(vocab_size=4000)

    for strategy in strategies:
        if strategy.requires_fit:
            strategy.fit([x.text for x in data])

    rows = []
    for ex in data:
        outputs = {s.name: s.tokenize(ex.text) for s in strategies}
        if not outputs:
            continue

        # Keep sentences where at least one strategy disagrees with gold tokenization.
        if any(tokens != ex.gold_tokens for tokens in outputs.values()):
            rows.append((ex.text, ex.gold_tokens, outputs))

        if len(rows) >= 20:
            break

    lines = [
        "# Qualitative Error Analysis",
        "",
        f"- Sentences scanned: {len(data)}",
        f"- Strategies available: {[s.name for s in strategies]}",
    ]

    if skipped:
        lines.extend(["- Skipped strategies:", *[f"  - {msg}" for msg in skipped]])

    for idx, (text, gold, outputs) in enumerate(rows, start=1):
        lines.extend(
            [
                "",
                f"## Example {idx}",
                f"- Text: {text}",
                f"- Gold: {format_tokens(gold)}",
            ]
        )
        for name, toks in outputs.items():
            lines.append(f"- {name}: {format_tokens(toks)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
