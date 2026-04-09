"""Shared utilities for the tokenization project."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple


Span = Tuple[int, int]


def tokens_to_spans(text: str, tokens: Sequence[str]) -> List[Span]:
    """Align a sequence of token strings back to character spans in *text*.

    Skips whitespace between tokens.  Returns an empty list if any token
    cannot be located (alignment failure).
    """
    spans: List[Span] = []
    cursor = 0

    for token in tokens:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        if text.startswith(token, cursor):
            start = cursor
            end = cursor + len(token)
            spans.append((start, end))
            cursor = end
            continue

        fallback = text.find(token, cursor)
        if fallback == -1:
            return []
        start = fallback
        end = fallback + len(token)
        spans.append((start, end))
        cursor = end

    return spans


def resolve_default_corpus(root: Path | None = None) -> Path:
    """Try standard corpus locations and return the first that exists."""
    if root is None:
        root = Path(__file__).resolve().parents[2]

    candidates = [
        root / "data" / "ud" / "en" / "en_ewt-ud-train.conllu",
        root / "data" / "ud" / "en" / "en_ewt_ud_train.conllu",
        root / "data" / "ud" / "en_ewt-ud-train.conllu",
        root / "data" / "ud" / "en_ewt_ud_train.conllu",
        root.parent / "en_ewt-ud-train.conllu",
        root.parent / "en_ewt_ud_train.conllu",
        root / "en_ewt-ud-train.conllu",
        root / "en_ewt_ud_train.conllu",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fallback for a clear error message later.
    return root / "data" / "ud" / "en" / "en_ewt-ud-train.conllu"
