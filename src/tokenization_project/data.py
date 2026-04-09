from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class SentenceExample:
    text: str
    gold_tokens: List[str]


def parse_conllu(path: Path, max_sentences: int | None = None) -> List[SentenceExample]:
    """Parse a CoNLL-U file into sentence text and gold token sequences."""
    examples: List[SentenceExample] = []
    current_tokens: List[str] = []
    current_text: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line:
                if current_tokens:
                    text = current_text if current_text else " ".join(current_tokens)
                    examples.append(SentenceExample(text=text, gold_tokens=current_tokens.copy()))
                    if max_sentences is not None and len(examples) >= max_sentences:
                        return examples
                current_tokens = []
                current_text = None
                continue

            if line.startswith("#"):
                if line.startswith("# text = "):
                    current_text = line[len("# text = ") :]
                continue

            fields = line.split("\t")
            if len(fields) < 2:
                continue

            token_id = fields[0]
            # Skip multiword and empty nodes.
            if "-" in token_id or "." in token_id:
                continue

            current_tokens.append(fields[1])

    if current_tokens:
        text = current_text if current_text else " ".join(current_tokens)
        examples.append(SentenceExample(text=text, gold_tokens=current_tokens.copy()))

    return examples


def train_test_split(
    data: Sequence[SentenceExample], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[SentenceExample], List[SentenceExample]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    indices = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split = int(len(indices) * (1.0 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    return train, test


def texts(data: Iterable[SentenceExample]) -> List[str]:
    return [x.text for x in data]
