#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenization_project.experiment import run_experiment, run_experiment_with_ud_splits
from tokenization_project.utils import resolve_default_corpus as _resolve_corpus


def resolve_default_corpus() -> Path:
    return _resolve_corpus(ROOT)


def infer_ud_split_paths(train_path: Path) -> Tuple[Path | None, Path | None]:
    name = train_path.name
    if "-train.conllu" in name:
        return (
            train_path.with_name(name.replace("-train.conllu", "-dev.conllu")),
            train_path.with_name(name.replace("-train.conllu", "-test.conllu")),
        )
    if "_train.conllu" in name:
        return (
            train_path.with_name(name.replace("_train.conllu", "_dev.conllu")),
            train_path.with_name(name.replace("_train.conllu", "_test.conllu")),
        )
    return None, None


def parse_args() -> argparse.Namespace:
    default_corpus = resolve_default_corpus()

    parser = argparse.ArgumentParser(
        description="Run tokenization strategy comparison on a CoNLL-U corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=default_corpus,
        help="Path to one CoNLL-U file",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        type=Path,
        default=None,
        help="One or more CoNLL-U files for multilingual comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory where CSV/plots/summary are saved",
    )
    parser.add_argument("--train-corpus", type=Path, default=None)
    parser.add_argument("--dev-corpus", type=Path, default=None)
    parser.add_argument("--test-corpus", type=Path, default=None)
    parser.add_argument(
        "--use-ud-splits",
        action="store_true",
        help="Use fixed UD train/dev/test splits (auto-detect dev/test from train filename).",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--subword-vocab-size", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.use_ud_splits or args.train_corpus or args.dev_corpus or args.test_corpus:
        train_corpus = args.train_corpus if args.train_corpus is not None else args.corpus
        dev_corpus = args.dev_corpus
        test_corpus = args.test_corpus

        if args.use_ud_splits:
            auto_dev, auto_test = infer_ud_split_paths(train_corpus)
            if dev_corpus is None and auto_dev is not None and auto_dev.exists():
                dev_corpus = auto_dev
            if test_corpus is None and auto_test is not None and auto_test.exists():
                test_corpus = auto_test

        missing = [p for p in [train_corpus, dev_corpus, test_corpus] if p is not None and not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Some UD split corpus files were not found:\n" + "\n".join(str(p) for p in missing)
            )

        if dev_corpus is None and test_corpus is None:
            raise ValueError(
                "UD split mode needs at least dev or test corpus. "
                "Use --use-ud-splits with a train file that has matching -dev/-test files, "
                "or pass --dev-corpus / --test-corpus explicitly."
            )

        report = run_experiment_with_ud_splits(
            train_corpus_path=train_corpus,
            output_dir=args.output_dir,
            dev_corpus_path=dev_corpus,
            test_corpus_path=test_corpus,
            max_sentences=args.max_sentences,
            vocab_size_for_subword=args.subword_vocab_size,
        )

        print("Experiment completed in UD split mode.")
        print(f"Split summary: {report['summary_md']}")
        runs = report["runs"]
        for split_name in ["dev", "test"]:
            if split_name not in runs:
                continue
            split_report = runs[split_name]
            print(f"\nSplit: {split_name}")
            print(f"Results CSV: {split_report['results_csv']}")
            print(f"Summary: {split_report['summary_md']}")

            if split_report["skipped"]:
                print("Skipped strategies:")
                for item in split_report["skipped"]:
                    print(f"- {item}")

            if split_report["errors"]:
                print("Strategy errors:")
                for key, value in split_report["errors"].items():
                    print(f"- {key}: {value}")
        return

    corpora = args.corpora if args.corpora else [args.corpus]
    missing = [p for p in corpora if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Some corpus files were not found:\n"
            + "\n".join(str(p) for p in missing)
        )

    reports = []
    for corpus in corpora:
        multi = len(corpora) > 1
        target_output = args.output_dir / corpus.stem if multi else args.output_dir

        report = run_experiment(
            corpus_path=corpus,
            output_dir=target_output,
            test_ratio=args.test_ratio,
            seed=args.seed,
            max_sentences=args.max_sentences,
            vocab_size_for_subword=args.subword_vocab_size,
        )
        reports.append((corpus, target_output, report))

    print("Experiment completed.")
    for corpus, target_output, report in reports:
        print(f"\nCorpus: {corpus}")
        print(f"Output dir: {target_output}")
        print(f"Results CSV: {report['results_csv']}")
        print(f"Summary: {report['summary_md']}")

        if report["skipped"]:
            print("Skipped strategies:")
            for item in report["skipped"]:
                print(f"- {item}")

        if report["errors"]:
            print("Strategy errors:")
            for key, value in report["errors"].items():
                print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
