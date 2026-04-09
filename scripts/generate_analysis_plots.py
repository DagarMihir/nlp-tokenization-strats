#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS_DIR = RESULTS / "analysis"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = pd.read_csv(RESULTS / "dev" / "results.csv")
    test = pd.read_csv(RESULTS / "test" / "results.csv")
    return dev, test


def _save_grouped_bar(
    merged: pd.DataFrame, value_col_dev: str, value_col_test: str, title: str, ylabel: str, output: Path
) -> None:
    x = range(len(merged))
    width = 0.38

    plt.figure(figsize=(12, 5.5))
    plt.bar([i - width / 2 for i in x], merged[value_col_dev], width=width, label="dev")
    plt.bar([i + width / 2 for i in x], merged[value_col_test], width=width, label="test")
    plt.xticks(list(x), merged["strategy"], rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_subword_oov_vs_tokens(test: pd.DataFrame, output: Path) -> None:
    subword = test[test["family"].str.contains("subword", case=False)].copy()

    plt.figure(figsize=(8.5, 6))
    plt.scatter(subword["avg_tokens_per_sentence"], subword["oov_rate_test"], s=70)
    for _, row in subword.iterrows():
        plt.annotate(row["strategy"], (row["avg_tokens_per_sentence"], row["oov_rate_test"]), fontsize=8)
    plt.title("Subword Trade-off: Avg Tokens per Sentence vs OOV Rate (test)")
    plt.xlabel("avg_tokens_per_sentence")
    plt.ylabel("oov_rate_test")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_quality_vs_oov(test: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(8.5, 6))

    for family, frame in test.groupby("family"):
        plt.scatter(frame["oov_rate_test"], frame["boundary_f1"], s=70, label=family)
        for _, row in frame.iterrows():
            plt.annotate(row["strategy"], (row["oov_rate_test"], row["boundary_f1"]), fontsize=8)

    plt.title("Boundary F1 vs OOV Rate (test)")
    plt.xlabel("oov_rate_test")
    plt.ylabel("boundary_f1")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_error_confusion_heatmap(output: Path) -> None:
    csv_path = RESULTS / "error_confusion_examples.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    cols = ["whitespace", "regex_treebank", "spacy", "subword_family"]
    labels = ["Whitespace", "Regex/Treebank", "spaCy", "Subword family"]

    matrix = df[cols].replace({"Y": 1, "N": 0}).astype(int).values

    plt.figure(figsize=(10.5, 5.8))
    plt.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(label="Divergence from gold (0=N, 1=Y)")
    plt.xticks(range(len(cols)), labels, rotation=20, ha="right")
    plt.yticks(range(len(df)), df["pattern"])
    plt.title("Tokenizer Family Divergence Heatmap (curated examples)")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            text = "Y" if matrix[r, c] == 1 else "N"
            color = "black" if matrix[r, c] == 0 else "white"
            plt.text(c, r, text, ha="center", va="center", fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _innovation_delta_frame(dev: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["boundary_f1", "tokens_per_second", "oov_rate_test"]
    pairs = [
        ("error_aware_repair", "regex_wordpunct", "EBR vs Regex"),
        ("hybrid_adaptive_hat", "wordpiece", "HAT vs WordPiece"),
    ]

    for split_name, frame in [("dev", dev), ("test", test)]:
        indexed = frame.set_index("strategy")
        for innovation, baseline, label in pairs:
            if innovation not in indexed.index or baseline not in indexed.index:
                continue
            delta = indexed.loc[innovation, metric_cols] - indexed.loc[baseline, metric_cols]
            rows.append(
                {
                    "comparison": label,
                    "split": split_name,
                    "delta_f1": float(delta["boundary_f1"]),
                    "delta_speed": float(delta["tokens_per_second"]),
                    "delta_oov": float(delta["oov_rate_test"]),
                }
            )

    return pd.DataFrame(rows)


def _plot_innovation_metric_deltas(delta_df: pd.DataFrame, output: Path, metric: str, title: str, ylabel: str) -> None:
    if delta_df.empty:
        return

    labels = []
    values = []
    colors = []

    for _, row in delta_df.iterrows():
        labels.append(f"{row['comparison']} ({row['split']})")
        values.append(row[metric])
        colors.append("#2e8b57" if row[metric] >= 0 else "#c0392b")

    plt.figure(figsize=(10.5, 5.5))
    bars = plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=22, ha="right")

    for bar, value in zip(bars, values):
        offset = 0.01 * max(1.0, abs(value))
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        plt.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.4f}" if abs(value) < 100 else f"{value:+.0f}", ha="center", va=va, fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_innovation_exact_match_gains(output: Path) -> None:
    csv_path = RESULTS / "error_analysis_summary_test.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path).set_index("strategy")
    pairs = [
        ("error_aware_repair", "regex_wordpunct", "EBR vs Regex"),
        ("hybrid_adaptive_hat", "wordpiece", "HAT vs WordPiece"),
    ]

    labels = []
    values = []
    rates = []

    for innovation, baseline, label in pairs:
        if innovation not in df.index or baseline not in df.index:
            continue
        exact_delta = int(df.loc[innovation, "exact_match_sentences"] - df.loc[baseline, "exact_match_sentences"])
        mismatch_improve = float(df.loc[baseline, "mismatch_rate"] - df.loc[innovation, "mismatch_rate"])
        labels.append(label)
        values.append(exact_delta)
        rates.append(mismatch_improve)

    if not labels:
        return

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(labels, values, color="#1f77b4")
    plt.title("Innovation Gains: Exact-Match Sentences (test)")
    plt.ylabel("delta exact-match sentences")

    for bar, value, rate in zip(bars, values, rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(2, int(0.02 * max(values))),
            f"+{value} (mismatch -{rate:.4f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    dev, test = _load()
    merged = dev[["strategy", "boundary_f1", "tokens_per_second"]].merge(
        test[["strategy", "boundary_f1", "tokens_per_second"]],
        on="strategy",
        suffixes=("_dev", "_test"),
    )
    merged = merged.sort_values("boundary_f1_test", ascending=False)

    _save_grouped_bar(
        merged,
        value_col_dev="boundary_f1_dev",
        value_col_test="boundary_f1_test",
        title="Boundary F1 by Strategy (dev vs test)",
        ylabel="boundary_f1",
        output=ANALYSIS_DIR / "boundary_f1_dev_vs_test.png",
    )

    _save_grouped_bar(
        merged,
        value_col_dev="tokens_per_second_dev",
        value_col_test="tokens_per_second_test",
        title="Speed by Strategy (dev vs test)",
        ylabel="tokens_per_second",
        output=ANALYSIS_DIR / "speed_dev_vs_test.png",
    )

    _plot_subword_oov_vs_tokens(test, ANALYSIS_DIR / "subword_oov_vs_token_count_test.png")
    _plot_quality_vs_oov(test, ANALYSIS_DIR / "quality_vs_oov_test.png")
    _plot_error_confusion_heatmap(ANALYSIS_DIR / "error_confusion_heatmap.png")

    delta_df = _innovation_delta_frame(dev, test)
    _plot_innovation_metric_deltas(
        delta_df,
        ANALYSIS_DIR / "innovation_delta_f1.png",
        metric="delta_f1",
        title="Innovation Delta: Boundary F1 vs Baseline",
        ylabel="delta boundary_f1",
    )
    _plot_innovation_metric_deltas(
        delta_df,
        ANALYSIS_DIR / "innovation_delta_speed.png",
        metric="delta_speed",
        title="Innovation Delta: Speed vs Baseline",
        ylabel="delta tokens_per_second",
    )
    _plot_innovation_metric_deltas(
        delta_df,
        ANALYSIS_DIR / "innovation_delta_oov.png",
        metric="delta_oov",
        title="Innovation Delta: OOV Rate vs Baseline",
        ylabel="delta oov_rate_test",
    )
    _plot_innovation_exact_match_gains(ANALYSIS_DIR / "innovation_exact_match_gain_test.png")

    print(f"Wrote plots to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()