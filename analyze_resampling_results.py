#!/usr/bin/env python3
"""
Statistical analysis and comparison figures for the imbalance-handling
resampling benchmark (AUC, Accuracy, F1, G-Mean across datasets D1-D17
and classifiers CatBoost/LR/RF/SVM).

Reads the four tidy CSV tables produced from the reported benchmark
results (data/auc.csv, data/accuracy.csv, data/f1.csv, data/gmean.csv),
then produces:

  1. Per-(metric, classifier) and pooled summary tables (mean +/- std,
     mean rank) for every resampling method, saved as CSV.
  2. A basic data-quality report flagging statistically anomalous cells
     (values far outside a dataset row's typical spread) so they can be
     checked against the source tables before the results are trusted.
  3. Friedman rank-sum tests with Nemenyi post-hoc critical-difference
     (CD) diagrams for each metric, pooling every (classifier, dataset)
     combination as one block -- the standard Demsar (2006) protocol for
     comparing multiple algorithms over multiple datasets.
  4. A boxplot grid comparing the distribution of each method across all
     classifier/dataset combinations, one panel per metric.
  5. A mean-rank heatmap (methods x metrics) summarising the CD-diagram
     results at a glance.

All figures are saved into --output-dir (default: ./results), next to
the summary CSVs and a metadata.json describing exactly what was run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "results"

METHODS = [
    "RUS", "ROS", "SMT", "BS", "Ads", "SSMT", "SMTN", "SMTT",
    "RSMT", "TSMT", "SPYM", "TGF",
]
METHOD_LABELS = {
    "RUS": "Random Under-Sampling",
    "ROS": "Random Over-Sampling",
    "SMT": "SMOTE",
    "BS": "Borderline-SMOTE",
    "Ads": "ADASYN",
    "SSMT": "SVM-SMOTE",
    "SMTN": "SMOTE-ENN",
    "SMTT": "SMOTE-Tomek",
    "RSMT": "SMOTE-RD [1]",
    "TSMT": "Triangular Six-Line (TSMT) [2]",
    "SPYM": "Potato-Yield SMOTE (SPYM) [3]",
    "TGF": "TopoGeoFlow (Proposed)",
}
HIGHLIGHT_METHOD = "TGF"

METRIC_FILES = {
    "AUC": "auc.csv",
    "Accuracy": "accuracy.csv",
    "F1": "f1.csv",
    "G-Mean": "gmean.csv",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_long_table(data_dir: Path) -> pd.DataFrame:
    """Load and melt the four metric tables into one tidy long DataFrame."""
    frames = []
    for metric, filename in METRIC_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing data file: {path}")
        wide = pd.read_csv(path)
        id_columns = ["Classifier", "Dataset"]
        missing = [c for c in id_columns if c not in wide.columns]
        if missing:
            raise ValueError(f"{path} is missing required column(s) {missing}")
        long = wide.melt(
            id_vars=id_columns,
            value_vars=[m for m in METHODS if m in wide.columns],
            var_name="Method",
            value_name="Value",
        )
        long["Metric"] = metric
        frames.append(long)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Value"])
    combined["Method"] = pd.Categorical(
        combined["Method"], categories=METHODS, ordered=True
    )
    return combined


def flag_anomalies(long_df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag cells that are extreme outliers within their own (Metric,
    Classifier, Dataset) row, or that fall outside the valid [0, 1] range
    for a bounded score. This is a diagnostic only -- flagged rows are
    reported, not altered or removed from the statistical analysis, so the
    source tables can be checked and corrected upstream if needed.
    """
    records = []
    group_cols = ["Metric", "Classifier", "Dataset"]
    for key, group in long_df.groupby(group_cols, observed=True):
        values = group["Value"].to_numpy(dtype=float)
        if len(values) < 3:
            continue
        mean, std = values.mean(), values.std(ddof=0)
        if std == 0:
            continue
        z_scores = (values - mean) / std
        for (_, row), z in zip(group.iterrows(), z_scores):
            out_of_range = row["Value"] < 0.0 or row["Value"] > 1.0
            if abs(z) >= z_threshold or out_of_range:
                records.append(
                    {
                        **dict(zip(group_cols, key)),
                        "Method": row["Method"],
                        "Value": row["Value"],
                        "RowZScore": round(float(z), 2),
                        "OutOfRange": out_of_range,
                    }
                )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Summary tables
# --------------------------------------------------------------------------- #
def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std per (Metric, Classifier, Method), aggregated over datasets."""
    summary = (
        long_df.groupby(["Metric", "Classifier", "Method"], observed=True)["Value"]
        .agg(Mean="mean", SD="std", N="count")
        .reset_index()
        .sort_values(["Metric", "Classifier", "Mean"], ascending=[True, True, False])
    )
    return summary


def pooled_summary_with_rank(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean, std, and mean rank per (Metric, Method) across every
    (Classifier, Dataset) block. Ranks are computed within each block
    (1 = best) before averaging, which is the standard way to compare
    algorithms fairly across heterogeneous datasets/classifiers.
    """
    ranked = rank_within_blocks(long_df)
    pooled = (
        long_df.groupby(["Metric", "Method"], observed=True)["Value"]
        .agg(Mean="mean", SD="std", N="count")
        .reset_index()
    )
    mean_rank = (
        ranked.groupby(["Metric", "Method"], observed=True)["Rank"]
        .mean()
        .reset_index()
        .rename(columns={"Rank": "MeanRank"})
    )
    pooled = pooled.merge(mean_rank, on=["Metric", "Method"])
    return pooled.sort_values(["Metric", "MeanRank"])


def rank_within_blocks(long_df: pd.DataFrame) -> pd.DataFrame:
    """Rank methods within each (Metric, Classifier, Dataset) block, higher
    score = rank 1 (best). Blocks with missing methods are dropped so every
    remaining block has a complete set of methods (required for Friedman).
    """
    pivot = long_df.pivot_table(
        index=["Metric", "Classifier", "Dataset"],
        columns="Method",
        values="Value",
        observed=True,
    )
    pivot = pivot.dropna(axis=0, how="any")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    ranks_long = ranks.reset_index().melt(
        id_vars=["Metric", "Classifier", "Dataset"],
        var_name="Method",
        value_name="Rank",
    )
    return ranks_long


# --------------------------------------------------------------------------- #
# Friedman test + Nemenyi post-hoc (no external stats dependency)
# --------------------------------------------------------------------------- #
def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi post-hoc test (Demsar, 2006)."""
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


def friedman_and_cd(
    long_df: pd.DataFrame, metric: str, alpha: float = 0.05
) -> dict:
    """Run the Friedman test and compute Nemenyi CD for one metric, pooling
    every (Classifier, Dataset) pair as one block.
    """
    subset = long_df[long_df["Metric"] == metric]
    pivot = subset.pivot_table(
        index=["Classifier", "Dataset"], columns="Method", values="Value",
        observed=True,
    )
    pivot = pivot.dropna(axis=0, how="any")
    methods_present = [m for m in METHODS if m in pivot.columns]
    pivot = pivot[methods_present]
    n_blocks, k_methods = pivot.shape

    statistic, p_value = stats.friedmanchisquare(
        *[pivot[m].to_numpy() for m in methods_present]
    )
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0).reindex(methods_present)
    cd = nemenyi_critical_difference(k_methods, n_blocks, alpha)

    return {
        "metric": metric,
        "methods": methods_present,
        "n_blocks": int(n_blocks),
        "k_methods": int(k_methods),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "critical_difference": float(cd),
        "mean_ranks": mean_ranks,
    }


def find_cd_groups(mean_ranks: pd.Series, cd: float) -> list[list[str]]:
    """Return maximal cliques of methods whose mean-rank spread is < CD,
    i.e. statistically indistinguishable groups for the CD diagram bars.
    """
    ordered = mean_ranks.sort_values()
    names = ordered.index.tolist()
    values = ordered.to_numpy()
    groups: list[list[str]] = []
    i = 0
    n = len(names)
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] - values[i] < cd:
            j += 1
        groups.append(names[i : j + 1])
        i += 1 if j == i else j - i  # allow overlapping cliques to start next
        if j > i:
            i = j
        else:
            i += 1
    return groups


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_cd_diagram(result: dict, output_path: Path) -> None:
    """Draw a critical-difference diagram: methods placed on a rank axis,
    with a CD bar and horizontal bars linking statistically indistinguishable
    groups (lower rank = better). Labels for closely-spaced methods are
    staggered onto alternating depths so text never overlaps.
    """
    mean_ranks = result["mean_ranks"].sort_values()
    methods = mean_ranks.index.tolist()
    ranks = mean_ranks.to_numpy()
    cd = result["critical_difference"]
    k = len(methods)

    # Stagger label depth: any point within ~0.55 rank units of the previous
    # one is pushed one level deeper so the rotated text labels don't collide.
    depths = [0]
    for i in range(1, k):
        depths.append(depths[i - 1] + 1 if ranks[i] - ranks[i - 1] < 1.0 else 0)
    max_depth = max(depths)

    stem_top = 0.62
    depth_step = 0.34
    fig_height = 1.6 + stem_top + depth_step * (max_depth + 1)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.7 * k), fig_height))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    y_top = 1.0
    y_bottom = -(depth_step * (max_depth + 1) + 0.55)
    ax.set_ylim(y_bottom, y_top + 0.18)
    ax.axhline(stem_top, color="#4B5563", linewidth=1.0)
    ax.set_xticks(range(lo, hi + 1))
    ax.set_xlabel("Average rank across all (classifier, dataset) blocks — lower is better")
    ax.get_yaxis().set_visible(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    for name, rank, depth in zip(methods, ranks, depths):
        proposed = name == HIGHLIGHT_METHOD
        color = "#007C83" if proposed else "#374151"
        text_y = stem_top - 0.06 - depth_step * depth
        ax.plot([rank, rank], [stem_top, text_y + 0.05], color=color, linewidth=1.0)
        ax.plot(rank, stem_top, "o", color=color, markersize=5, zorder=5)
        ax.text(
            rank, text_y, f"{name} ({rank:.2f})", ha="center", va="top", fontsize=8.5,
            fontweight="bold" if proposed else "normal", color=color,
        )

    groups = find_cd_groups(mean_ranks, cd)
    y = stem_top + 0.09
    for group in groups:
        if len(group) < 2:
            continue
        xs = [mean_ranks[m] for m in group]
        ax.plot(
            [min(xs), max(xs)], [y, y], color="#9CA3AF", linewidth=3.2,
            solid_capstyle="round",
        )
        y += 0.06

    cd_x1 = hi + 0.5
    cd_x0 = cd_x1 - cd
    cd_y = y_top + 0.1
    ax.annotate(
        "", xy=(cd_x1, cd_y), xytext=(cd_x0, cd_y),
        arrowprops=dict(arrowstyle="-", color="#111827", linewidth=1.2),
    )
    ax.text((cd_x0 + cd_x1) / 2, cd_y + 0.015, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8.5)

    title = (
        f"{result['metric']}: Friedman χ² = {result['statistic']:.1f}, "
        f"p = {result['p_value']:.2e}  (N blocks = {result['n_blocks']}, "
        f"k methods = {result['k_methods']}, α = {result['alpha']})"
    )
    ax.set_title(title, fontsize=10, pad=28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_metric_boxplots(long_df: pd.DataFrame, output_path: Path) -> None:
    """2x2 grid of boxplots: one panel per metric, one box per method,
    pooled across every classifier and dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()
    for axis, (metric, _) in zip(axes, METRIC_FILES.items()):
        subset = long_df[long_df["Metric"] == metric]
        data = [
            subset.loc[subset["Method"] == m, "Value"].to_numpy()
            for m in METHODS
            if (subset["Method"] == m).any()
        ]
        labels = [m for m in METHODS if (subset["Method"] == m).any()]
        colors = ["#007C83" if m == HIGHLIGHT_METHOD else "#8A95A5" for m in labels]

        box = axis.boxplot(
            data, tick_labels=labels, patch_artist=True, widths=0.6,
            medianprops=dict(color="white", linewidth=1.3),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#374151")
        axis.set_title(metric, fontweight="bold")
        axis.tick_params(axis="x", rotation=45, labelsize=7.5)
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    fig.suptitle(
        "Distribution of each metric across all classifier/dataset combinations, by method",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_rank_heatmap(pooled: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of mean rank (methods x metrics); lower rank is better."""
    table = pooled.pivot(index="Method", columns="Metric", values="MeanRank")
    table = table.reindex([m for m in METHODS if m in table.index])
    table = table[[m for m in METRIC_FILES if m in table.columns]]

    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(table) + 1.5))
    im = ax.imshow(table.to_numpy(), cmap="RdYlGn_r", vmin=1, vmax=len(table))
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for label in ax.get_yticklabels():
        if label.get_text() == HIGHLIGHT_METHOD:
            label.set_color("#007C83")
            label.set_fontweight("bold")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iat[i, j]
            if pd.notna(value):
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=8, color="#111827",
                )
    ax.set_title("Mean rank by method and metric (1 = best)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean rank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_long_table(args.data_dir)

    # --- data quality report -------------------------------------------------
    anomalies = flag_anomalies(long_df)
    anomalies_path = args.output_dir / "data_quality_flags.csv"
    anomalies.to_csv(anomalies_path, index=False)

    # --- summary tables --------------------------------------------------------
    per_classifier_summary = summarize(long_df)
    per_classifier_summary.to_csv(
        args.output_dir / "summary_by_classifier.csv", index=False
    )

    pooled = pooled_summary_with_rank(long_df)
    pooled.to_csv(args.output_dir / "summary_pooled_with_rank.csv", index=False)

    # --- Friedman + Nemenyi CD diagrams ----------------------------------------
    friedman_report = {}
    for metric in METRIC_FILES:
        result = friedman_and_cd(long_df, metric, alpha=args.alpha)
        friedman_report[metric] = {
            k: v for k, v in result.items() if k != "mean_ranks"
        }
        friedman_report[metric]["mean_ranks"] = {
            m: round(float(r), 3) for m, r in result["mean_ranks"].items()
        }
        cd_path = args.output_dir / f"cd_diagram_{metric.replace(' ', '_').replace('-', '')}.png"
        plot_cd_diagram(result, cd_path)

    # --- boxplots and rank heatmap ----------------------------------------------
    boxplot_path = args.output_dir / "metric_boxplots_by_method.png"
    plot_metric_boxplots(long_df, boxplot_path)

    heatmap_path = args.output_dir / "mean_rank_heatmap.png"
    plot_rank_heatmap(pooled, heatmap_path)

    metadata = {
        "methods": {code: METHOD_LABELS[code] for code in METHODS},
        "highlighted_method": HIGHLIGHT_METHOD,
        "n_flagged_cells": int(len(anomalies)),
        "friedman_nemenyi": friedman_report,
        "outputs": {
            "summary_by_classifier": "summary_by_classifier.csv",
            "summary_pooled_with_rank": "summary_pooled_with_rank.csv",
            "data_quality_flags": "data_quality_flags.csv",
            "metric_boxplots": boxplot_path.name,
            "mean_rank_heatmap": heatmap_path.name,
            "cd_diagrams": [
                f"cd_diagram_{m.replace(' ', '_').replace('-', '')}.png"
                for m in METRIC_FILES
            ],
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved statistical analysis and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
