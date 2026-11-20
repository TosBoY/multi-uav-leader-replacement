"""Summarize safety metrics and create leader-replacement comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ._common import PROJECT_ROOT, load_run_metrics, summarize_by_conditions
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from analysis.metrics_analysis._common import (
        PROJECT_ROOT,
        load_run_metrics,
        summarize_by_conditions,
    )


OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "analysis" / "safety_performance"

METRIC_COLUMNS = [
    "run_id",
    "baseline_nonfail_run_id",
    "path_id",
    "number_of_drone",
    "shape",
    "spread",
    "plane",
    "election_type",
    "fail_time",
    "min_separation",
    "min_separation_time",
    "baseline_min_separation",
    "n_close_steps",
]


def _load_metrics() -> pd.DataFrame:
    """Load run-level safety metrics from DuckDB."""
    return load_run_metrics(METRIC_COLUMNS)


def _prepare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add baseline safety comparisons and success flags."""
    result = metrics.copy()
    result["separation_change_from_baseline"] = (
        result["min_separation"] - result["baseline_min_separation"]
    )
    result["separation_ratio_to_baseline"] = (
        result["min_separation"] / result["baseline_min_separation"]
    )
    result["safety_success"] = result["n_close_steps"].fillna(0).eq(0)
    return result


def _method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create one safety summary row for each election method."""
    summary = (
        metrics.groupby("election_type", as_index=False)
        .agg(
            run_count=("run_id", "count"),
            safety_success_rate=("safety_success", "mean"),
            mean_min_separation=("min_separation", "mean"),
            median_min_separation=("min_separation", "median"),
            p05_min_separation=(
                "min_separation",
                lambda values: values.quantile(0.05),
            ),
            mean_min_separation_time=("min_separation_time", "mean"),
            mean_close_steps=("n_close_steps", "mean"),
            total_close_steps=("n_close_steps", "sum"),
            mean_baseline_min_separation=("baseline_min_separation", "mean"),
            mean_separation_change_from_baseline=(
                "separation_change_from_baseline",
                "mean",
            ),
            mean_separation_ratio_to_baseline=(
                "separation_ratio_to_baseline",
                "mean",
            ),
        )
        .sort_values("election_type")
        .reset_index(drop=True)
    )
    summary["safety_success_rate"] *= 100.0
    return summary


def _write_conclusion(summary: pd.DataFrame) -> None:
    """Write a detailed, data-driven safety conclusion."""
    if summary.empty:
        conclusion = "No safety-performance data was available.\n"
    else:
        def percent(value: object) -> str:
            return "N/A" if pd.isna(value) else f"{float(value):.1f}%"

        safest = summary.loc[summary["safety_success_rate"].idxmax()]
        strongest_margin = summary.loc[
            summary["median_min_separation"].idxmax()
        ]
        fewest_violations = summary.loc[
            summary["mean_close_steps"].idxmin()
        ]
        lines = [
            "SAFETY-PERFORMANCE ANALYSIS CONCLUSION",
            "=======================================",
            "",
            "Overall scope",
            f"- Failed runs analysed: {int(summary['run_count'].sum())}",
            f"- Election methods compared: {len(summary)}",
            "",
            "Method comparison",
            (
                f"- Best safety success: {safest['election_type']} "
                f"({percent(safest['safety_success_rate'])})"
            ),
            (
                f"- Largest median minimum-separation margin: "
                f"{strongest_margin['election_type']} "
                f"({strongest_margin['median_min_separation']:.3f} m)"
            ),
            (
                f"- Fewest mean close-threshold steps: "
                f"{fewest_violations['election_type']} "
                f"({fewest_violations['mean_close_steps']:.3f} steps)"
            ),
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['election_type']}: runs={int(row['run_count'])}, "
                f"safety success={percent(row['safety_success_rate'])}, "
                f"mean separation={row['mean_min_separation']:.3f} m, "
                f"median separation={row['median_min_separation']:.3f} m, "
                f"5th-percentile separation="
                f"{row['p05_min_separation']:.3f} m, "
                f"mean close steps={row['mean_close_steps']:.3f}, "
                f"total close steps={row['total_close_steps']:.0f}, "
                f"change from baseline="
                f"{row['mean_separation_change_from_baseline']:.3f} m, "
                f"ratio to baseline="
                f"{row['mean_separation_ratio_to_baseline']:.3f}"
            )
        lines.extend(
            [
                "",
                "Interpretation",
                (
                    "Higher separation and safety-success values indicate "
                    "greater safety margin. Lower close-step counts indicate "
                    "fewer threshold violations. A negative change from "
                    "baseline means the failure reduced separation."
                ),
            ]
        )
        conclusion = "\n".join(lines) + "\n"
    (OUTPUT_DIRECTORY / "safety_performance_conclusion.txt").write_text(
        conclusion,
        encoding="utf-8",
    )


def _save_bar_chart(
    summary: pd.DataFrame,
    columns: list[tuple[str, str]],
    title: str,
    description: str,
    ylabel: str,
    filename: str,
) -> None:
    """Save a grouped safety bar chart."""
    figure, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    width = 0.8 / len(columns)
    for index, (column, label) in enumerate(columns):
        axis.bar(
            x + (index - (len(columns) - 1) / 2) * width,
            summary[column],
            width,
            label=label,
        )
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.text(0.5, 0.925, description, ha="center", wrap=True)
    axis.set_xlabel("Election method")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, summary["election_type"], rotation=30)
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=180)
    plt.close(figure)


def _save_boxplot(
    metrics: pd.DataFrame,
    column: str,
    title: str,
    description: str,
    ylabel: str,
    filename: str,
) -> None:
    """Save a dense-data-friendly safety distribution plot by method."""
    groups = []
    labels = []
    for method, group in metrics.groupby("election_type", sort=True):
        values = group[column].dropna().to_numpy()
        if len(values):
            labels.append(method)
            groups.append(values)
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    positions = np.arange(1, len(groups) + 1)
    violin = axis.violinplot(
        groups,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#8ecae6")
        body.set_edgecolor("#277da1")
        body.set_alpha(0.65)
    axis.boxplot(
        groups,
        positions=positions,
        widths=0.18,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#f8f9fa", "edgecolor": "#222222"},
        medianprops={"color": "#d00000", "linewidth": 2},
        whiskerprops={"color": "#222222"},
        capprops={"color": "#222222"},
    )
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.text(0.5, 0.925, description, ha="center", wrap=True)
    axis.set_xlabel("Election method")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=180)
    plt.close(figure)


def _save_condition_plot(
    metrics: pd.DataFrame,
    condition: str,
    title: str,
    description: str,
    xlabel: str,
    filename: str,
) -> None:
    """Save minimum separation by method and scenario condition."""
    grouped = (
        metrics.groupby(["election_type", condition], as_index=False)
        .agg(minimum_separation=("min_separation", "mean"))
    )
    categories = sorted(grouped[condition].dropna().unique())
    methods = sorted(grouped["election_type"].dropna().unique())
    if not categories or not methods:
        return

    x = np.arange(len(categories))
    width = 0.8 / len(methods)
    figure, axis = plt.subplots(figsize=(10, 6))
    for index, method in enumerate(methods):
        values = (
            grouped[grouped["election_type"] == method]
            .set_index(condition)
            .reindex(categories)["minimum_separation"]
        )
        axis.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            values,
            width,
            label=method,
        )
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.text(0.5, 0.925, description, ha="center", wrap=True)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Mean minimum separation (m)")
    axis.set_xticks(x, categories, rotation=30)
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    """Create safety CSV summaries and graphs."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metrics = _prepare_metrics(_load_metrics())
    summary = _method_summary(metrics)
    condition_summary = summarize_by_conditions(
        metrics,
        {
            "run_count": ("run_id", "count"),
            "safety_success_rate": ("safety_success", "mean"),
            "mean_min_separation": ("min_separation", "mean"),
            "p05_min_separation": (
                "min_separation",
                lambda values: values.quantile(0.05),
            ),
            "mean_min_separation_time": (
                "min_separation_time",
                "mean",
            ),
            "mean_close_steps": ("n_close_steps", "mean"),
            "mean_baseline_min_separation": (
                "baseline_min_separation",
                "mean",
            ),
            "mean_separation_change_from_baseline": (
                "separation_change_from_baseline",
                "mean",
            ),
        },
    )
    condition_summary["safety_success_rate"] *= 100.0

    metrics.to_csv(
        OUTPUT_DIRECTORY / "safety_run_results.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIRECTORY / "safety_method_summary.csv",
        index=False,
    )
    condition_summary.to_csv(
        OUTPUT_DIRECTORY / "safety_by_condition.csv",
        index=False,
    )
    _write_conclusion(summary)

    _save_bar_chart(
        summary,
        [
            ("mean_min_separation", "Failed run"),
            ("mean_baseline_min_separation", "Baseline"),
        ],
        "Minimum separation by election method",
        "Higher minimum separation indicates a larger safety margin.",
        "Minimum separation (m)",
        "minimum_separation_by_method.png",
    )
    _save_bar_chart(
        summary,
        [
            ("mean_close_steps", "Mean close steps"),
            ("total_close_steps", "Total close steps"),
        ],
        "Safety violations by election method",
        "Fewer timesteps below the safety threshold indicate safer behavior.",
        "Close timesteps",
        "safety_violations_by_method.png",
    )
    _save_bar_chart(
        summary,
        [
            ("safety_success_rate", "Safety success rate"),
        ],
        "Safety success rate by election method",
        "A run is successful when it has no timesteps below the safety "
        "distance threshold.",
        "Successful runs (%)",
        "safety_success_rate_by_method.png",
    )
    _save_boxplot(
        metrics,
        "separation_change_from_baseline",
        "Safety degradation relative to baseline",
        "Values below zero indicate that the failed run reduced minimum "
        "separation compared with its non-failure baseline.",
        "Failed minimum separation - baseline (m)",
        "safety_degradation_by_method.png",
    )
    _save_condition_plot(
        metrics,
        "shape",
        "Safety by formation shape",
        "Higher minimum separation indicates better safety margins.",
        "Formation shape",
        "safety_by_shape.png",
    )
    _save_condition_plot(
        metrics,
        "spread",
        "Safety by formation spread",
        "Shows how formation spacing affects minimum separation.",
        "Formation spread (m)",
        "safety_by_spread.png",
    )
    _save_condition_plot(
        metrics,
        "number_of_drone",
        "Safety by swarm size",
        "Shows how the number of drones affects minimum separation.",
        "Number of drones",
        "safety_by_swarm_size.png",
    )
    _save_condition_plot(
        metrics,
        "difficulty_band",
        "Safety by trajectory difficulty",
        "Compares safety margins across easy-to-hard trajectories.",
        "Trajectory difficulty",
        "safety_by_difficulty.png",
    )
    print(f"Wrote safety CSV files and graphs to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
