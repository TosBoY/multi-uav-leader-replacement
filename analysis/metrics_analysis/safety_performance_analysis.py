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
    """Save a safety boxplot grouped by election method."""
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
    axis.boxplot(groups, tick_labels=labels)
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.text(0.5, 0.925, description, ha="center", wrap=True)
    axis.set_xlabel("Election method")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
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
