"""Summarize leader-replacement recovery metrics and create comparison plots."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ._common import (
        PROJECT_ROOT,
        load_drone_metrics,
        load_run_metrics,
        summarize_by_conditions,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from analysis.metrics_analysis._common import (
        PROJECT_ROOT,
        load_drone_metrics,
        load_run_metrics,
        summarize_by_conditions,
    )


OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "analysis" / "recovery_performance"

RUN_COLUMNS = [
    "run_id",
    "baseline_nonfail_run_id",
    "path_id",
    "number_of_drone",
    "shape",
    "spread",
    "plane",
    "election_type",
    "fail_time",
    "settle_time_01",
    "settle_time_002",
    "n_matched_01",
    "n_never_matched_01",
    "n_unstable_01",
    "n_matched_002",
    "n_never_matched_002",
    "n_unstable_002",
    "n_close_steps",
]

DRONE_COLUMNS = [
    "run_id",
    "excursions_01",
    "excursions_002",
]


def _load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the run-level and drone-level recovery metrics."""
    run_metrics = load_run_metrics(RUN_COLUMNS)
    drone_metrics = load_drone_metrics(DRONE_COLUMNS)
    return run_metrics, drone_metrics


def _prepare_run_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add success flags and normalized percentages to each run."""
    result = metrics.copy()
    survivor_count = result["number_of_drone"] - 1
    result["recovery_success_01"] = result["settle_time_01"].notna()
    result["recovery_success_002"] = result["settle_time_002"].notna()
    result["matched_percentage_01"] = (
        100.0 * result["n_matched_01"] / survivor_count
    )
    result["matched_percentage_002"] = (
        100.0 * result["n_matched_002"] / survivor_count
    )
    result["unstable_percentage_01"] = (
        100.0 * result["n_unstable_01"] / survivor_count
    )
    result["unstable_percentage_002"] = (
        100.0 * result["n_unstable_002"] / survivor_count
    )
    result["never_matched_percentage_01"] = (
        100.0 * result["n_never_matched_01"] / survivor_count
    )
    result["never_matched_percentage_002"] = (
        100.0 * result["n_never_matched_002"] / survivor_count
    )
    result["safety_success"] = result["n_close_steps"].fillna(0).eq(0)
    return result


def _prepare_drone_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-drone excursions to one row per failed run."""
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "mean_excursions_01",
                "mean_excursions_002",
            ]
        )
    return (
        metrics.groupby("run_id", as_index=False)
        .agg(
            mean_excursions_01=("excursions_01", "mean"),
            mean_excursions_002=("excursions_002", "mean"),
        )
    )


def _method_summary(
    run_metrics: pd.DataFrame,
    drone_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row for each election method."""
    run_metrics = run_metrics.merge(drone_metrics, on="run_id", how="left")
    summary = (
        run_metrics.groupby("election_type", as_index=False)
        .agg(
            run_count=("run_id", "count"),
            recovery_success_rate_01=("recovery_success_01", "mean"),
            recovery_success_rate_002=("recovery_success_002", "mean"),
            mean_settle_time_01=("settle_time_01", "mean"),
            median_settle_time_01=("settle_time_01", "median"),
            p95_settle_time_01=(
                "settle_time_01",
                lambda values: values.quantile(0.95),
            ),
            mean_settle_time_002=("settle_time_002", "mean"),
            median_settle_time_002=("settle_time_002", "median"),
            p95_settle_time_002=(
                "settle_time_002",
                lambda values: values.quantile(0.95),
            ),
            mean_matched_percentage_01=("matched_percentage_01", "mean"),
            mean_matched_percentage_002=("matched_percentage_002", "mean"),
            mean_unstable_percentage_01=("unstable_percentage_01", "mean"),
            mean_unstable_percentage_002=("unstable_percentage_002", "mean"),
            mean_never_matched_percentage_01=(
                "never_matched_percentage_01",
                "mean",
            ),
            mean_never_matched_percentage_002=(
                "never_matched_percentage_002",
                "mean",
            ),
            mean_excursions_01=("mean_excursions_01", "mean"),
            mean_excursions_002=("mean_excursions_002", "mean"),
        )
    )
    summary["recovery_success_rate_01"] *= 100.0
    summary["recovery_success_rate_002"] *= 100.0
    return summary.sort_values("election_type").reset_index(drop=True)


def _save_boxplot(
    metrics: pd.DataFrame,
    column: str,
    title: str,
    description: str,
    ylabel: str,
    filename: str,
) -> None:
    """Save a boxplot grouped by election method."""
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


def _save_bar_chart(
    summary: pd.DataFrame,
    columns: list[tuple[str, str]],
    title: str,
    description: str,
    ylabel: str,
    filename: str,
) -> None:
    """Save a grouped bar chart from the method summary."""
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


def _save_condition_plot(
    metrics: pd.DataFrame,
    condition: str,
    title: str,
    description: str,
    xlabel: str,
    filename: str,
) -> None:
    """Save median recovery time by method and scenario condition."""
    grouped = (
        metrics.groupby(["election_type", condition], as_index=False)
        .agg(recovery_time=("settle_time_01", "median"))
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
            .reindex(categories)["recovery_time"]
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
    axis.set_ylabel("Median settle time (s)")
    axis.set_xticks(x, categories, rotation=30)
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    """Create recovery CSV summaries and graphs."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    run_metrics, raw_drone_metrics = _load_metrics()
    run_metrics = _prepare_run_metrics(run_metrics)
    drone_metrics = _prepare_drone_metrics(raw_drone_metrics)
    summary = _method_summary(run_metrics, drone_metrics)
    condition_metrics = run_metrics.merge(
        drone_metrics,
        on="run_id",
        how="left",
    )
    condition_summary = summarize_by_conditions(
        condition_metrics,
        {
            "run_count": ("run_id", "count"),
            "recovery_success_rate_01": (
                "recovery_success_01",
                "mean",
            ),
            "median_settle_time_01": ("settle_time_01", "median"),
            "mean_matched_percentage_01": (
                "matched_percentage_01",
                "mean",
            ),
            "mean_unstable_percentage_01": (
                "unstable_percentage_01",
                "mean",
            ),
            "mean_never_matched_percentage_01": (
                "never_matched_percentage_01",
                "mean",
            ),
            "mean_excursions_01": ("mean_excursions_01", "mean"),
            "safety_success_rate": ("safety_success", "mean"),
        },
    )
    condition_summary["recovery_success_rate_01"] *= 100.0
    condition_summary["safety_success_rate"] *= 100.0

    run_metrics.to_csv(
        OUTPUT_DIRECTORY / "recovery_run_results.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIRECTORY / "recovery_method_summary.csv",
        index=False,
    )
    condition_summary.to_csv(
        OUTPUT_DIRECTORY / "recovery_by_condition.csv",
        index=False,
    )

    _save_boxplot(
        run_metrics,
        "settle_time_01",
        "Recovery time by election method",
        "Time until all surviving drones remain within the loose baseline "
        "error tolerance.",
        "Settle time (s)",
        "recovery_time_by_method.png",
    )
    _save_bar_chart(
        summary,
        [
            ("recovery_success_rate_01", "Loose tolerance"),
            ("recovery_success_rate_002", "Tight tolerance"),
        ],
        "Recovery success rate by election method",
        "Percentage of runs where every surviving drone reached the loose "
        "or tight recovery tolerance.",
        "Successful runs (%)",
        "recovery_success_rate_by_method.png",
    )
    _save_bar_chart(
        summary,
        [
            ("mean_matched_percentage_01", "Matched"),
            ("mean_unstable_percentage_01", "Unstable"),
            ("mean_never_matched_percentage_01", "Never matched"),
        ],
        "Recovery status by election method",
        "Average percentage of surviving drones classified as matched, "
        "unstable, or never matched.",
        "Drones (%)",
        "recovery_status_by_method.png",
    )
    _save_boxplot(
        run_metrics.merge(drone_metrics, on="run_id", how="left"),
        "mean_excursions_01",
        "Recovery excursions by election method",
        "Average number of times a drone leaves the loose tolerance after "
        "first matching it.",
        "Mean excursions per drone",
        "recovery_excursions_by_method.png",
    )
    _save_condition_plot(
        run_metrics,
        "shape",
        "Recovery time by formation shape",
        "Lower values indicate faster recovery for each formation shape.",
        "Formation shape",
        "recovery_time_by_shape.png",
    )
    _save_condition_plot(
        run_metrics,
        "spread",
        "Recovery time by formation spread",
        "Shows how the distance between formation members affects recovery.",
        "Formation spread (m)",
        "recovery_time_by_spread.png",
    )
    _save_condition_plot(
        run_metrics,
        "number_of_drone",
        "Recovery time by swarm size",
        "Shows how the number of drones affects recovery.",
        "Number of drones",
        "recovery_time_by_swarm_size.png",
    )
    _save_condition_plot(
        run_metrics,
        "difficulty_band",
        "Recovery time by trajectory difficulty",
        "Compares recovery across easy-to-hard trajectory difficulty bands.",
        "Trajectory difficulty",
        "recovery_time_by_difficulty.png",
    )

    print(f"Wrote CSV files and graphs to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
