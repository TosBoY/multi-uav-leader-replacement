"""Summarize failure-induced trajectory deviation and create comparison plots."""

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


OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "analysis" / "trajectory_deviation"

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
    "mean_error",
    "peak_error",
    "peak_error_time",
    "error_integral_sum",
    "final_error_max",
]

DRONE_COLUMNS = [
    "run_id",
    "drone",
    "role",
    "mean_error",
    "rms_error",
    "peak_error",
    "peak_error_time",
    "error_integral",
    "final_error",
]


def _load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load run-level and per-drone deviation metrics."""
    run_metrics = load_run_metrics(RUN_COLUMNS)
    drone_metrics = load_drone_metrics(DRONE_COLUMNS)
    return run_metrics, drone_metrics


def _method_summary(
    run_metrics: pd.DataFrame,
    drone_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create one deviation summary row for each election method."""
    run_summary = (
        run_metrics.groupby("election_type", as_index=False)
        .agg(
            run_count=("run_id", "count"),
            mean_error=("mean_error", "mean"),
            peak_error=("peak_error", "mean"),
            p95_peak_error=("peak_error", lambda values: values.quantile(0.95)),
            mean_peak_error_time=("peak_error_time", "mean"),
            mean_error_integral=("error_integral_sum", "mean"),
            mean_final_error=("final_error_max", "mean"),
        )
    )
    drone_summary = (
        drone_metrics.merge(
            run_metrics[["run_id", "election_type"]],
            on="run_id",
            how="left",
        )
        .groupby("election_type", as_index=False)
        .agg(
            mean_drone_error=("mean_error", "mean"),
            mean_drone_rms_error=("rms_error", "mean"),
            mean_drone_peak_error=("peak_error", "mean"),
            mean_drone_error_integral=("error_integral", "mean"),
            mean_drone_final_error=("final_error", "mean"),
        )
    )
    return (
        run_summary.merge(drone_summary, on="election_type", how="left")
        .sort_values("election_type")
        .reset_index(drop=True)
    )


def _write_conclusion(summary: pd.DataFrame) -> None:
    """Write a detailed, data-driven trajectory-deviation conclusion."""
    if summary.empty:
        conclusion = "No trajectory-deviation data was available.\n"
    else:
        lowest_mean = summary.loc[summary["mean_error"].idxmin()]
        lowest_peak = summary.loc[summary["p95_peak_error"].idxmin()]
        lowest_drone_rms = summary.loc[
            summary["mean_drone_rms_error"].idxmin()
        ]
        lines = [
            "TRAJECTORY-DEVIATION ANALYSIS CONCLUSION",
            "=========================================",
            "",
            "Overall scope",
            f"- Failed runs analysed: {int(summary['run_count'].sum())}",
            f"- Election methods compared: {len(summary)}",
            "",
            "Method comparison",
            (
                f"- Lowest mean trajectory deviation: "
                f"{lowest_mean['election_type']} "
                f"({lowest_mean['mean_error']:.3f} m)"
            ),
            (
                f"- Lowest 95th-percentile peak deviation: "
                f"{lowest_peak['election_type']} "
                f"({lowest_peak['p95_peak_error']:.3f} m)"
            ),
            (
                f"- Lowest mean per-drone RMS error: "
                f"{lowest_drone_rms['election_type']} "
                f"({lowest_drone_rms['mean_drone_rms_error']:.3f} m)"
            ),
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['election_type']}: runs={int(row['run_count'])}, "
                f"mean error={row['mean_error']:.3f} m, "
                f"mean peak error={row['peak_error']:.3f} m, "
                f"p95 peak error={row['p95_peak_error']:.3f} m, "
                f"mean peak-error time={row['mean_peak_error_time']:.3f} s, "
                f"error integral={row['mean_error_integral']:.3f}, "
                f"final error={row['mean_final_error']:.3f} m, "
                f"per-drone RMS={row['mean_drone_rms_error']:.3f} m, "
                f"per-drone peak={row['mean_drone_peak_error']:.3f} m, "
                f"per-drone final={row['mean_drone_final_error']:.3f} m"
            )
        lines.extend(
            [
                "",
                "Interpretation",
                (
                    "Lower mean and peak errors indicate closer tracking of "
                    "the non-failure baseline. The 95th-percentile peak error "
                    "describes a high-deviation case while limiting the "
                    "influence of the single most extreme run."
                ),
            ]
        )
        conclusion = "\n".join(lines) + "\n"
    (OUTPUT_DIRECTORY / "trajectory_deviation_conclusion.txt").write_text(
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


def _save_boxplot(
    metrics: pd.DataFrame,
    column: str,
    title: str,
    description: str,
    ylabel: str,
    filename: str,
) -> None:
    """Save a dense-data-friendly deviation distribution plot by method."""
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
    """Save mean trajectory deviation by method and scenario condition."""
    grouped = (
        metrics.groupby(["election_type", condition], as_index=False)
        .agg(deviation=("mean_error", "mean"))
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
            .reindex(categories)["deviation"]
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
    axis.set_ylabel("Mean deviation (m)")
    axis.set_xticks(x, categories, rotation=30)
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    """Create trajectory-deviation CSV summaries and graphs."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    run_metrics, drone_metrics = _load_metrics()
    method_summary = _method_summary(run_metrics, drone_metrics)
    drone_summary = drone_metrics.groupby("run_id", as_index=False).agg(
        mean_drone_rms_error=("rms_error", "mean"),
        mean_drone_peak_error=("peak_error", "mean"),
        mean_drone_error_integral=("error_integral", "mean"),
        mean_drone_final_error=("final_error", "mean"),
    )
    run_metrics = run_metrics.merge(
        drone_summary,
        on="run_id",
        how="left",
    )
    condition_metrics = run_metrics.merge(
        drone_summary,
        on="run_id",
        how="left",
        suffixes=("", "_duplicate"),
    )
    condition_summary = summarize_by_conditions(
        condition_metrics,
        {
            "run_count": ("run_id", "count"),
            "mean_error": ("mean_error", "mean"),
            "mean_peak_error": ("peak_error", "mean"),
            "mean_error_integral": (
                "error_integral_sum",
                "mean",
            ),
            "mean_final_error": ("final_error_max", "mean"),
            "mean_drone_rms_error": (
                "mean_drone_rms_error",
                "mean",
            ),
        },
    )

    run_metrics.to_csv(
        OUTPUT_DIRECTORY / "trajectory_deviation_run_results.csv",
        index=False,
    )
    method_summary.to_csv(
        OUTPUT_DIRECTORY / "trajectory_deviation_method_summary.csv",
        index=False,
    )
    condition_summary.to_csv(
        OUTPUT_DIRECTORY / "trajectory_deviation_by_condition.csv",
        index=False,
    )
    _write_conclusion(method_summary)

    _save_bar_chart(
        method_summary,
        [
            ("mean_error", "Mean error"),
            ("mean_drone_rms_error", "Mean per-drone RMS error"),
        ],
        "Overall trajectory deviation by election method",
        "Lower values indicate less average deviation from the non-failure "
        "baseline trajectory.",
        "Deviation (m)",
        "overall_deviation_by_method.png",
    )
    _save_bar_chart(
        method_summary,
        [
            ("peak_error", "Mean peak error"),
            ("p95_peak_error", "95th percentile peak error"),
        ],
        "Peak trajectory deviation by election method",
        "Lower values indicate smaller worst-case deviations from the "
        "non-failure baseline.",
        "Peak deviation (m)",
        "peak_deviation_by_method.png",
    )
    _save_bar_chart(
        method_summary,
        [
            ("mean_error_integral", "Error integral"),
            ("mean_final_error", "Final error"),
        ],
        "Accumulated and final deviation by election method",
        "The integral measures accumulated deviation; final error measures "
        "the remaining deviation at the end of the analysis window.",
        "Deviation",
        "accumulated_final_deviation_by_method.png",
    )
    _save_boxplot(
        drone_metrics.merge(
            run_metrics[["run_id", "election_type"]],
            on="run_id",
            how="left",
        ),
        "error_integral",
        "Per-drone accumulated deviation by election method",
        "Each value is the accumulated post-failure deviation for one drone "
        "relative to its non-failure baseline.",
        "Error integral",
        "per_drone_error_integral_by_method.png",
    )
    _save_condition_plot(
        run_metrics,
        "shape",
        "Trajectory deviation by formation shape",
        "Lower values indicate less deviation from the non-failure baseline.",
        "Formation shape",
        "deviation_by_shape.png",
    )
    _save_condition_plot(
        run_metrics,
        "spread",
        "Trajectory deviation by formation spread",
        "Shows how formation spacing affects failure-induced deviation.",
        "Formation spread (m)",
        "deviation_by_spread.png",
    )
    _save_condition_plot(
        run_metrics,
        "number_of_drone",
        "Trajectory deviation by swarm size",
        "Shows how the number of drones affects failure-induced deviation.",
        "Number of drones",
        "deviation_by_swarm_size.png",
    )
    _save_condition_plot(
        run_metrics,
        "difficulty_band",
        "Trajectory deviation by difficulty",
        "Compares deviation across easy-to-hard trajectory difficulty bands.",
        "Trajectory difficulty",
        "deviation_by_difficulty.png",
    )
    print(f"Wrote deviation CSV files and graphs to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
