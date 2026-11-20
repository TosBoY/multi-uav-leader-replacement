"""Analyze leader-replacement robustness across experimental conditions."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

try:
    from ._common import PROJECT_ROOT, load_run_metrics
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from analysis.metrics_analysis._common import PROJECT_ROOT, load_run_metrics


OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "analysis" / "robustness"

RUN_COLUMNS = [
    "run_id",
    "path_id",
    "number_of_drone",
    "shape",
    "spread",
    "plane",
    "election_type",
    "fail_time",
    "mean_error",
    "peak_error",
    "settle_time_01",
    "n_matched_01",
    "n_never_matched_01",
    "n_unstable_01",
    "min_separation",
    "n_close_steps",
]

def _load_metrics() -> pd.DataFrame:
    """Load run metrics and post-failure trajectory difficulty."""
    return load_run_metrics(RUN_COLUMNS)


def _prepare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add normalized recovery and safety values."""
    result = metrics.copy()
    survivor_count = result["number_of_drone"] - 1
    result["recovery_success_01"] = result["settle_time_01"].notna()
    result["matched_percentage_01"] = (
        100.0 * result["n_matched_01"] / survivor_count
    )
    result["safety_success"] = result["n_close_steps"].fillna(0).eq(0)
    return result


def _group_summary(
    metrics: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarize recovery, deviation, and safety by condition."""
    return (
        metrics.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            run_count=("run_id", "count"),
            recovery_success_rate=("recovery_success_01", "mean"),
            mean_settle_time=("settle_time_01", "mean"),
            median_settle_time=("settle_time_01", "median"),
            mean_matched_percentage=("matched_percentage_01", "mean"),
            mean_error=("mean_error", "mean"),
            mean_peak_error=("peak_error", "mean"),
            mean_min_separation=("min_separation", "mean"),
            mean_close_steps=("n_close_steps", "mean"),
            safety_success_rate=("safety_success", "mean"),
        )
        .assign(
            recovery_success_rate=lambda frame: (
                frame["recovery_success_rate"] * 100.0
            ),
            safety_success_rate=lambda frame: (
                frame["safety_success_rate"] * 100.0
            ),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def _write_conclusion(summary: pd.DataFrame) -> None:
    """Write a detailed, data-driven robustness conclusion."""
    if summary.empty:
        conclusion = "No robustness data was available.\n"
    else:
        def percent(value: object) -> str:
            return "N/A" if pd.isna(value) else f"{float(value):.1f}%"

        best_recovery = summary.loc[
            summary["recovery_success_rate"].idxmax()
        ]
        valid_median = summary.dropna(subset=["median_settle_time"])
        lines = [
            "ROBUSTNESS ANALYSIS CONCLUSION",
            "==============================",
            "",
            "Overall scope",
            f"- Failed runs analysed: {int(summary['run_count'].sum())}",
            f"- Election methods compared: {len(summary)}",
            "",
            "Method comparison",
            (
                f"- Best recovery success: "
                f"{best_recovery['election_type']} "
                f"({percent(best_recovery['recovery_success_rate'])})"
            ),
        ]
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['election_type']}: runs={int(row['run_count'])}, "
                f"recovery success={percent(row['recovery_success_rate'])}, "
                f"safety success={percent(row['safety_success_rate'])}, "
                f"median settle={row['median_settle_time']:.3f} s, "
                f"mean settle={row['mean_settle_time']:.3f} s, "
                f"matched={row['mean_matched_percentage']:.1f}%, "
                f"mean error={row['mean_error']:.3f} m, "
                f"mean peak error={row['mean_peak_error']:.3f} m, "
                f"minimum separation={row['mean_min_separation']:.3f} m, "
                f"close steps={row['mean_close_steps']:.3f}"
            )
        if not valid_median.empty:
            fastest = valid_median.loc[
                valid_median["median_settle_time"].idxmin()
            ]
            lines.append(
                f"- Fastest median recovery: "
                f"{fastest['election_type']} "
                f"({fastest['median_settle_time']:.3f} s)."
            )
        safest = summary.loc[summary["safety_success_rate"].idxmax()]
        lines.append(
            f"- Best safety success: "
            f"{safest['election_type']} "
            f"({percent(safest['safety_success_rate'])})"
        )
        lines.extend(
            [
                "",
                "Interpretation",
                (
                    "A robust election method should combine high recovery "
                    "and safety success with low recovery time, trajectory "
                    "error, close-step count, and stable minimum separation."
                ),
            ]
        )
        conclusion = "\n".join(lines) + "\n"
    (OUTPUT_DIRECTORY / "robustness_conclusion.txt").write_text(
        conclusion,
        encoding="utf-8",
    )


def _save_line_plot(
    metrics: pd.DataFrame,
    x_column: str,
    title: str,
    description: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    """Save a method comparison line plot for one numeric condition."""
    grouped = (
        metrics.groupby(["election_type", x_column], as_index=False)
        .agg(
            recovery_time=("settle_time_01", "median"),
            recovery_success_rate=("recovery_success_01", "mean"),
        )
        .sort_values(x_column)
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    for method, group in grouped.groupby("election_type", sort=True):
        axis.plot(
            group[x_column],
            group["recovery_time"],
            marker="o",
            label=method,
        )
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.text(0.5, 0.925, description, ha="center", wrap=True)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if x_column == "number_of_drone":
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=180)
    plt.close(figure)


def _save_category_plot(
    metrics: pd.DataFrame,
    category: str,
    title: str,
    description: str,
    xlabel: str,
    filename: str,
) -> None:
    """Save a grouped recovery-time plot for a categorical condition."""
    grouped = (
        metrics.groupby(["election_type", category], as_index=False)
        .agg(recovery_time=("settle_time_01", "median"))
    )
    categories = sorted(grouped[category].dropna().unique())
    methods = sorted(grouped["election_type"].dropna().unique())
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(methods))
    figure, axis = plt.subplots(figsize=(10, 6))
    for index, method in enumerate(methods):
        values = (
            grouped[grouped["election_type"] == method]
            .set_index(category)
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


def _save_difficulty_scatter(metrics: pd.DataFrame) -> None:
    """Save binned recovery-time distributions against trajectory difficulty."""
    valid = metrics.dropna(
        subset=["difficulty_score", "settle_time_01"],
    ).copy()
    if valid.empty:
        return
    bin_count = min(10, valid["difficulty_score"].nunique())
    if bin_count > 1:
        valid["difficulty_bin"] = pd.qcut(
            valid["difficulty_score"],
            q=bin_count,
            duplicates="drop",
        )
    else:
        valid["difficulty_bin"] = "all"
    grouped = (
        valid.groupby(["election_type", "difficulty_bin"], observed=True)
        .agg(
            difficulty=("difficulty_score", "median"),
            recovery_time=("settle_time_01", "median"),
            lower=("settle_time_01", lambda values: values.quantile(0.25)),
            upper=("settle_time_01", lambda values: values.quantile(0.75)),
        )
        .reset_index()
        .sort_values("difficulty")
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    for method, group in grouped.groupby("election_type", sort=True):
        axis.plot(
            group["difficulty"],
            group["recovery_time"],
            marker="o",
            linewidth=2,
            label=method,
        )
        axis.fill_between(
            group["difficulty"],
            group["lower"],
            group["upper"],
            alpha=0.15,
        )
    figure.suptitle(
        "Recovery time versus trajectory difficulty",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Lines show median recovery time in difficulty quantile bins; "
        "shaded areas show the interquartile range.",
        ha="center",
        wrap=True,
    )
    axis.set_xlabel("Trajectory difficulty score")
    axis.set_ylabel("Settle time (s)")
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(
        OUTPUT_DIRECTORY / "recovery_time_vs_difficulty.png",
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    """Create robustness CSV summaries and graphs."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metrics = _prepare_metrics(_load_metrics())

    condition_columns = [
        "election_type",
        "number_of_drone",
        "shape",
        "spread",
        "plane",
        "path_id",
    ]
    condition_summary = _group_summary(metrics, condition_columns)
    method_summary = _group_summary(metrics, ["election_type"])
    difficulty_summary = _group_summary(
        metrics,
        ["election_type", "difficulty_score"],
    )

    metrics.to_csv(
        OUTPUT_DIRECTORY / "robustness_run_results.csv",
        index=False,
    )
    condition_summary.to_csv(
        OUTPUT_DIRECTORY / "robustness_by_condition.csv",
        index=False,
    )
    method_summary.to_csv(
        OUTPUT_DIRECTORY / "robustness_method_summary.csv",
        index=False,
    )
    difficulty_summary.to_csv(
        OUTPUT_DIRECTORY / "robustness_by_difficulty.csv",
        index=False,
    )
    _write_conclusion(method_summary)

    _save_line_plot(
        metrics,
        "number_of_drone",
        "Robustness across swarm size",
        "Lower recovery time indicates faster recovery as the number of "
        "drones changes.",
        "Number of drones",
        "Median settle time (s)",
        "recovery_time_by_swarm_size.png",
    )
    _save_category_plot(
        metrics,
        "shape",
        "Robustness across formation shape",
        "Lower values indicate faster recovery for each formation shape.",
        "Formation shape",
        "recovery_time_by_formation_shape.png",
    )
    _save_category_plot(
        metrics,
        "spread",
        "Robustness across formation spread",
        "Lower values indicate faster recovery at each formation spread.",
        "Formation spread (m)",
        "recovery_time_by_formation_spread.png",
    )
    _save_category_plot(
        metrics,
        "difficulty_band",
        "Robustness across trajectory difficulty",
        "Lower values indicate faster recovery across trajectory difficulty.",
        "Trajectory difficulty",
        "recovery_time_by_difficulty.png",
    )
    _save_difficulty_scatter(metrics)

    print(f"Wrote robustness CSV files and graphs to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
