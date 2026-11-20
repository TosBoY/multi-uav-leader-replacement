"""Classify failed runs into mission-success tiers and rank them."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ._common import (
        PROJECT_ROOT,
        load_run_metrics,
        summarize_by_conditions,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from analysis.metrics_analysis._common import (
        PROJECT_ROOT,
        load_run_metrics,
        summarize_by_conditions,
    )


OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "analysis" / "mission_success"

# The test vehicle is a Crazyflie-like quadrotor. Its body radius is taken as
# 0.25 m, so the absolute safety distance between vehicle centres is 0.50 m.
DRONE_BODY_SIZE_M = 0.25
SAFETY_DISTANCE_M = 2.0 * DRONE_BODY_SIZE_M
T_MAX_S = 4.0
T_MAX = T_MAX_S
SAFETY_DISTANCE = SAFETY_DISTANCE_M

TIER_NAMES = {
    0: "Unsafe",
    1: "No recovery",
    2: "Partial recovery",
    3: "Full, coarse",
    4: "Full, fine",
}

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
    "n_matched_01",
    "n_never_matched_01",
    "n_unstable_01",
    "time_to_match_max_01",
    "settle_time_01",
    "n_matched_002",
    "n_never_matched_002",
    "n_unstable_002",
    "time_to_match_max_002",
    "settle_time_002",
    "error_integral_sum",
    "min_separation",
]


def _finite_number(value: object) -> float | None:
    """Convert a scalar to a finite float, or return None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and number == number else None


def _count(value: object) -> int:
    """Convert a metric count to an integer; missing counts are zero."""
    number = _finite_number(value)
    return int(number) if number is not None else 0


def _full_recovery(row: Mapping[str, object], suffix: str) -> bool:
    """Return True only when every survivor is stably and timely matched."""
    try:
        survivor_count = int(row["number_of_drone"]) - 1
    except (KeyError, TypeError, ValueError):
        return False
    if survivor_count < 1:
        return False

    matched = _count(row.get(f"n_matched_{suffix}", 0))
    never_matched = _count(row.get(f"n_never_matched_{suffix}", 0))
    unstable = _count(row.get(f"n_unstable_{suffix}", 0))
    time_to_match = _finite_number(row.get(f"time_to_match_max_{suffix}"))

    return (
        matched == survivor_count
        and never_matched == 0
        and unstable == 0
        and time_to_match is not None
        and time_to_match <= T_MAX_S
    )


def _timely_recovery(row: Mapping[str, object], suffix: str = "01") -> bool:
    """Return True if at least one drone recovered stably by the deadline."""
    matched = _count(row.get(f"n_matched_{suffix}", 0))
    time_to_match = _finite_number(row.get(f"time_to_match_max_{suffix}"))
    return (
        matched > 0
        and time_to_match is not None
        and time_to_match <= T_MAX_S
    )


def classify_mission_tier(row: Mapping[str, object]) -> int:
    """Apply the safety-first mission-success tier rules to one run."""
    min_separation = _finite_number(row.get("min_separation"))
    if min_separation is None or min_separation < SAFETY_DISTANCE_M:
        return 0

    coarse_full = _full_recovery(row, "01")
    if not coarse_full:
        if not _timely_recovery(row, "01"):
            return 1
        return 2

    if _full_recovery(row, "002"):
        return 4
    return 3


def _classify_mission_tier(row: Mapping[str, object]) -> int:
    """Backward-compatible shorthand for mission tier classification."""
    return classify_mission_tier(row)


def assign_mission_tier(row: Mapping[str, object]) -> int:
    """Public alias for classifying a single run."""
    return classify_mission_tier(row)


def classify_mission_success(row: Mapping[str, object]) -> dict[str, object]:
    """Return tier and recovery flags for one failed run."""
    tier = classify_mission_tier(row)
    return {
        "mission_tier": tier,
        "mission_tier_name": TIER_NAMES[tier],
        "coarse_recovery": _full_recovery(row, "01"),
        "fine_recovery": _full_recovery(row, "002"),
    }


def add_mission_success_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add tier, tier name, and recovery flags to a run-level table."""
    result = metrics.copy()
    classifications = result.apply(classify_mission_success, axis=1)
    for column in (
        "mission_tier",
        "mission_tier_name",
        "coarse_recovery",
        "fine_recovery",
    ):
        result[column] = classifications.map(lambda value: value[column])
    result["tier"] = result["mission_tier"]
    result["tier_name"] = result["mission_tier_name"]
    return result


def _prepare_run_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Alias for preparing mission-success run metrics."""
    return add_mission_success_columns(metrics)


def _prepare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Alias for preparing mission-success run metrics."""
    return add_mission_success_columns(metrics)


def _load_metrics() -> pd.DataFrame:
    """Load run-level metrics required for mission tier classification."""
    return load_run_metrics(RUN_COLUMNS)


def rank_mission_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank runs from tier 4 to tier 0, then by settle time and integral."""
    result = add_mission_success_columns(metrics)
    result["_settle_sort"] = result.apply(
        lambda row: (
            _finite_number(row["settle_time_002"])
            if int(row["mission_tier"]) == 4
            else _finite_number(row["settle_time_01"])
        ),
        axis=1,
    )
    result["_integral_sort"] = result["error_integral_sum"].apply(
        _finite_number
    )
    result = result.sort_values(
        ["mission_tier", "_settle_sort", "_integral_sort", "run_id"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    result["mission_rank"] = result.index + 1
    result["rank"] = result["mission_rank"]
    return result.drop(columns=["_settle_sort", "_integral_sort"])


def _method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize mission tiers and continuous metrics by election method."""
    return (
        metrics.groupby("election_type", as_index=False)
        .agg(
            run_count=("run_id", "count"),
            tier_4_count=("mission_tier", lambda values: (values == 4).sum()),
            tier_3_count=("mission_tier", lambda values: (values == 3).sum()),
            tier_2_count=("mission_tier", lambda values: (values == 2).sum()),
            tier_1_count=("mission_tier", lambda values: (values == 1).sum()),
            tier_0_count=("mission_tier", lambda values: (values == 0).sum()),
            tier_4_rate=("mission_tier", lambda values: 100.0 * (values == 4).mean()),
            tier_3_rate=("mission_tier", lambda values: 100.0 * (values == 3).mean()),
            full_recovery_rate=("coarse_recovery", lambda values: 100.0 * values.mean()),
            mean_settle_time_01=("settle_time_01", "mean"),
            mean_settle_time_002=("settle_time_002", "mean"),
            mean_error_integral_sum=("error_integral_sum", "mean"),
        )
        .sort_values("election_type")
        .reset_index(drop=True)
    )


def _condition_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize mission tiers by formation, swarm, and trajectory conditions."""
    return summarize_by_conditions(
        metrics,
        {
            "run_count": ("run_id", "count"),
            "tier_4_count": (
                "mission_tier",
                lambda values: int((values == 4).sum()),
            ),
            "tier_3_count": (
                "mission_tier",
                lambda values: int((values == 3).sum()),
            ),
            "tier_2_count": (
                "mission_tier",
                lambda values: int((values == 2).sum()),
            ),
            "tier_1_count": (
                "mission_tier",
                lambda values: int((values == 1).sum()),
            ),
            "tier_0_count": (
                "mission_tier",
                lambda values: int((values == 0).sum()),
            ),
            "tier_4_rate": (
                "mission_tier",
                lambda values: 100.0 * (values == 4).mean(),
            ),
            "tier_3_rate": (
                "mission_tier",
                lambda values: 100.0 * (values == 3).mean(),
            ),
            "tier_2_rate": (
                "mission_tier",
                lambda values: 100.0 * (values == 2).mean(),
            ),
            "tier_1_rate": (
                "mission_tier",
                lambda values: 100.0 * (values == 1).mean(),
            ),
            "tier_0_rate": (
                "mission_tier",
                lambda values: 100.0 * (values == 0).mean(),
            ),
            "full_recovery_rate": (
                "coarse_recovery",
                lambda values: 100.0 * values.mean(),
            ),
            "mean_settle_time_01": ("settle_time_01", "mean"),
            "mean_settle_time_002": ("settle_time_002", "mean"),
            "mean_error_integral_sum": ("error_integral_sum", "mean"),
        },
    )


def _write_conclusion(
    method_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
) -> None:
    """Write a detailed, data-driven mission-success conclusion."""
    if method_summary.empty:
        conclusion = "No mission-success data was available.\n"
    else:
        def percent(value: object) -> str:
            return "N/A" if pd.isna(value) else f"{float(value):.1f}%"

        best_fine = method_summary.loc[
            method_summary["tier_4_rate"].idxmax()
        ]
        best_coarse = method_summary.loc[
            method_summary["full_recovery_rate"].idxmax()
        ]
        tier_counts = [
            int(method_summary[f"tier_{tier}_count"].sum())
            for tier in range(5)
        ]
        total_runs = sum(tier_counts)
        tier_rates = [
            100.0 * count / total_runs if total_runs else 0.0
            for count in tier_counts
        ]
        conclusion_lines = [
            "MISSION-SUCCESS ANALYSIS CONCLUSION",
            "===================================",
            "",
            "Overall scope",
            f"- Failed runs analysed: {total_runs}",
            f"- Election methods compared: {len(method_summary)}",
            (
                "- Overall tier distribution: "
                + ", ".join(
                    f"Tier {tier}={percent(rate)}"
                    for tier, rate in enumerate(tier_rates)
                )
            ),
            "",
            "Method comparison",
            (
                f"- Best fine recovery: {best_fine['election_type']} "
                f"({percent(best_fine['tier_4_rate'])})"
            ),
            (
                f"- Best coarse recovery: {best_coarse['election_type']} "
                f"({percent(best_coarse['full_recovery_rate'])})"
            ),
        ]
        for _, row in method_summary.iterrows():
            conclusion_lines.append(
                f"- {row['election_type']}: runs={int(row['run_count'])}, "
                f"Tier 4={percent(row['tier_4_rate'])}, "
                f"Tier 3={percent(row['tier_3_rate'])}, "
                f"mean settle (coarse)="
                f"{row['mean_settle_time_01']:.3f} s, "
                f"mean settle (fine)="
                f"{row['mean_settle_time_002']:.3f} s, "
                f"mean error integral={row['mean_error_integral_sum']:.3f}"
            )
        valid_conditions = condition_summary.dropna(
            subset=["tier_4_rate"],
        )
        if not valid_conditions.empty:
            best_condition = valid_conditions.loc[
                valid_conditions["tier_4_rate"].idxmax()
            ]
            worst_condition = valid_conditions.loc[
                valid_conditions["tier_4_rate"].idxmin()
            ]
            conclusion_lines.extend(
                [
                    "",
                    "Condition comparison",
                    (
                        f"- Strongest fine-recovery condition: "
                        f"{best_condition['condition']}="
                        f"{best_condition['condition_value']} for "
                        f"{best_condition['election_type']} "
                        f"({percent(best_condition['tier_4_rate'])})"
                    ),
                    (
                        f"- Weakest fine-recovery condition: "
                        f"{worst_condition['condition']}="
                        f"{worst_condition['condition_value']} for "
                        f"{worst_condition['election_type']} "
                        f"({percent(worst_condition['tier_4_rate'])})"
                    ),
                ]
            )
        conclusion_lines.extend(
            [
                "",
                "Interpretation",
                f"{best_condition['condition']}="
                f"{best_condition['condition_value']} is the strongest "
                "observed condition for fine recovery."
                if not valid_conditions.empty
                else "- No condition-level conclusion was available.",
                (
                    "Higher Tier 3 and Tier 4 rates indicate more complete "
                    "and stable recovery, while Tier 0 indicates a safety "
                    "failure before recovery could be credited."
                ),
            ]
        )
        conclusion = "\n".join(conclusion_lines) + "\n"
    (OUTPUT_DIRECTORY / "mission_success_conclusion.txt").write_text(
        conclusion,
        encoding="utf-8",
    )


def _save_tier_distribution_by_method(metrics: pd.DataFrame) -> None:
    """Save the percentage of runs in each mission tier by election method."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if metrics.empty or "election_type" not in metrics:
        return

    grouped = (
        metrics.groupby("election_type", sort=True)["mission_tier"]
        .value_counts(normalize=True)
        .mul(100.0)
        .rename("percentage")
        .reset_index()
    )
    methods = sorted(grouped["election_type"].dropna().unique())
    if not methods:
        return

    x = np.arange(len(methods))
    width = 0.8
    bottom = np.zeros(len(methods))
    colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]

    figure, axis = plt.subplots(figsize=(10, 6))
    for tier in range(5):
        values = (
            grouped[grouped["mission_tier"] == tier]
            .set_index("election_type")
            .reindex(methods)["percentage"]
            .fillna(0.0)
            .to_numpy()
        )
        axis.bar(
            x,
            values,
            width,
            bottom=bottom,
            label=f"Tier {tier}: {TIER_NAMES[tier]}",
            color=colors[tier],
        )
        bottom += values

    figure.suptitle(
        "Mission-success tier distribution by election method",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Shows the proportion of failed runs in each safety and recovery tier.",
        ha="center",
        wrap=True,
    )
    axis.set_xlabel("Election method")
    axis.set_ylabel("Runs (%)")
    axis.set_xticks(x, methods, rotation=30)
    axis.set_ylim(0, 100)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.tight_layout(rect=(0, 0, 0.78, 0.88))
    figure.savefig(
        OUTPUT_DIRECTORY / "mission_tier_distribution_by_method.png",
        dpi=180,
    )
    plt.close(figure)


def _save_tier_rates_by_condition(
    metrics: pd.DataFrame,
    condition: str,
    tier: int,
    title: str,
    xlabel: str,
    filename: str,
) -> None:
    """Save one mission-tier rate by method and scenario condition."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if tier not in TIER_NAMES:
        raise ValueError(f"Unsupported mission tier: {tier}")
    if metrics.empty or condition not in metrics or "election_type" not in metrics:
        return

    grouped = (
        metrics.groupby(["election_type", condition], as_index=False)
        .agg(
            tier_rate=(
                "mission_tier",
                lambda values: 100.0 * (values == tier).mean(),
            ),
        )
    )
    categories = sorted(grouped[condition].dropna().unique())
    methods = sorted(grouped["election_type"].dropna().unique())
    if not categories or not methods:
        return

    x = np.arange(len(categories))
    width = 0.8 / len(methods)
    figure, axis = plt.subplots(figsize=(10, 6))
    for index, method in enumerate(methods):
        method_values = (
            grouped[grouped["election_type"] == method]
            .set_index(condition)
            .reindex(categories)
        )
        axis.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            method_values["tier_rate"].fillna(0.0),
            width,
            label=method,
        )
    figure.suptitle(title, fontsize=14, fontweight="bold")
    figure.text(
        0.5,
        0.925,
        f"{TIER_NAMES[tier]}: percentage of failed runs assigned to Tier {tier}.",
        ha="center",
        wrap=True,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(f"Tier {tier} runs (%)")
    axis.set_xticks(x, categories, rotation=30)
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=180)
    plt.close(figure)


def _save_tier_distribution_by_tier(metrics: pd.DataFrame) -> None:
    """Save one percentage bar chart for each mission tier by method."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if metrics.empty or "election_type" not in metrics:
        return

    grouped = (
        metrics.groupby("election_type", as_index=False)
        .agg(
            **{
                f"tier_{tier}_rate": (
                    "mission_tier",
                    lambda values, tier=tier: 100.0 * (values == tier).mean(),
                )
                for tier in range(5)
            }
        )
        .sort_values("election_type")
        .reset_index(drop=True)
    )
    if grouped.empty:
        return

    x = np.arange(len(grouped))
    for tier in range(5):
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.bar(
            x,
            grouped[f"tier_{tier}_rate"],
            color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"][tier],
        )
        figure.suptitle(
            f"Mission Tier {tier}: {TIER_NAMES[tier]} by election method",
            fontsize=14,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.925,
            f"Percentage of failed runs assigned to Tier {tier} "
            f"({TIER_NAMES[tier]}).",
            ha="center",
            wrap=True,
        )
        axis.set_xlabel("Election method")
        axis.set_ylabel(f"Tier {tier} runs (%)")
        axis.set_xticks(x, grouped["election_type"], rotation=30)
        axis.set_ylim(bottom=0)
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        figure.savefig(
            OUTPUT_DIRECTORY / f"mission_tier_{tier}_by_method.png",
            dpi=180,
        )
        plt.close(figure)


def _save_recovery_rates_by_method(summary: pd.DataFrame) -> None:
    """Save coarse and fine full-recovery rates by election method."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    figure, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    width = 0.36
    axis.bar(
        x - width / 2,
        summary["full_recovery_rate"],
        width,
        label="Coarse tolerance",
    )
    axis.bar(
        x + width / 2,
        summary["tier_4_rate"],
        width,
        label="Fine tolerance",
    )
    figure.suptitle(
        "Full-recovery rate by election method",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Compares the percentage of runs where every surviving drone fully "
        "recovered.",
        ha="center",
        wrap=True,
    )
    axis.set_xlabel("Election method")
    axis.set_ylabel("Successful runs (%)")
    axis.set_xticks(x, summary["election_type"], rotation=30)
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(
        OUTPUT_DIRECTORY / "full_recovery_rate_by_method.png",
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    """Load failed runs, classify them, and write ranked CSV outputs and plots."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metrics = rank_mission_runs(load_run_metrics(RUN_COLUMNS))
    metrics.to_csv(OUTPUT_DIRECTORY / "mission_success_run_results.csv", index=False)
    method_summary = _method_summary(metrics)
    method_summary.to_csv(
        OUTPUT_DIRECTORY / "mission_success_method_summary.csv",
        index=False,
    )
    condition_summary = _condition_summary(metrics)
    condition_summary.to_csv(
        OUTPUT_DIRECTORY / "mission_success_by_condition.csv",
        index=False,
    )
    (
        condition_summary
        .query("condition == 'number_of_drone'")
        .drop(columns="condition")
        .rename(columns={"condition_value": "number_of_drone"})
        .to_csv(
            OUTPUT_DIRECTORY / "mission_success_by_swarm_size.csv",
            index=False,
        )
    )
    _write_conclusion(method_summary, condition_summary)
    _save_tier_distribution_by_method(metrics)
    _save_tier_distribution_by_tier(metrics)
    _save_recovery_rates_by_method(method_summary)
    condition_specs = [
        (
            "number_of_drone",
            "mission-success rate across swarm size",
            "Number of drones",
            "swarm_size",
        ),
        (
            "shape",
            "mission-success rate across formation shape",
            "Formation shape",
            "formation_shape",
        ),
        (
            "spread",
            "mission-success rate across formation spread",
            "Formation spread (m)",
            "formation_spread",
        ),
        (
            "plane",
            "mission-success rate across trajectory plane",
            "Trajectory plane",
            "trajectory_plane",
        ),
        (
            "difficulty_band",
            "mission-success rate across trajectory difficulty",
            "Trajectory difficulty",
            "difficulty",
        ),
    ]
    for tier in range(5):
        for condition, description, xlabel, suffix in condition_specs:
            _save_tier_rates_by_condition(
                metrics,
                condition,
                tier,
                f"{TIER_NAMES[tier]} - Tier {tier} {description}",
                xlabel,
                f"mission_tier_{tier}_by_{suffix}.png",
            )
    print(f"Wrote mission-success results to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
