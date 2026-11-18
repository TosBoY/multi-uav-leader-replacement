"""Classify failed runs into mission-success tiers and rank them."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

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


def main() -> None:
    """Load failed runs, classify them, and write ranked CSV outputs."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    metrics = rank_mission_runs(load_run_metrics(RUN_COLUMNS))
    metrics.to_csv(OUTPUT_DIRECTORY / "mission_success_run_results.csv", index=False)
    _method_summary(metrics).to_csv(
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
    print(f"Wrote mission-success results to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
