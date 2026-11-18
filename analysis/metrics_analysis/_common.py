"""Shared data loading helpers for metrics-based analysis scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from experiments.setup.common import PROJECT_ROOT, connect_database
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.setup.common import PROJECT_ROOT, connect_database


DIFFICULTY_COLUMNS = [
    "path_length_m",
    "mean_speed_m_s",
    "p95_turn_rate_rad_s",
    "p95_acceleration_m_s2",
    "p95_jerk_m_s3",
    "vertical_range_m",
    "difficulty_score",
]


def load_run_metrics(columns: list[str]) -> pd.DataFrame:
    """Load selected run metrics with post-failure difficulty values."""
    run_columns = ", ".join(f"m.{column}" for column in columns)
    difficulty_columns = ", ".join(
        f"d.{column}" for column in DIFFICULTY_COLUMNS
    )
    with connect_database() as connection:
        metrics = connection.execute(
            f"""
            WITH difficulty AS (
                SELECT
                    path_id,
                    fail_time,
                    {", ".join(DIFFICULTY_COLUMNS)}
                FROM analysis.trajectory_difficulty
                WHERE window_name = 'post_fail_3s'
            )
            SELECT
                {run_columns},
                {difficulty_columns}
            FROM analysis.failed_runs_metrics AS m
            LEFT JOIN difficulty AS d
                ON d.path_id = m.path_id
                AND d.fail_time = m.fail_time
            ORDER BY m.run_id
            """
        ).df()
    if metrics.empty:
        raise ValueError(
            "analysis.failed_runs_metrics contains no rows. "
            "Run metrics_calculate.py first."
        )
    return add_difficulty_band(metrics)


def load_drone_metrics(columns: list[str]) -> pd.DataFrame:
    """Load selected per-drone metrics from the analysis schema."""
    with connect_database() as connection:
        return connection.execute(
            f"""
            SELECT {", ".join(columns)}
            FROM analysis.fail_runs_drone_metrics
            ORDER BY run_id
            """
        ).df()


def add_difficulty_band(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add easy-to-hard quartile labels to trajectory difficulty values."""
    result = metrics.copy()
    result["difficulty_band"] = "unknown"
    valid = result["difficulty_score"].notna()
    if valid.any():
        values = result.loc[valid, "difficulty_score"]
        quantile_count = min(4, len(values))
        if values.nunique() == 1:
            result.loc[valid, "difficulty_band"] = "same"
        else:
            labels = ["easy", "medium", "hard", "very_hard"][:quantile_count]
            ranked = values.rank(method="first")
            result.loc[valid, "difficulty_band"] = pd.qcut(
                ranked,
                q=quantile_count,
                labels=labels,
            ).astype(str)
    return result


def summarize_by_conditions(
    metrics: pd.DataFrame,
    aggregations: dict[str, tuple[str, Any]],
) -> pd.DataFrame:
    """Summarize metrics by method and each supported scenario condition."""
    conditions = [
        "shape",
        "plane",
        "spread",
        "number_of_drone",
        "difficulty_band",
    ]
    summaries = []
    for condition in conditions:
        summary = (
            metrics.groupby(
                ["election_type", condition],
                as_index=False,
                dropna=False,
            )
            .agg(**aggregations)
            .rename(columns={condition: "condition_value"})
        )
        summary.insert(0, "condition", condition)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)
