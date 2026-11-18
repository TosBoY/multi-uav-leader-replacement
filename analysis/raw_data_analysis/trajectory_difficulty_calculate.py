"""Calculate difficulty around leader-failure windows from non-failure paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from experiments.setup.common import PROJECT_ROOT, connect_database
    from analysis.setup.trajectory_difficulty import (
        data_input,
        make_trajectory_difficulty_table,
    )
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from experiments.setup.common import PROJECT_ROOT, connect_database
    from analysis.setup.trajectory_difficulty import (
        data_input,
        make_trajectory_difficulty_table,
    )


TRAJECTORY_DIRECTORY = PROJECT_ROOT / "results" / "data" / "trajectories"
LEADER_DRONE = 0
REQUIRED_COLUMNS = {"timestamp", "drone", "x", "y", "z"}
WINDOWS = (
    ("pre_fail_1s", -1.0, 0.0),
    ("post_fail_1s", 0.0, 1.0),
    ("post_fail_3s", 0.0, 3.0),
)
SCORE_COLUMNS = (
    "path_length_m",
    "p95_turn_rate_rad_s",
    "p95_acceleration_m_s2",
    "p95_jerk_m_s3",
    "vertical_range_m",
)


def _latest_done_results_query() -> str:
    """Return a query fragment selecting the latest successful attempts."""
    return """
        SELECT run_id
        FROM (
            SELECT
                run_id,
                outcome,
                ROW_NUMBER() OVER (
                    PARTITION BY run_id
                    ORDER BY CAST(attempt AS BIGINT) DESC
                ) AS attempt_order
            FROM registry.runs_result
        )
        WHERE attempt_order = 1 AND outcome = 'done'
    """


def _load_scenarios() -> list[dict[str, Any]]:
    """Load unique path/failure-time scenarios with a usable baseline run."""
    query = f"""
        WITH completed_runs AS ({_latest_done_results_query()}),
        completed_nonfail AS (
            SELECT
                r.runs_id AS baseline_nonfail_run_id,
                r.path_id,
                r.formation_id,
                r.init_leader
            FROM registry.runs AS r
            INNER JOIN completed_runs AS result
                ON result.run_id = r.runs_id
            WHERE NOT r.leader_failure_enabled
              AND r.status != 'infeasible'
        ),
        scenarios AS (
            SELECT
                failure.path_id,
                failure.fail_time,
                baseline.baseline_nonfail_run_id,
                ROW_NUMBER() OVER (
                    PARTITION BY failure.path_id, failure.fail_time
                    ORDER BY baseline.baseline_nonfail_run_id
                ) AS scenario_order
            FROM registry.runs AS failure
            INNER JOIN completed_nonfail AS baseline
                ON baseline.path_id = failure.path_id
                AND baseline.formation_id = failure.formation_id
                AND baseline.init_leader = failure.init_leader
            WHERE failure.leader_failure_enabled
              AND failure.fail_time IS NOT NULL
              AND failure.status != 'infeasible'
        )
        SELECT
            path_id,
            fail_time,
            baseline_nonfail_run_id
        FROM scenarios
        WHERE scenario_order = 1
        ORDER BY path_id, fail_time
    """
    with connect_database() as connection:
        result = connection.execute(query)
        columns = [description[0] for description in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def _relative_source_path(path: Path) -> str:
    """Return a stable project-relative path for a source file."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _raw_file_run_id(path: Path) -> int | None:
    """Read a run ID from an unmerged ``<run_id>_path.parquet`` filename."""
    match = re.fullmatch(r"(\d+)_path", path.stem)
    return None if match is None else int(match.group(1))


def _read_leader_path(
    run_id: int,
    trajectory_files: list[Path],
) -> tuple[pd.DataFrame, str]:
    """Read the leader path for one run from a merged or raw Parquet file."""
    matches: list[tuple[pd.DataFrame, Path]] = []
    for path in trajectory_files:
        columns = set(pq.read_schema(path).names)
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        filters: list[tuple[str, str, object]] = [
            ("drone", "==", LEADER_DRONE)
        ]
        if "run_id" in columns:
            filters.insert(0, ("run_id", "==", run_id))
        elif _raw_file_run_id(path) != run_id:
            continue

        frame = pd.read_parquet(
            path,
            columns=["run_id", "timestamp", "drone", "x", "y", "z"]
            if "run_id" in columns
            else ["timestamp", "drone", "x", "y", "z"],
            filters=filters,
        )
        if not frame.empty:
            matches.append((frame, path))

    if not matches:
        raise FileNotFoundError(
            f"No trajectory data found for non-failure run {run_id}."
        )
    if len(matches) > 1:
        source_files = ", ".join(str(path) for _, path in matches)
        raise ValueError(
            f"Run {run_id} appears in multiple trajectory files: {source_files}"
        )

    frame, source = matches[0]
    frame = frame.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"],
        keep="first",
    )
    frame = frame[["timestamp", "x", "y", "z"]].copy()
    for column in ("timestamp", "x", "y", "z"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(
            f"Run {run_id} has fewer than two valid leader trajectory samples."
        )
    return frame, _relative_source_path(source)


def _percentile_or_zero(values: np.ndarray, percentile: float = 95.0) -> float:
    """Return a percentile for finite values, or zero when no value exists."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.percentile(finite_values, percentile))


def _window_metrics(
    frame: pd.DataFrame,
    window_start: float,
    window_end: float,
) -> dict[str, float]:
    """Calculate raw difficulty components for one time window."""
    window = frame[
        (frame["timestamp"] >= window_start)
        & (frame["timestamp"] <= window_end)
    ]
    if len(window) < 2:
        raise ValueError(
            f"Window [{window_start}, {window_end}] has fewer than two samples."
        )

    times = window["timestamp"].to_numpy(dtype=float)
    positions = window[["x", "y", "z"]].to_numpy(dtype=float)
    time_deltas = np.diff(times)
    valid_deltas = time_deltas > 0
    if not valid_deltas.all():
        times = times[np.r_[True, valid_deltas]]
        positions = positions[np.r_[True, valid_deltas]]
        time_deltas = np.diff(times)
    if len(times) < 2 or not np.all(time_deltas > 0):
        raise ValueError(
            f"Window [{window_start}, {window_end}] has no increasing samples."
        )

    displacements = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(displacements, axis=1)
    path_length = float(segment_lengths.sum())
    actual_duration = float(times[-1] - times[0])
    if actual_duration <= 0:
        raise ValueError(
            f"Window [{window_start}, {window_end}] has zero duration."
        )

    velocities = displacements / time_deltas[:, None]
    speeds = np.linalg.norm(velocities, axis=1)

    unit_velocities = np.zeros_like(velocities)
    moving = speeds > 0
    unit_velocities[moving] = (
        velocities[moving] / speeds[moving, None]
    )
    turn_angles = np.zeros(max(0, len(velocities) - 1), dtype=float)
    if len(velocities) > 1:
        dot_products = np.sum(
            unit_velocities[:-1] * unit_velocities[1:],
            axis=1,
        )
        dot_products = np.clip(dot_products, -1.0, 1.0)
        valid_turns = moving[:-1] & moving[1:]
        turn_angles[valid_turns] = np.arccos(dot_products[valid_turns])
        turn_rates = turn_angles / time_deltas[1:]
    else:
        turn_rates = np.array([], dtype=float)

    accelerations = np.diff(velocities, axis=0) / time_deltas[1:, None]
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
    jerks = np.diff(accelerations, axis=0) / time_deltas[2:, None]
    jerk_magnitudes = np.linalg.norm(jerks, axis=1)

    return {
        "actual_window_duration": actual_duration,
        "path_length_m": path_length,
        "mean_speed_m_s": path_length / actual_duration,
        "p95_turn_rate_rad_s": _percentile_or_zero(turn_rates),
        "p95_acceleration_m_s2": _percentile_or_zero(acceleration_magnitudes),
        "p95_jerk_m_s3": _percentile_or_zero(jerk_magnitudes),
        "vertical_range_m": float(
            positions[:, 2].max() - positions[:, 2].min()
        ),
    }


def _add_difficulty_scores(
    rows: list[dict[str, Any]],
) -> None:
    """Add a per-window min-max difficulty score to raw metric rows."""
    for window_name in {str(row["window_name"]) for row in rows}:
        window_rows = [
            row for row in rows if row["window_name"] == window_name
        ]
        normalized_values: list[list[float]] = []
        for column in SCORE_COLUMNS:
            values = np.asarray(
                [float(row[column]) for row in window_rows],
                dtype=float,
            )
            minimum = float(values.min())
            maximum = float(values.max())
            if maximum == minimum:
                normalized = np.zeros_like(values)
            else:
                normalized = (values - minimum) / (maximum - minimum)
            normalized_values.append(normalized.tolist())

        scores = np.mean(np.asarray(normalized_values), axis=0)
        for row, score in zip(window_rows, scores):
            row["difficulty_score"] = float(score)


def calculate_trajectory_difficulty() -> int:
    """Calculate and store all configured trajectory difficulty windows."""
    scenarios = _load_scenarios()
    trajectory_files = sorted(TRAJECTORY_DIRECTORY.glob("*.parquet"))
    if not trajectory_files:
        raise FileNotFoundError(
            f"No trajectory Parquet files found in {TRAJECTORY_DIRECTORY}."
        )

    rows: list[dict[str, Any]] = []
    skipped: list[tuple[int, str]] = []
    for scenario in scenarios:
        path_id = int(scenario["path_id"])
        fail_time = float(scenario["fail_time"])
        baseline_run_id = int(scenario["baseline_nonfail_run_id"])
        try:
            frame, source_file = _read_leader_path(
                baseline_run_id,
                trajectory_files,
            )
            mission_end = float(frame["timestamp"].max())
            for window_name, start_offset, end_offset in WINDOWS:
                requested_start = fail_time + start_offset
                requested_end = fail_time + end_offset
                window_start = max(0.0, requested_start)
                window_end = min(mission_end, requested_end)
                if window_end <= window_start:
                    raise ValueError(
                        f"Window {window_name} is outside the trajectory."
                    )
                metrics = _window_metrics(
                    frame,
                    window_start,
                    window_end,
                )
                rows.append(
                    {
                        "path_id": path_id,
                        "fail_time": fail_time,
                        "window_name": window_name,
                        "window_start": window_start,
                        "window_end": window_end,
                        "source_file": source_file,
                        **metrics,
                        "difficulty_score": 0.0,
                    }
                )
        except (FileNotFoundError, ValueError) as error:
            skipped.append((path_id, str(error)))

    _add_difficulty_scores(rows)
    make_trajectory_difficulty_table()
    with connect_database() as connection:
        for row in rows:
            data_input(connection=connection, **row)

    print(f"Calculated {len(rows)} trajectory difficulty rows.")
    for path_id, message in skipped:
        print(f"Skipped path {path_id}: {message}")
    return len(rows)


if __name__ == "__main__":
    calculate_trajectory_difficulty()
