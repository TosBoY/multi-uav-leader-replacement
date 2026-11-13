"""Calculate and store metrics for completed leader-failure runs."""

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from analysis.setup.fail_runs_drone_metrics import (
        FAIL_RUNS_DRONE_METRICS_COLUMNS,
        FAIL_RUNS_DRONE_METRICS_TABLE,
        make_fail_runs_drone_metrics_table,
    )
    from analysis.setup.failed_runs_metrics import (
        FAILED_RUNS_METRICS_COLUMNS,
        FAILED_RUNS_METRICS_TABLE,
        make_failed_runs_metrics_table,
    )
    from experiments.setup.common import PROJECT_ROOT, connect_database
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from analysis.setup.fail_runs_drone_metrics import (
        FAIL_RUNS_DRONE_METRICS_COLUMNS,
        FAIL_RUNS_DRONE_METRICS_TABLE,
        make_fail_runs_drone_metrics_table,
    )
    from analysis.setup.failed_runs_metrics import (
        FAILED_RUNS_METRICS_COLUMNS,
        FAILED_RUNS_METRICS_TABLE,
        make_failed_runs_metrics_table,
    )
    from experiments.setup.common import PROJECT_ROOT, connect_database


WINDOW_S = 7.0
HOLD_TIME_S_01 = 0.5
HOLD_TIME_S_002 = 1.0
TOL_LOOSE_M = 0.1
TOL_TIGHT_M = 0.02
SAFETY_DISTANCE_M = 0.5
BATCH_RUNS = 50
POSITION_COLUMNS = ("x", "y", "z")
REQUIRED_COLUMNS = {"timestamp", "drone", *POSITION_COLUMNS}


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as days, hours, minutes, and seconds."""
    whole_seconds = int(seconds)
    days, remainder = divmod(whole_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, remaining_seconds = divmod(remainder, 60)

    if days:
        return f"{days}d {hours}h {minutes}m {remaining_seconds}s"
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{seconds:.2f}s"


def _resolve_data_file(data_file: str) -> Path:
    """Resolve a registry data path relative to the project root."""
    source = Path(data_file)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _read_run_data(data_file: str, run_id: int) -> pd.DataFrame:
    """Read one run from a raw or merged Parquet file."""
    path = _resolve_data_file(data_file)
    columns = set(pq.read_schema(path).names)
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"{data_file} is missing columns: {sorted(missing)}")
    if "run_id" in columns:
        frame = pd.read_parquet(path, filters=[("run_id", "==", run_id)])
    else:
        frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"No rows for run {run_id} in {data_file}.")
    frame = frame.copy()
    frame["timestamp"] = frame["timestamp"].astype(float).round(9)
    frame["drone"] = frame["drone"].astype(int)
    return frame.sort_values(["timestamp", "drone"]).reset_index(drop=True)


@lru_cache(maxsize=128)
def _read_baseline_data(data_file: str, run_id: int) -> pd.DataFrame:
    """Read and cache one baseline run from Parquet."""
    return _read_run_data(data_file, run_id)


def _position_before(
    frame: pd.DataFrame,
    drone: int,
    timestamp: float,
) -> np.ndarray:
    """Return a drone's latest position strictly before a timestamp."""
    rows = frame[
        (frame["drone"] == drone)
        & (frame["timestamp"] < timestamp)
    ]
    if rows.empty:
        raise ValueError(
            f"Drone {drone} has no position data before t={timestamp}."
        )
    row = rows.iloc[-1]
    return row.loc[list(POSITION_COLUMNS)].to_numpy(dtype=float)


def _actual_change_time(frame: pd.DataFrame, old_leader: int) -> float:
    """Return the first timestamp where the failed leader has no position."""
    leader_rows = frame[frame["drone"] == old_leader].sort_values("timestamp")
    missing_position = leader_rows[list(POSITION_COLUMNS)].isna().any(axis=1)
    if not missing_position.any():
        raise ValueError(
            f"Leader drone {old_leader} never has a missing position in the "
            "failure data."
        )
    return float(leader_rows.loc[missing_position, "timestamp"].iloc[0])


def _error_series(
    failure: pd.DataFrame,
    baseline: pd.DataFrame,
    drone: int,
    reference_drone: int,
    change_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare a failed-run drone with its baseline reference trajectory."""
    failure_rows = failure[failure["drone"] == drone][
        ["timestamp", *POSITION_COLUMNS]
    ]
    baseline_rows = baseline[baseline["drone"] == reference_drone][
        ["timestamp", *POSITION_COLUMNS]
    ]
    merged = failure_rows.merge(
        baseline_rows,
        on="timestamp",
        suffixes=("_failure", "_baseline"),
    )
    merged = merged[
        (merged["timestamp"] >= change_time)
        & (merged["timestamp"] <= change_time + WINDOW_S + 1e-9)
    ].sort_values("timestamp")
    if merged.empty:
        raise ValueError(
            f"No matching timestamps for drone {drone} and reference "
            f"{reference_drone}."
        )
    if merged[
        [f"{column}_failure" for column in POSITION_COLUMNS]
        + [f"{column}_baseline" for column in POSITION_COLUMNS]
    ].isna().any().any():
        raise ValueError(
            f"NaN position data for drone {drone} or reference "
            f"{reference_drone} inside the analysis window."
        )
    failure_positions = merged[
        [f"{column}_failure" for column in POSITION_COLUMNS]
    ].to_numpy(dtype=float)
    baseline_positions = merged[
        [f"{column}_baseline" for column in POSITION_COLUMNS]
    ].to_numpy(dtype=float)
    return (
        merged["timestamp"].to_numpy(dtype=float),
        np.linalg.norm(failure_positions - baseline_positions, axis=1),
    )


def _first_match(
    within: np.ndarray,
    hold_steps: int,
) -> int | None:
    """Return the first index with the required consecutive match samples."""
    if len(within) < hold_steps:
        return None
    sums = np.convolve(
        within.astype(int),
        np.ones(hold_steps, dtype=int),
        mode="valid",
    )
    matches = np.flatnonzero(sums == hold_steps)
    return int(matches[0]) if len(matches) else None


def _match_metrics(
    times: np.ndarray,
    errors: np.ndarray,
    change_time: float,
    tolerance: float,
    hold_time: float,
) -> tuple[str, float | None, int | None]:
    """Calculate match status, time, and post-match excursions."""
    finite = np.isfinite(errors)
    if not finite.any():
        return "failed", None, None
    within = finite & (errors <= tolerance)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else hold_time
    hold_steps = max(1, int(round(hold_time / dt)))
    match_index = _first_match(within, hold_steps)
    if match_index is None:
        return "never_matched", None, None
    excursions = int(np.sum(within[match_index:-1] & ~within[match_index + 1:]))
    status = "matched" if excursions == 0 else "unstable"
    return status, float(times[match_index] - change_time), excursions


def _integral(times: np.ndarray, errors: np.ndarray) -> float:
    """Calculate trapezoidal error integral."""
    if len(times) < 2:
        return 0.0
    return float(np.sum(0.5 * (errors[1:] + errors[:-1]) * np.diff(times)))


def _distance_metrics(
    frame: pd.DataFrame,
    drones: Iterable[int],
    change_time: float,
) -> tuple[float | None, float | None, int]:
    """Calculate minimum pairwise separation and close-step count."""
    drone_ids = list(drones)
    if len(drone_ids) < 2:
        return None, None, 0
    selected = frame[
        frame["drone"].isin(drone_ids)
        & (frame["timestamp"] >= change_time)
        & (frame["timestamp"] <= change_time + WINDOW_S + 1e-9)
    ]
    if selected.empty:
        return None, None, 0
    indexed = selected.set_index(["timestamp", "drone"])[
        list(POSITION_COLUMNS)
    ].unstack("drone")
    timestamps = indexed.index.to_numpy(dtype=float)
    positions = np.stack(
        [
            indexed[column]
            .reindex(columns=drone_ids)
            .to_numpy(dtype=float)
            for column in POSITION_COLUMNS
        ],
        axis=-1,
    )
    valid = ~np.isnan(positions).any(axis=(1, 2))
    if not valid.any():
        return None, None, 0
    positions = positions[valid]
    timestamps = timestamps[valid]
    differences = positions[:, :, None, :] - positions[:, None, :, :]
    pairwise = np.linalg.norm(differences, axis=-1)
    indices = np.arange(len(drone_ids))
    pairwise[:, indices, indices] = np.inf
    separations = pairwise.min(axis=(1, 2))
    minimum_index = int(np.argmin(separations))
    return (
        float(separations[minimum_index]),
        float(timestamps[minimum_index] - change_time),
        int(np.count_nonzero(separations < SAFETY_DISTANCE_M)),
    )


def _load_runs() -> list[dict[str, Any]]:
    """Load the latest result for each completed failed run."""
    with connect_database() as connection:
        result = connection.execute(
            """
            WITH ranked_results AS (
                SELECT
                    rr.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY rr.run_id
                        ORDER BY CAST(rr.attempt AS BIGINT) DESC
                    ) AS result_rank
                FROM registry.runs_result AS rr
            )
            SELECT
                r.runs_id AS run_id,
                r.path_id,
                r.formation_id,
                r.leader_failure_enabled,
                r.fail_time,
                r.init_leader,
                r.election_type,
                rr.outcome,
                rr.data_file,
                rr.promoted_leader,
                f.number_of_drone,
                f.shape,
                f.spread,
                f.plane
            FROM ranked_results AS rr
            INNER JOIN registry.runs AS r ON r.runs_id = rr.run_id
            INNER JOIN registry.formations AS f
                ON f.formation_id = r.formation_id
            WHERE rr.result_rank = 1
              AND r.leader_failure_enabled
            ORDER BY r.runs_id
            """
        )
        rows = result.fetchall()
        columns = [description[0] for description in result.description]
    return [dict(zip(columns, row)) for row in rows]


@lru_cache(maxsize=128)
def _baseline_for_cached(
    path_id: int,
    formation_id: int,
    init_leader: int,
) -> dict[str, Any]:
    """Find the non-failure run paired with a failed run."""
    with connect_database() as connection:
        candidates = connection.execute(
            """
            SELECT DISTINCT r.runs_id, r.init_leader
            FROM registry.runs AS r
            WHERE r.path_id = ?
              AND r.formation_id = ?
              AND r.init_leader = ?
              AND NOT r.leader_failure_enabled
            """,
            [path_id, formation_id, init_leader],
        ).fetchall()
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one baseline run for path {path_id}, "
                f"formation {formation_id}, and initial leader "
                f"{init_leader}; found {len(candidates)}."
            )
        baseline_run_id, baseline_leader = candidates[0]
        row = connection.execute(
            """
            SELECT data_file
            FROM registry.runs_result
            WHERE run_id = ? AND outcome = 'done' AND data_file IS NOT NULL
            ORDER BY CAST(attempt AS BIGINT) DESC
            LIMIT 1
            """,
            [baseline_run_id],
        ).fetchone()
    if row is None:
        raise ValueError(
            f"No completed baseline for path {path_id} and "
            f"formation {formation_id}."
        )
    return {
        "run_id": int(baseline_run_id),
        "init_leader": int(baseline_leader),
        "data_file": str(row[0]),
    }


def _baseline_for(run: dict[str, Any]) -> dict[str, Any]:
    """Find the cached non-failure run paired with a failed run."""
    return _baseline_for_cached(
        int(run["path_id"]),
        int(run["formation_id"]),
        int(run["init_leader"]),
    )


def _drone_row(
    run: dict[str, Any],
    baseline: dict[str, Any],
    failure: pd.DataFrame,
    baseline_data: pd.DataFrame,
    drone: int,
    promoted_leader: int,
    old_leader: int,
    survivors: list[int],
    change_time: float,
    positions_before_change: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Calculate one survivor's metrics."""
    role = "new leader" if drone == promoted_leader else "follower"
    reference = old_leader if role == "new leader" else drone
    times, errors = _error_series(
        failure, baseline_data, drone, reference, change_time
    )
    peak_index = int(np.nanargmax(errors))
    fail_position = positions_before_change[drone]
    old_leader_position = positions_before_change[old_leader]
    distance = float(np.linalg.norm(fail_position - old_leader_position))
    rank = 1 + sum(
        (
            float(
                np.linalg.norm(
                    positions_before_change[other] - old_leader_position
                )
            ),
            other,
        )
        < (distance, drone)
        for other in survivors
        if other != drone
    )
    status_01, time_01, excursions_01 = _match_metrics(
        times, errors, change_time, TOL_LOOSE_M, HOLD_TIME_S_01
    )
    status_002, time_002, excursions_002 = _match_metrics(
        times, errors, change_time, TOL_TIGHT_M, HOLD_TIME_S_002
    )
    return {
        "run_id": run["run_id"],
        "baseline_nonfail_run_id": baseline["run_id"],
        "drone": drone,
        "_window_sample_count": len(times),
        "role": role,
        "reference_drone": reference,
        "distance_to_old_leader_at_change": distance,
        "distance_rank_at_change": rank,
        "mean_error": float(np.mean(errors)),
        "rms_error": float(np.sqrt(np.mean(errors**2))),
        "peak_error": float(errors[peak_index]),
        "peak_error_time": float(times[peak_index] - change_time),
        "error_integral": _integral(times, errors),
        "final_error": float(errors[-1]),
        "status_01": status_01,
        "time_to_match_01": time_01,
        "excursions_01": excursions_01,
        "status_002": status_002,
        "time_to_match_002": time_002,
        "excursions_002": excursions_002,
    }


def _aggregate_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-drone metrics into one run-level row."""
    peak = max(rows, key=lambda row: (row["peak_error"], -row["drone"]))
    result: dict[str, Any] = {
        "run_id": rows[0]["run_id"],
        "baseline_nonfail_run_id": rows[0]["baseline_nonfail_run_id"],
        "number_of_drone": len(rows) + 1,
        "mean_error": float(np.mean([row["mean_error"] for row in rows])),
        "peak_error": peak["peak_error"],
        "peak_error_time": peak["peak_error_time"],
        "error_integral_sum": float(sum(row["error_integral"] for row in rows)),
        "final_error_max": max(row["final_error"] for row in rows),
    }
    for suffix in ("01", "002"):
        statuses = [row[f"status_{suffix}"] for row in rows]
        times = [
            row[f"time_to_match_{suffix}"]
            for row in rows
            if row[f"time_to_match_{suffix}"] is not None
        ]
        result[f"n_matched_{suffix}"] = statuses.count("matched")
        result[f"n_unstable_{suffix}"] = statuses.count("unstable")
        result[f"n_never_matched_{suffix}"] = statuses.count("never_matched")
        result[f"time_to_match_mean_{suffix}"] = (
            float(np.mean(times)) if times else None
        )
        result[f"time_to_match_median_{suffix}"] = (
            float(np.median(times)) if times else None
        )
        result[f"time_to_match_max_{suffix}"] = max(times) if times else None
        result[f"settle_time_{suffix}"] = (
            max(times) if len(times) == len(rows) else None
        )
    return result


def _insert_batch(
    drone_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
) -> None:
    """Insert one batch of drone and run rows in a single transaction."""
    if not drone_rows and not run_rows:
        return
    drone_frame = pd.DataFrame(drone_rows).loc[
        :, FAIL_RUNS_DRONE_METRICS_COLUMNS
    ]
    run_frame = pd.DataFrame(run_rows).loc[:, FAILED_RUNS_METRICS_COLUMNS]
    with connect_database() as connection:
        connection.register("_drone_metrics_batch", drone_frame)
        connection.register("_run_metrics_batch", run_frame)
        try:
            connection.execute("BEGIN TRANSACTION")
            drone_columns = ", ".join(FAIL_RUNS_DRONE_METRICS_COLUMNS)
            run_columns = ", ".join(FAILED_RUNS_METRICS_COLUMNS)
            connection.execute(
                f"""
                INSERT INTO {FAIL_RUNS_DRONE_METRICS_TABLE} ({drone_columns})
                SELECT {drone_columns} FROM _drone_metrics_batch
                """
            )
            connection.execute(
                f"""
                INSERT INTO {FAILED_RUNS_METRICS_TABLE} ({run_columns})
                SELECT {run_columns} FROM _run_metrics_batch
                """
            )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            connection.unregister("_drone_metrics_batch")
            connection.unregister("_run_metrics_batch")


def _validate_drone_rows(rows: list[dict[str, Any]]) -> None:
    """Validate aggregation assumptions before writing any rows."""
    for suffix in ("01", "002"):
        statuses = [row[f"status_{suffix}"] for row in rows]
        if "failed" in statuses:
            raise ValueError(
                f"Run {rows[0]['run_id']} contains a drone with no finite "
                f"error data for tolerance {suffix}."
            )
        if sum(
            statuses.count(status)
            for status in ("matched", "unstable", "never_matched")
        ) != len(rows):
            raise ValueError(f"Unrecognized status in tolerance {suffix}.")
    for row in rows:
        tight_status = row["status_002"]
        loose_status = row["status_01"]
        if (
            tight_status in {"matched", "unstable"}
            and loose_status == "never_matched"
        ):
            raise ValueError(
                f"Run {row['run_id']} drone {row['drone']} has a tight "
                "match without a loose match."
            )
        loose_time = row["time_to_match_01"]
        tight_time = row["time_to_match_002"]
        if (
            loose_time is not None
            and tight_time is not None
            and tight_time < loose_time
        ):
            raise ValueError(
                f"Run {row['run_id']} drone {row['drone']} matched the "
                "tight tolerance before the loose tolerance."
            )


def calculate_metrics() -> int:
    """Recreate both analysis tables and calculate all available metrics."""
    start_time = perf_counter()
    make_fail_runs_drone_metrics_table()
    make_failed_runs_metrics_table()
    processed = 0
    pending_drone_rows: list[dict[str, Any]] = []
    pending_run_rows: list[dict[str, Any]] = []
    baseline_separation_cache: dict[
        tuple[int, float], tuple[float | None, float | None, int]
    ] = {}
    skipped: list[tuple[int, str]] = []
    not_done = 0
    flush_error: Exception | None = None

    def flush_pending() -> int:
        """Write pending rows and clear them only after a successful commit."""
        if not pending_run_rows:
            return 0
        rows_to_write = len(pending_run_rows)
        _insert_batch(pending_drone_rows, pending_run_rows)
        pending_drone_rows.clear()
        pending_run_rows.clear()
        return rows_to_write

    for raw_run in _load_runs():
        run_id = int(raw_run["run_id"])
        try:
            if raw_run["outcome"] != "done" or not raw_run["data_file"]:
                not_done += 1
                continue
            run = dict(raw_run)
            run["fail_time"] = float(run["fail_time"])
            run["number_of_drone"] = int(run["number_of_drone"])
            baseline = _baseline_for(run)
            failure = _read_run_data(str(run["data_file"]), run_id)
            baseline_data = _read_baseline_data(
                baseline["data_file"], int(baseline["run_id"])
            )
            old_leader = int(baseline["init_leader"])
            change_time = _actual_change_time(failure, old_leader)
            last_time = min(
                float(failure["timestamp"].max()),
                float(baseline_data["timestamp"].max()),
            )
            if last_time < change_time + WINDOW_S - 1e-9:
                raise ValueError(
                    f"data ends at t={last_time:g}, before the required "
                    f"window end t={change_time + WINDOW_S:g}."
                )
            if raw_run["promoted_leader"] is None:
                raise ValueError("the result has no promoted leader.")
            promoted_leader = int(raw_run["promoted_leader"])
            survivors = sorted(
                int(drone)
                for drone in failure["drone"].unique()
                if int(drone) != old_leader
            )
            if len(survivors) + 1 != run["number_of_drone"]:
                raise ValueError(
                    f"expected {run['number_of_drone'] - 1} survivors, "
                    f"found {len(survivors)}."
                )
            if promoted_leader not in survivors:
                raise ValueError(
                    f"promoted leader {promoted_leader} is not among "
                    f"surviving drones {survivors}."
                )
            positions_before_change = {
                drone: _position_before(failure, drone, change_time)
                for drone in [old_leader, *survivors]
            }
            drone_rows = [
                _drone_row(
                    run,
                    baseline,
                    failure,
                    baseline_data,
                    drone,
                    promoted_leader,
                    old_leader,
                    survivors,
                    change_time,
                    positions_before_change,
                )
                for drone in survivors
            ]
            sample_counts = {
                int(row["_window_sample_count"]) for row in drone_rows
            }
            if len(sample_counts) != 1:
                raise ValueError(
                    f"survivors have inconsistent window sample counts: "
                    f"{sorted(sample_counts)}."
                )
            _validate_drone_rows(drone_rows)
            aggregate = _aggregate_run(drone_rows)
            fail_separation = _distance_metrics(
                failure, survivors, change_time
            )
            separation_key = (baseline["run_id"], change_time)
            if separation_key not in baseline_separation_cache:
                baseline_separation_cache[separation_key] = _distance_metrics(
                    baseline_data, survivors, change_time
                )
            baseline_separation = baseline_separation_cache[separation_key]
            aggregate.update(
                {
                    "number_of_drone": run["number_of_drone"],
                    "shape": run["shape"],
                    "spread": run["spread"],
                    "plane": run["plane"],
                    "election_type": run["election_type"],
                    "fail_time": run["fail_time"],
                    "promoted_leader": promoted_leader,
                    "min_separation": fail_separation[0],
                    "min_separation_time": fail_separation[1],
                    "baseline_min_separation": baseline_separation[0],
                    "n_close_steps": fail_separation[2],
                }
            )
            for row in drone_rows:
                row.pop("_window_sample_count")
            pending_drone_rows.extend(drone_rows)
            pending_run_rows.append(aggregate)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            skipped.append((run_id, message))
            print(f"Skipped run {run_id}: {message}")
        if len(pending_run_rows) >= BATCH_RUNS:
            try:
                processed += flush_pending()
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                print(f"Failed to flush metrics batch: {message}")
                flush_error = error
                break
    if flush_error is None:
        try:
            processed += flush_pending()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"Failed to flush final metrics batch: {message}")
            flush_error = error
    if skipped:
        print(f"Skipped {len(skipped)} runs:")
        for run_id, message in skipped:
            print(f"  {run_id}: {message}")
    if not_done:
        print(f"Skipped {not_done} runs that were not completed.")
    elapsed_time = perf_counter() - start_time
    print(f"Finished metrics calculation in {_format_elapsed_time(elapsed_time)}.")
    if flush_error is not None:
        raise flush_error
    return processed


if __name__ == "__main__":
    print(f"Processed {calculate_metrics()} failed runs.")
