"""Check whether a recorded run stays close to its intended trajectory."""

from pathlib import Path
import sys
from typing import Any

import duckdb


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DATA = _PROJECT_ROOT / "results" / "data"
_RUN_DIRECTORY = _RESULTS_DATA / "nonfail"
_TRAJECTORY_DIRECTORY = _RESULTS_DATA / "trajectories"

MAX_POSITION_DEVIATION = 2.0
MAX_DEVIATION_DURATION = 1.5
_POSITION_COLUMNS = ["timestamp", "drone", "x", "y", "z"]


def _resolve_source(filename: str | Path, directory: Path) -> Path:
    """Resolve a named Parquet or CSV input inside its registry directory."""
    source = directory / Path(filename).name
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError(f"Unsupported file type: {source.suffix}")
    return source


def _reader(source: Path) -> str:
    """Return the DuckDB reader expression for a registry file."""
    if source.suffix.lower() == ".parquet":
        return "read_parquet(?)"
    return "read_csv_auto(?)"


def _validate_position_columns(
    connection: duckdb.DuckDBPyConnection,
    reader: str,
    source: Path,
    label: str,
) -> None:
    """Ensure an input contains the position columns required for comparison."""
    columns = [
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM {reader}", [str(source)]
        ).fetchall()
    ]
    missing = [column for column in _POSITION_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _infeasible_result(
    run_filename: str | Path,
    trajectory_filename: str | Path,
    reason: str,
    drone: object | None = None,
    infeasible_since: float | None = None,
) -> dict[str, Any]:
    """Build the established infeasibility result shape."""
    result: dict[str, Any] = {
        "result": "infeasible",
        "run_file": Path(run_filename).name,
        "trajectory_file": Path(trajectory_filename).name,
        "reason": reason,
    }
    if drone is not None:
        result["drone"] = drone
    if infeasible_since is not None:
        result["infeasible_since"] = infeasible_since
    return result


def check_feasibility(
    run_filename: str | Path,
    trajectory_filename: str | Path,
) -> dict[str, Any]:
    """Compare actual run positions with intended trajectory positions.

    A run is infeasible when any drone remains more than
    ``MAX_POSITION_DEVIATION`` metres from its intended position for longer
    than ``MAX_DEVIATION_DURATION`` seconds continuously.
    """
    run_source = _resolve_source(run_filename, _RUN_DIRECTORY)
    trajectory_source = _resolve_source(trajectory_filename, _TRAJECTORY_DIRECTORY)
    run_reader = _reader(run_source)
    trajectory_reader = _reader(trajectory_source)

    with duckdb.connect() as connection:
        _validate_position_columns(connection, run_reader, run_source, "run data")
        _validate_position_columns(
            connection,
            trajectory_reader,
            trajectory_source,
            "trajectory data",
        )

        run_keys_sql = f"""
            SELECT DISTINCT
                ROUND(CAST(timestamp AS DOUBLE), 9) AS timestamp,
                CAST(drone AS VARCHAR) AS drone_key
            FROM {run_reader}
        """
        trajectory_keys_sql = f"""
            SELECT DISTINCT
                ROUND(CAST(timestamp AS DOUBLE), 9) AS timestamp,
                CAST(drone AS VARCHAR) AS drone_key
            FROM {trajectory_reader}
        """
        mismatch_row = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT run_keys.timestamp, run_keys.drone_key
                FROM ({run_keys_sql}) AS run_keys
                FULL OUTER JOIN ({trajectory_keys_sql}) AS trajectory_keys
                    ON run_keys.timestamp = trajectory_keys.timestamp
                   AND run_keys.drone_key = trajectory_keys.drone_key
                WHERE run_keys.timestamp IS NULL
                   OR trajectory_keys.timestamp IS NULL
            )
            """,
            [
                str(run_source),
                str(trajectory_source),
            ],
        ).fetchone()
        mismatch = mismatch_row[0] if mismatch_row is not None else 0
        if mismatch:
            return _infeasible_result(
                run_filename,
                trajectory_filename,
                "Run and trajectory timestamps or drones do not match.",
            )

        deviation = connection.execute(
            f"""
            WITH run_data AS (
                SELECT
                    ROUND(CAST(timestamp AS DOUBLE), 9) AS timestamp,
                    CAST(drone AS VARCHAR) AS drone_key,
                    drone,
                    CAST(x AS DOUBLE) AS actual_x,
                    CAST(y AS DOUBLE) AS actual_y,
                    CAST(z AS DOUBLE) AS actual_z
                FROM {run_reader}
            ),
            trajectory_data AS (
                SELECT
                    ROUND(CAST(timestamp AS DOUBLE), 9) AS timestamp,
                    CAST(drone AS VARCHAR) AS drone_key,
                    CAST(x AS DOUBLE) AS intended_x,
                    CAST(y AS DOUBLE) AS intended_y,
                    CAST(z AS DOUBLE) AS intended_z
                FROM {trajectory_reader}
            ),
            deviations AS (
                SELECT
                    run_data.drone,
                    run_data.drone_key,
                    run_data.timestamp,
                    SQRT(
                        POW(actual_x - intended_x, 2)
                        + POW(actual_y - intended_y, 2)
                        + POW(actual_z - intended_z, 2)
                    ) AS deviation
                FROM run_data
                INNER JOIN trajectory_data
                    ON run_data.timestamp = trajectory_data.timestamp
                   AND run_data.drone_key = trajectory_data.drone_key
            ),
            exceeded AS (
                SELECT
                    *,
                    NOT isfinite(deviation)
                        OR deviation > {MAX_POSITION_DEVIATION}
                        AS exceeds_limit
                FROM deviations
            ),
            marked AS (
                SELECT
                    *,
                    SUM(
                        CASE WHEN exceeds_limit THEN 0 ELSE 1 END
                    ) OVER (
                        PARTITION BY drone_key
                        ORDER BY timestamp
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS segment
                FROM exceeded
            )
            SELECT
                drone,
                MIN(timestamp) AS start_time,
                MAX(timestamp) - MIN(timestamp) AS duration
            FROM marked
            WHERE exceeds_limit
            GROUP BY drone_key, drone, segment
            HAVING MAX(timestamp) - MIN(timestamp) >= {MAX_DEVIATION_DURATION}
            ORDER BY start_time
            LIMIT 1
            """,
            [
                str(run_source),
                str(trajectory_source),
            ],
        ).fetchone()

    if deviation is not None:
        return _infeasible_result(
            run_filename,
            trajectory_filename,
            "A drone exceeded the deviation limit for too long.",
            deviation[0],
            deviation[1],
        )

    return {
        "result": "feasible",
        "run_file": Path(run_filename).name,
        "trajectory_file": Path(trajectory_filename).name,
    }


def main() -> None:
    """Run the checker from the command line."""
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python check_feasibility.py RUN_FILENAME TRAJECTORY_FILENAME"
        )
    result = check_feasibility(sys.argv[1], sys.argv[2])
    print(result)


if __name__ == "__main__":
    main()
