"""Create and append experiment run-result records."""

from datetime import datetime
from pathlib import Path

try:
    from experiments.setup.common import DATABASE_PATH, connect_database
except ModuleNotFoundError:
    from common import DATABASE_PATH, connect_database


RESULT_COLUMNS = [
    "run_id",
    "attempt",
    "outcome",
    "error",
    "started_at",
    "finished_at",
    "duration",
    "data_file",
    "n_steps",
    "promoted_leader",
    "promotion_time",
]
RESULT_OUTCOMES = ("done", "failed", "infeasible")
DEFAULT_DATABASE = DATABASE_PATH


def make_runs_result_table(
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Create or replace the empty runs-result registry table."""
    with connect_database() as connection:
        connection.execute("DROP TABLE IF EXISTS registry.runs_result")
        connection.execute(
            """
            CREATE TABLE registry.runs_result (
                run_id BIGINT, attempt BIGINT, outcome VARCHAR, error VARCHAR,
                started_at VARCHAR, finished_at VARCHAR, duration DOUBLE,
                data_file VARCHAR, n_steps BIGINT, promoted_leader VARCHAR,
                promotion_time DOUBLE
            )
            """
        )
    return Path(output_path)


def _next_attempt(run_id: int) -> int:
    """Return the next attempt number for a run using DuckDB."""
    with connect_database() as connection:
        result = connection.execute(
            """
            SELECT COALESCE(MAX(CAST(attempt AS BIGINT)), 0) + 1
            FROM registry.runs_result
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
    return int(result[0])


def data_input(
    run_id: int,
    outcome: str,
    started_at: datetime | str,
    finished_at: datetime | str,
    duration: float,
    n_steps: int,
    data_file: str | Path | None = None,
    promoted_leader: int | str | None = None,
    promotion_time: float | None = None,
    error: str | None = None,
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Append one result row and calculate its attempt number.

    ``started_at`` and ``finished_at`` may be datetime objects or strings.
    ``data_file`` is populated later by the Parquet merger.
    """
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise ValueError("run_id must be a positive integer.")
    if outcome not in RESULT_OUTCOMES:
        raise ValueError('outcome must be either "done" or "failed".')
    if outcome == "failed" and not error:
        raise ValueError("error is required when outcome is failed.")
    if duration < 0:
        raise ValueError("duration must be non-negative.")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")

    attempt = _next_attempt(run_id)
    started_value = (
        started_at.isoformat() if isinstance(started_at, datetime) else str(started_at)
    )
    finished_value = (
        finished_at.isoformat()
        if isinstance(finished_at, datetime)
        else str(finished_at)
    )
    with connect_database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS registry.runs_result (
                run_id BIGINT, attempt BIGINT, outcome VARCHAR, error VARCHAR,
                started_at VARCHAR, finished_at VARCHAR, duration DOUBLE,
                data_file VARCHAR, n_steps BIGINT, promoted_leader VARCHAR,
                promotion_time DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO registry.runs_result
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, attempt, outcome, error if outcome == "failed" else "",
                started_value, finished_value, duration, data_file, n_steps,
                promoted_leader, promotion_time,
            ],
        )
    return Path(output_path)


if __name__ == "__main__":
    make_runs_result_table()


make_runs_result_csv = make_runs_result_table
