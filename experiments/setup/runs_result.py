"""Create and append experiment run-result records."""

from datetime import datetime
from pathlib import Path
import csv

import duckdb

try:
    from experiments.setup.common import REGISTRY_DIRECTORY
except ModuleNotFoundError:
    from common import REGISTRY_DIRECTORY


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
DEFAULT_RESULTS_FILE = REGISTRY_DIRECTORY / "runs_result.csv"


def make_runs_result_csv(
    output_path: str | Path = DEFAULT_RESULTS_FILE,
) -> Path:
    """Create an empty runs-result CSV with the required header."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(RESULT_COLUMNS)
    return destination


def _next_attempt(run_id: int, output_path: Path) -> int:
    """Return the next attempt number for a run using DuckDB."""
    with duckdb.connect() as connection:
        result = connection.execute(
            """
            SELECT COALESCE(MAX(CAST(attempt AS BIGINT)), 0) + 1
            FROM read_csv_auto(?, HEADER = TRUE)
            WHERE run_id = ?
            """,
            [str(output_path), run_id],
        ).fetchone()
    return int(result[0])


def data_input(
    run_id: int,
    outcome: str,
    started_at: datetime | str,
    finished_at: datetime | str,
    duration: float,
    data_file: str | Path,
    n_steps: int,
    promoted_leader: int | str | None = None,
    promotion_time: float | None = None,
    error: str | None = None,
    output_path: str | Path = DEFAULT_RESULTS_FILE,
) -> Path:
    """Append one result row and calculate its attempt number.

    ``started_at`` and ``finished_at`` may be datetime objects or strings.
    Error text is written as a quoted CSV field when it contains special
    characters. Successful rows always store an empty error field.
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

    destination = Path(output_path)
    if not destination.exists() or destination.stat().st_size == 0:
        make_runs_result_csv(destination)

    attempt = _next_attempt(run_id, destination)
    started_value = (
        started_at.isoformat() if isinstance(started_at, datetime) else str(started_at)
    )
    finished_value = (
        finished_at.isoformat()
        if isinstance(finished_at, datetime)
        else str(finished_at)
    )
    row = {
        "run_id": run_id,
        "attempt": attempt,
        "outcome": outcome,
        "error": error if outcome == "failed" else "",
        "started_at": started_value,
        "finished_at": finished_value,
        "duration": duration,
        "data_file": str(data_file),
        "n_steps": n_steps,
        "promoted_leader": "" if promoted_leader is None else promoted_leader,
        "promotion_time": "" if promotion_time is None else promotion_time,
    }
    with destination.open("a", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS).writerow(row)
    return destination


if __name__ == "__main__":
    make_runs_result_csv()
