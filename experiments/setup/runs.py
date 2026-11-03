"""Generate and edit the experiment runs setup table."""

from collections.abc import Iterator
from pathlib import Path
import tempfile

import duckdb

try:
    from experiments.setup.common import (
        REGISTRY_DIRECTORY,
        setup_path,
        validate_choice,
    )
except ModuleNotFoundError:
    from common import (
        REGISTRY_DIRECTORY,
        setup_path,
        validate_choice,
    )


RUN_COLUMNS = [
    "runs_id",
    "path_id",
    "formation_id",
    "leader_failure_enabled",
    "fail_time",
    "init_leader",
    "election_type",
    "status",
]
ELECTION_TYPES = ("closest", "minimum_travel")
RUN_STATUSES = ("planned", "running", "done", "failed", "infeasible", "cancelled")
FAIL_TIMES = tuple(2.5 + 0.5 * index for index in range(11))
DEFAULT_PATHS_FILE = setup_path("paths.csv")
DEFAULT_FORMATIONS_FILE = setup_path("formations.csv")
DEFAULT_RUNS_FILE = REGISTRY_DIRECTORY / "runs.csv"


_RUN_SELECT = """
WITH path_ids AS (
    SELECT DISTINCT CAST(path_id AS BIGINT) AS path_id
    FROM read_csv_auto(?)
),
formation_ids AS (
    SELECT DISTINCT CAST(formation_id AS BIGINT) AS formation_id
    FROM read_csv_auto(?)
),
combinations AS (
    SELECT path_id, formation_id
    FROM path_ids CROSS JOIN formation_ids
),
run_rows AS (
    SELECT
        path_id,
        formation_id,
        FALSE AS leader_failure_enabled,
        CAST(NULL AS DOUBLE) AS fail_time,
        0 AS init_leader,
        'closest' AS election_type,
        'planned' AS status,
        0 AS row_order
    FROM combinations
    UNION ALL
    SELECT
        combinations.path_id,
        combinations.formation_id,
        TRUE AS leader_failure_enabled,
        fail_times.fail_time,
        0 AS init_leader,
        election_types.election_type,
        'planned' AS status,
        1 AS row_order
    FROM combinations
    CROSS JOIN (SELECT range / 2.0 AS fail_time FROM range(5, 16)) AS fail_times
    CROSS JOIN (VALUES ('closest'), ('minimum_travel')) AS election_types(election_type)
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY path_id, formation_id, row_order, fail_time, election_type
    ) AS runs_id,
    path_id,
    formation_id,
    leader_failure_enabled,
    fail_time,
    init_leader,
    election_type,
    status
FROM run_rows
ORDER BY runs_id
"""


def generate_runs(
    paths_path: str | Path = DEFAULT_PATHS_FILE,
    formations_path: str | Path = DEFAULT_FORMATIONS_FILE,
) -> Iterator[dict[str, object]]:
    """Yield runs for every path/formation pair."""
    with duckdb.connect() as connection:
        result = connection.execute(
            _RUN_SELECT,
            [str(paths_path), str(formations_path)],
        )
        columns = [description[0] for description in result.description]
        while rows := result.fetchmany(8192):
            for row in rows:
                yield dict(zip(columns, row))


def make_runs_csv(
    output_path: str | Path = DEFAULT_RUNS_FILE,
    paths_path: str | Path = DEFAULT_PATHS_FILE,
    formations_path: str | Path = DEFAULT_FORMATIONS_FILE,
) -> Path:
    """Stream all generated runs into the runs registry CSV."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute(
            f"COPY ({_RUN_SELECT}) TO ? (HEADER, DELIMITER ',')",
            [str(paths_path), str(formations_path), str(destination)],
        )
    return destination


def edit_run(
    runs_id: int,
    output_path: str | Path = DEFAULT_RUNS_FILE,
    **updates: object,
) -> Path:
    """Edit one run row while streaming the remaining rows unchanged."""
    if not isinstance(runs_id, int) or isinstance(runs_id, bool):
        raise TypeError("runs_id must be an integer.")
    if any(field not in RUN_COLUMNS[1:] for field in updates):
        raise ValueError("Updates contain an unknown or non-editable field.")
    if "election_type" in updates:
        validate_choice(str(updates["election_type"]), ELECTION_TYPES, "election_type")
    if "status" in updates:
        validate_choice(str(updates["status"]), RUN_STATUSES, "status")

    source = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)

    temporary_path = _update_csv_with_duckdb(source, runs_id, updates)
    if temporary_path is None:
        raise ValueError(f"No run with runs_id={runs_id} exists.")
    return source


def _update_csv_with_duckdb(
    source: Path,
    runs_id: int,
    updates: dict[str, object],
    same_configuration: bool = False,
) -> Path | None:
    """Update one CSV row through a DuckDB temporary table."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        dir=source.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with duckdb.connect() as connection:
            connection.execute(
                "CREATE TEMP TABLE runs AS SELECT * FROM read_csv_auto(?)",
                [str(source)],
            )
            selected = connection.execute(
                "SELECT path_id, formation_id FROM runs WHERE runs_id = ?",
                [runs_id],
            ).fetchone()
            if selected is None:
                return None
            if same_configuration:
                where_clause = "path_id = ? AND formation_id = ?"
                where_values = [selected[0], selected[1]]
            else:
                where_clause = "runs_id = ?"
                where_values = [runs_id]
            for field, value in updates.items():
                connection.execute(
                    f'UPDATE runs SET "{field}" = ? WHERE {where_clause}',
                    [value, *where_values],
                )
            columns = ", ".join(f'"{column}"' for column in RUN_COLUMNS)
            connection.execute(
                f"COPY (SELECT {columns} FROM runs ORDER BY runs_id) "
                "TO ? (HEADER, DELIMITER ',')",
                [str(temporary_path)],
            )
        temporary_path.replace(source)
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def edit_run_status(
    run_id: int,
    new_status: str,
    output_path: str | Path = DEFAULT_RUNS_FILE,
) -> Path:
    """Update one run status, propagating infeasibility by configuration."""
    if not isinstance(run_id, int) or isinstance(run_id, bool):
        raise TypeError("run_id must be an integer.")
    validate_choice(new_status, RUN_STATUSES, "new_status")
    source = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)
    updated = _update_csv_with_duckdb(
        source,
        run_id,
        {"status": new_status},
        same_configuration=new_status == "infeasible",
    )
    if updated is None:
        raise ValueError(f"No run with runs_id={run_id} exists.")
    return source


def ensure_status_column(output_path: str | Path = DEFAULT_RUNS_FILE) -> Path:
    """Add ``status=planned`` to an older runs CSV in a streaming pass."""
    source = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(source)

    with duckdb.connect() as connection:
        columns = [row[0] for row in connection.execute(
            "DESCRIBE SELECT * FROM read_csv_auto(?)", [str(source)]
        ).fetchall()]
    if columns == RUN_COLUMNS:
        return source
    if columns != RUN_COLUMNS[:-1]:
        raise ValueError(f"CSV columns must be {RUN_COLUMNS[:-1]}, got {columns}.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", dir=source.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with duckdb.connect() as connection:
            connection.execute(
                "COPY (SELECT *, 'planned' AS status FROM read_csv_auto(?)) "
                "TO ? (HEADER, DELIMITER ',')",
                [str(source), str(temporary_path)],
            )
        temporary_path.replace(source)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return source


if __name__ == "__main__":
    make_runs_csv()
