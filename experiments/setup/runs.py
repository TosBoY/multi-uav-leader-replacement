"""Generate and edit the experiment runs setup table."""

from collections.abc import Iterator
from pathlib import Path

try:
    from experiments.setup.common import (
        DATABASE_PATH,
        connect_database,
        validate_choice,
    )
except ModuleNotFoundError:
    from common import (
        DATABASE_PATH,
        connect_database,
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
ELECTION_TYPES = ("closest", "minimum_travel", "smallest_id", "central")
RUN_STATUSES = ("planned", "running", "done", "failed", "infeasible", "cancelled")
FAIL_TIMES = tuple(2.5 + 0.5 * index for index in range(11))
DEFAULT_DATABASE = DATABASE_PATH


_RUN_SELECT = """
WITH path_ids AS (
    SELECT DISTINCT CAST(path_id AS BIGINT) AS path_id
    FROM registry.paths
),
formation_ids AS (
    SELECT DISTINCT CAST(formation_id AS BIGINT) AS formation_id
    FROM registry.formations
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
    CROSS JOIN (
        VALUES ('closest'), ('minimum_travel'), ('smallest_id'), ('central')
    ) AS election_types(election_type)
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
    database_path: str | Path = DEFAULT_DATABASE,
) -> Iterator[dict[str, object]]:
    """Yield runs for every path/formation pair."""
    with connect_database() as connection:
        result = connection.execute(_RUN_SELECT)
        columns = [description[0] for description in result.description]
        while rows := result.fetchmany(8192):
            for row in rows:
                yield dict(zip(columns, row))


def make_runs_table(
    output_path: str | Path = DEFAULT_DATABASE,
    database_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Generate and replace the runs registry table."""
    with connect_database() as connection:
        connection.execute("DROP TABLE IF EXISTS registry.runs")
        connection.execute(f"CREATE TABLE registry.runs AS {_RUN_SELECT}")
    return Path(output_path)


def edit_run(
    runs_id: int,
    output_path: str | Path = DEFAULT_DATABASE,
    **updates: object,
) -> Path:
    """Edit one run row in the runs registry table."""
    if not isinstance(runs_id, int) or isinstance(runs_id, bool):
        raise TypeError("runs_id must be an integer.")
    if any(field not in RUN_COLUMNS[1:] for field in updates):
        raise ValueError("Updates contain an unknown or non-editable field.")
    if "election_type" in updates:
        validate_choice(str(updates["election_type"]), ELECTION_TYPES, "election_type")
    if "status" in updates:
        validate_choice(str(updates["status"]), RUN_STATUSES, "status")

    with connect_database() as connection:
        selected = connection.execute(
            "SELECT path_id, formation_id FROM registry.runs WHERE runs_id = ?",
            [runs_id],
        ).fetchone()
        if selected is None:
            raise ValueError(f"No run with runs_id={runs_id} exists.")
        for field, value in updates.items():
            connection.execute(
                f'UPDATE registry.runs SET "{field}" = ? WHERE runs_id = ?',
                [value, runs_id],
            )
    return Path(output_path)


def edit_run_status(
    run_id: int,
    new_status: str,
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Update one run status, propagating infeasibility by configuration."""
    if not isinstance(run_id, int) or isinstance(run_id, bool):
        raise TypeError("run_id must be an integer.")
    validate_choice(new_status, RUN_STATUSES, "new_status")
    with connect_database() as connection:
        selected = connection.execute(
            "SELECT path_id, formation_id FROM registry.runs WHERE runs_id = ?",
            [run_id],
        ).fetchone()
        if selected is None:
            raise ValueError(f"No run with runs_id={run_id} exists.")
        if new_status == "infeasible":
            connection.execute(
                """
                UPDATE registry.runs
                SET status = ?
                WHERE path_id = ? AND formation_id = ?
                """,
                [new_status, selected[0], selected[1]],
            )
        else:
            connection.execute(
                "UPDATE registry.runs SET status = ? WHERE runs_id = ?",
                [new_status, run_id],
            )
    return Path(output_path)


def ensure_status_column(output_path: str | Path = DEFAULT_DATABASE) -> Path:
    """Add ``status=planned`` to an older runs registry table."""
    with connect_database() as connection:
        columns = [row[0] for row in connection.execute(
            "DESCRIBE registry.runs"
        ).fetchall()]
    if columns == RUN_COLUMNS:
        return Path(output_path)
    if columns != RUN_COLUMNS[:-1]:
        raise ValueError(f"Columns must be {RUN_COLUMNS[:-1]}, got {columns}.")
    with connect_database() as connection:
        connection.execute(
            "ALTER TABLE registry.runs ADD COLUMN status VARCHAR DEFAULT 'planned'"
        )
    return Path(output_path)


if __name__ == "__main__":
    make_runs_table()


make_runs_csv = make_runs_table
