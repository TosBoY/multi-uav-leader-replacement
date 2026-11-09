"""Create and append feasibility-analysis records."""

from pathlib import Path

try:
    from experiments.setup.common import DATABASE_PATH, connect_database
except ModuleNotFoundError:
    from common import DATABASE_PATH, connect_database


FEASIBLE_COLUMNS = [
    "run_id",
    "MAX_POSITION_DEVIATION",
    "MAX_DEVIATION_DURATION",
    "run_file",
    "trajectory_file",
    "reason",
    "drone",
    "infeasible_since",
]
DEFAULT_DATABASE = DATABASE_PATH


def make_feasible_table(
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Create or replace the empty feasibility registry table."""
    with connect_database() as connection:
        connection.execute("DROP TABLE IF EXISTS registry.feasible")
        connection.execute(
            """
            CREATE TABLE registry.feasible (
                run_id BIGINT,
                "MAX_POSITION_DEVIATION" DOUBLE,
                "MAX_DEVIATION_DURATION" DOUBLE,
                run_file VARCHAR,
                trajectory_file VARCHAR,
                reason VARCHAR,
                drone VARCHAR,
                infeasible_since DOUBLE
            )
            """
        )
    return Path(output_path)


def data_input(
    run_id: int,
    max_position_deviation: float,
    max_deviation_duration: float,
    run_file: str,
    trajectory_file: str,
    reason: str = "",
    drone: int | str | None = None,
    infeasible_since: float | None = None,
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Append one feasibility result to the registry table."""
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise ValueError("run_id must be a positive integer.")
    if max_position_deviation < 0:
        raise ValueError("max_position_deviation must be non-negative.")
    if max_deviation_duration < 0:
        raise ValueError("max_deviation_duration must be non-negative.")

    with connect_database() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS registry.feasible (
                run_id BIGINT,
                "MAX_POSITION_DEVIATION" DOUBLE,
                "MAX_DEVIATION_DURATION" DOUBLE,
                run_file VARCHAR,
                trajectory_file VARCHAR,
                reason VARCHAR,
                drone VARCHAR,
                infeasible_since DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO registry.feasible VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, max_position_deviation, max_deviation_duration, run_file,
                trajectory_file, reason, None if drone is None else str(drone),
                infeasible_since,
            ],
        )
    return Path(output_path)


if __name__ == "__main__":
    make_feasible_table()


make_feasible_csv = make_feasible_table
