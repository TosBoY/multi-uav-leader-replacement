"""Create and append trajectory difficulty analysis records."""

from pathlib import Path

import duckdb

try:
    from experiments.setup.common import DATABASE_PATH, connect_database
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from experiments.setup.common import DATABASE_PATH, connect_database


TRAJECTORY_DIFFICULTY_TABLE = "analysis.trajectory_difficulty"
DEFAULT_DATABASE = DATABASE_PATH

TRAJECTORY_DIFFICULTY_COLUMNS = [
    "path_id",
    "fail_time",
    "window_name",
    "window_start",
    "window_end",
    "actual_window_duration",
    "path_length_m",
    "mean_speed_m_s",
    "p95_turn_rate_rad_s",
    "p95_acceleration_m_s2",
    "p95_jerk_m_s3",
    "vertical_range_m",
    "difficulty_score",
    "source_file",
]


def _create_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TRAJECTORY_DIFFICULTY_TABLE} (
            path_id BIGINT,
            fail_time DOUBLE,
            window_name VARCHAR,
            window_start DOUBLE,
            window_end DOUBLE,
            actual_window_duration DOUBLE,
            path_length_m DOUBLE,
            mean_speed_m_s DOUBLE,
            p95_turn_rate_rad_s DOUBLE,
            p95_acceleration_m_s2 DOUBLE,
            p95_jerk_m_s3 DOUBLE,
            vertical_range_m DOUBLE,
            difficulty_score DOUBLE,
            source_file VARCHAR
        )
        """
    )


def make_trajectory_difficulty_table(
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Create or replace the empty trajectory difficulty table."""
    with connect_database() as connection:
        connection.execute(
            f"DROP TABLE IF EXISTS {TRAJECTORY_DIFFICULTY_TABLE}"
        )
        _create_table(connection)
    return Path(output_path)


def data_input(
    path_id: int,
    fail_time: float,
    window_name: str,
    window_start: float,
    window_end: float,
    actual_window_duration: float,
    path_length_m: float,
    mean_speed_m_s: float,
    p95_turn_rate_rad_s: float,
    p95_acceleration_m_s2: float,
    p95_jerk_m_s3: float,
    vertical_range_m: float,
    difficulty_score: float,
    source_file: str,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Append one trajectory difficulty row."""
    values = [
        path_id,
        fail_time,
        window_name,
        window_start,
        window_end,
        actual_window_duration,
        path_length_m,
        mean_speed_m_s,
        p95_turn_rate_rad_s,
        p95_acceleration_m_s2,
        p95_jerk_m_s3,
        vertical_range_m,
        difficulty_score,
        source_file,
    ]
    placeholders = ", ".join("?" for _ in values)

    if connection is None:
        with connect_database() as connection:
            _create_table(connection)
            connection.execute(
                f"""
                INSERT INTO {TRAJECTORY_DIFFICULTY_TABLE}
                VALUES ({placeholders})
                """,
                values,
            )
    else:
        _create_table(connection)
        connection.execute(
            f"""
            INSERT INTO {TRAJECTORY_DIFFICULTY_TABLE}
            VALUES ({placeholders})
            """,
            values,
        )


if __name__ == "__main__":
    print(
        f"Created {TRAJECTORY_DIFFICULTY_TABLE} in "
        f"{make_trajectory_difficulty_table()}"
    )
