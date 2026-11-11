"""Create and append per-drone failed-run analysis metrics."""

from pathlib import Path

try:
    from experiments.setup.common import DATABASE_PATH, connect_database
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from experiments.setup.common import DATABASE_PATH, connect_database


FAIL_RUNS_DRONE_METRICS_TABLE = "analysis.fail_runs_drone_metrics"
DEFAULT_DATABASE = DATABASE_PATH

FAIL_RUNS_DRONE_METRICS_COLUMNS = [
    "run_id",
    "baseline_nonfail_run_id",
    "drone",
    "role",
    "reference_drone",
    "distance_to_old_leader_at_change",
    "distance_rank_at_change",
    "mean_error",
    "rms_error",
    "peak_error",
    "peak_error_time",
    "error_integral",
    "final_error",
    "status_005",
    "time_to_match_005",
    "excursions_005",
    "status_0001",
    "time_to_match_0001",
    "excursions_0001",
]


def _create_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FAIL_RUNS_DRONE_METRICS_TABLE} (
            run_id BIGINT,
            baseline_nonfail_run_id BIGINT,
            drone BIGINT,
            role VARCHAR,
            reference_drone BIGINT,
            distance_to_old_leader_at_change DOUBLE,
            distance_rank_at_change BIGINT,
            mean_error DOUBLE,
            rms_error DOUBLE,
            peak_error DOUBLE,
            peak_error_time DOUBLE,
            error_integral DOUBLE,
            final_error DOUBLE,
            status_005 VARCHAR,
            time_to_match_005 DOUBLE,
            excursions_005 BIGINT,
            status_0001 VARCHAR,
            time_to_match_0001 DOUBLE,
            excursions_0001 BIGINT
        )
        """
    )


def make_fail_runs_drone_metrics_table(
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Create or replace the per-drone failed-run metrics table."""
    with connect_database() as connection:
        connection.execute(f"DROP TABLE IF EXISTS {FAIL_RUNS_DRONE_METRICS_TABLE}")
        _create_table(connection)
    return Path(output_path)


def data_input(
    run_id: int,
    baseline_nonfail_run_id: int,
    drone: int,
    role: str,
    reference_drone: int | None,
    distance_to_old_leader_at_change: float | None,
    distance_rank_at_change: int | None,
    mean_error: float,
    rms_error: float,
    peak_error: float,
    peak_error_time: float,
    error_integral: float,
    final_error: float,
    status_005: str,
    time_to_match_005: float | None,
    excursions_005: int,
    status_0001: str,
    time_to_match_0001: float | None,
    excursions_0001: int,
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Append one per-drone failed-run metrics row."""
    required_integers = (
        run_id,
        baseline_nonfail_run_id,
        drone,
        excursions_005,
        excursions_0001,
    )
    optional_integers = (reference_drone, distance_rank_at_change)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in required_integers
    ):
        raise TypeError(
            "run IDs, drone IDs, and excursion counts must be integers."
        )
    if any(
        value is not None
        and (isinstance(value, bool) or not isinstance(value, int))
        for value in optional_integers
    ):
        raise TypeError("reference_drone and distance rank must be integers or None.")

    values = [
        run_id,
        baseline_nonfail_run_id,
        drone,
        role,
        reference_drone,
        distance_to_old_leader_at_change,
        distance_rank_at_change,
        mean_error,
        rms_error,
        peak_error,
        peak_error_time,
        error_integral,
        final_error,
        status_005,
        time_to_match_005,
        excursions_005,
        status_0001,
        time_to_match_0001,
        excursions_0001,
    ]
    with connect_database() as connection:
        _create_table(connection)
        connection.execute(
            f"""
            INSERT INTO {FAIL_RUNS_DRONE_METRICS_TABLE}
            VALUES ({", ".join("?" for _ in values)})
            """,
            values,
        )
    return Path(output_path)


if __name__ == "__main__":
    print(
        f"Created {FAIL_RUNS_DRONE_METRICS_TABLE} in "
        f"{make_fail_runs_drone_metrics_table()}"
    )
