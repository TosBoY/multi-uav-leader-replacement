"""Create and append failed-run analysis metrics."""

from pathlib import Path

try:
    from experiments.setup.common import DATABASE_PATH, connect_database
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from experiments.setup.common import DATABASE_PATH, connect_database


FAILED_RUNS_METRICS_TABLE = "analysis.failed_runs_metrics"
DEFAULT_DATABASE = DATABASE_PATH

FAILED_RUNS_METRICS_COLUMNS = [
    "run_id",
    "baseline_nonfail_run_id",
    "number_of_drone",
    "shape",
    "spread",
    "plane",
    "election_type",
    "fail_time",
    "promoted_leader",
    "mean_error",
    "peak_error",
    "peak_error_time",
    "error_integral_sum",
    "final_error_max",
    "n_matched_005",
    "n_never_matched_005",
    "n_unstable_005",
    "time_to_match_mean_005",
    "time_to_match_median_005",
    "time_to_match_max_005",
    "settle_time_005",
    "n_matched_0001",
    "n_never_matched_0001",
    "n_unstable_0001",
    "time_to_match_mean_0001",
    "time_to_match_median_0001",
    "time_to_match_max_0001",
    "settle_time_0001",
    "min_separation",
    "min_separation_time",
    "baseline_min_separation",
    "n_close_steps",
]


def _create_table(connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FAILED_RUNS_METRICS_TABLE} (
            run_id BIGINT,
            baseline_nonfail_run_id BIGINT,
            number_of_drone BIGINT,
            shape VARCHAR,
            spread DOUBLE,
            plane VARCHAR,
            election_type VARCHAR,
            fail_time DOUBLE,
            promoted_leader VARCHAR,
            mean_error DOUBLE,
            peak_error DOUBLE,
            peak_error_time DOUBLE,
            error_integral_sum DOUBLE,
            final_error_max DOUBLE,
            n_matched_005 BIGINT,
            n_never_matched_005 BIGINT,
            n_unstable_005 BIGINT,
            time_to_match_mean_005 DOUBLE,
            time_to_match_median_005 DOUBLE,
            time_to_match_max_005 DOUBLE,
            settle_time_005 DOUBLE,
            n_matched_0001 BIGINT,
            n_never_matched_0001 BIGINT,
            n_unstable_0001 BIGINT,
            time_to_match_mean_0001 DOUBLE,
            time_to_match_median_0001 DOUBLE,
            time_to_match_max_0001 DOUBLE,
            settle_time_0001 DOUBLE,
            min_separation DOUBLE,
            min_separation_time DOUBLE,
            baseline_min_separation DOUBLE,
            n_close_steps BIGINT
        )
        """
    )


def make_failed_runs_metrics_table(
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Create or replace the empty failed-run metrics table."""
    with connect_database() as connection:
        connection.execute("DROP TABLE IF EXISTS analysis.runs_metrics")
        connection.execute(f"DROP TABLE IF EXISTS {FAILED_RUNS_METRICS_TABLE}")
        _create_table(connection)
    return Path(output_path)


def data_input(
    run_id: int,
    baseline_nonfail_run_id: int,
    number_of_drone: int,
    shape: str,
    spread: float,
    plane: str | None,
    election_type: str,
    fail_time: float,
    promoted_leader: int | str,
    mean_error: float,
    peak_error: float,
    peak_error_time: float,
    error_integral_sum: float,
    final_error_max: float,
    n_matched_005: int,
    n_never_matched_005: int,
    n_unstable_005: int,
    time_to_match_mean_005: float,
    time_to_match_median_005: float,
    time_to_match_max_005: float,
    settle_time_005: float,
    n_matched_0001: int,
    n_never_matched_0001: int,
    n_unstable_0001: int,
    time_to_match_mean_0001: float,
    time_to_match_median_0001: float,
    time_to_match_max_0001: float,
    settle_time_0001: float,
    min_separation: float,
    min_separation_time: float,
    baseline_min_separation: float,
    n_close_steps: int,
    output_path: str | Path = DEFAULT_DATABASE,
) -> Path:
    """Append one failed-run metrics row."""
    integer_values = (
        run_id,
        baseline_nonfail_run_id,
        number_of_drone,
        n_matched_005,
        n_never_matched_005,
        n_unstable_005,
        n_matched_0001,
        n_never_matched_0001,
        n_unstable_0001,
        n_close_steps,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values
    ):
        raise TypeError("run IDs, drone counts, and metric counts must be integers.")

    values = [
        run_id, baseline_nonfail_run_id, number_of_drone, shape, spread, plane,
        election_type, fail_time, str(promoted_leader), mean_error, peak_error,
        peak_error_time, error_integral_sum, final_error_max, n_matched_005,
        n_never_matched_005, n_unstable_005, time_to_match_mean_005,
        time_to_match_median_005, time_to_match_max_005, settle_time_005,
        n_matched_0001, n_never_matched_0001, n_unstable_0001,
        time_to_match_mean_0001, time_to_match_median_0001,
        time_to_match_max_0001, settle_time_0001, min_separation,
        min_separation_time, baseline_min_separation, n_close_steps,
    ]
    with connect_database() as connection:
        _create_table(connection)
        connection.execute(
            f"""
            INSERT INTO {FAILED_RUNS_METRICS_TABLE}
            VALUES ({", ".join("?" for _ in values)})
            """,
            values,
        )
    return Path(output_path)


if __name__ == "__main__":
    print(
        f"Created {FAILED_RUNS_METRICS_TABLE} in "
        f"{make_failed_runs_metrics_table()}"
    )
