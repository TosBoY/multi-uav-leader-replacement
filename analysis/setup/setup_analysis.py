"""Set up all tables in the analysis schema."""

try:
    from analysis.setup.fail_runs_drone_metrics import (
        make_fail_runs_drone_metrics_table,
    )
    from analysis.setup.failed_runs_metrics import make_failed_runs_metrics_table
except ModuleNotFoundError:
    from fail_runs_drone_metrics import make_fail_runs_drone_metrics_table
    from failed_runs_metrics import make_failed_runs_metrics_table


def setup_all() -> None:
    """Create analysis tables in dependency order."""
    print(
        "Created analysis.failed_runs_metrics in "
        f"{make_failed_runs_metrics_table()}"
    )
    print(
        "Created analysis.fail_runs_drone_metrics in "
        f"{make_fail_runs_drone_metrics_table()}"
    )


if __name__ == "__main__":
    setup_all()
