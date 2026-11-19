"""Set up all tables in the analysis schema."""

try:
    from analysis.setup.fail_runs_drone_metrics import (
        make_fail_runs_drone_metrics_table,
    )
    from analysis.setup.failed_runs_metrics import make_failed_runs_metrics_table
    from analysis.setup.trajectory_difficulty import (
        make_trajectory_difficulty_table,
    )
except ModuleNotFoundError:
    from fail_runs_drone_metrics import make_fail_runs_drone_metrics_table
    from failed_runs_metrics import make_failed_runs_metrics_table
    from trajectory_difficulty import make_trajectory_difficulty_table


def setup_all() -> None:
    """Create analysis tables in dependency order."""
    print(
        "Created analysis.failed_runs_metrics in "
        f"{make_failed_runs_metrics_table(reset=True)}"
    )
    print(
        "Created analysis.fail_runs_drone_metrics in "
        f"{make_fail_runs_drone_metrics_table(reset=True)}"
    )
    print(
        "Created analysis.trajectory_difficulty in "
        f"{make_trajectory_difficulty_table(reset=True)}"
    )


if __name__ == "__main__":
    setup_all()
