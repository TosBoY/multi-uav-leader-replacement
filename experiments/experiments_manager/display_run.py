"""Run and display one registered experiment without saving its outputs."""

from pathlib import Path
from typing import Any, cast

try:
    from experiments.setup.common import connect_database
    from simulation.experiment_maker import make_experiment
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.setup.common import connect_database
    from simulation.experiment_maker import make_experiment


def _load_run(run_id: int) -> dict[str, Any]:
    """Load one run and its path and formation configuration."""
    with connect_database() as connection:
        result = connection.execute(
            """
            SELECT
                r.runs_id,
                r.leader_failure_enabled,
                r.fail_time,
                r.init_leader,
                r.election_type,
                p.path_type,
                p.first_waypoint_x,
                p.first_waypoint_y,
                p.first_waypoint_z,
                p.velocity,
                p.time,
                p.seed,
                p.initial_heading,
                f.number_of_drone,
                f.shape,
                f.spread,
                f.plane,
                f.leader_position_x,
                f.leader_position_y,
                f.leader_position_z
            FROM registry.runs AS r
            INNER JOIN registry.paths AS p ON p.path_id = r.path_id
            INNER JOIN registry.formations AS f
                ON f.formation_id = r.formation_id
            WHERE r.runs_id = ?
            """,
            [run_id],
        )
        row = result.fetchone()
        if row is None:
            raise ValueError(f"No run with runs_id={run_id} exists.")
        return dict(zip([description[0] for description in result.description], row))


def display_run(run_id: int) -> dict[str, Any]:
    """Run and display one registered experiment without writing output files."""
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise ValueError("run_id must be a positive integer.")

    run = _load_run(run_id)
    failure_enabled = bool(run["leader_failure_enabled"])
    result = make_experiment(
        number_of_drones=int(run["number_of_drone"]),
        initial_leader=int(run["init_leader"]),
        leader_position=[
            float(run["leader_position_x"]),
            float(run["leader_position_y"]),
            float(run["leader_position_z"]),
        ],
        shape=cast(Any, str(run["shape"])),
        spread=float(run["spread"]),
        plane=cast(Any, str(run["plane"] or "xy")),
        path_type=cast(Any, str(run["path_type"])),
        velocity=float(run["velocity"]),
        simulation_time=float(run["time"]),
        initial_heading=float(run["initial_heading"]),
        path_seed=None if run["seed"] is None else int(run["seed"]),
        leader_failure_time=(
            float(run["fail_time"]) if failure_enabled else None
        ),
        election_type=cast(Any, str(run["election_type"] or "closest")),
        sim_rate=100,
        display="yes",
        save_data=False,
        print_path=False,
        save_data_filename=None,
        print_path_filename=None,
    )
    if result.get("outcome") == "failed":
        raise RuntimeError(
            f"Run {run_id} failed: {result.get('error', 'unknown error')}"
        )
    return result


def main() -> None:
    """Read a run ID from the console and display that experiment."""
    try:
        run_id = int(input("Enter run ID: ").strip())
    except ValueError as error:
        raise ValueError("Run ID must be a positive integer.") from error
    display_run(run_id)


if __name__ == "__main__":
    main()
