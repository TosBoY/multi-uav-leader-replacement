"""Run registered experiments and record their outcomes."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb

try:
    from analysis.run_analysis.check_feasibility import check_feasibility
    from experiments.setup.common import REGISTRY_DIRECTORY
    from experiments.setup.feasible import data_input as feasible_data_input
    from experiments.setup.runs import edit_run_status
    from experiments.setup.runs_result import data_input as result_data_input
    from simulation.experiment_maker import make_experiment
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from analysis.run_analysis.check_feasibility import check_feasibility
    from experiments.setup.common import REGISTRY_DIRECTORY
    from experiments.setup.feasible import data_input as feasible_data_input
    from experiments.setup.runs import edit_run_status
    from experiments.setup.runs_result import data_input as result_data_input
    from simulation.experiment_maker import make_experiment


_REGISTRY_DIRECTORY = REGISTRY_DIRECTORY
_RUNS_FILE = _REGISTRY_DIRECTORY / "runs.csv"
_FORMATIONS_FILE = _REGISTRY_DIRECTORY / "formations.csv"
_PATHS_FILE = _REGISTRY_DIRECTORY / "paths.csv"


def _range_filter(
    value: Sequence[int] | None,
    column: str,
    parameters: list[Any],
) -> str:
    """Build an inclusive numeric filter from an empty, one, or two-value input."""
    if value is None or len(value) == 0:
        return ""
    if len(value) == 2:
        parameters.extend([min(value), max(value)])
        return f" AND {column} BETWEEN ? AND ?"
    parameters.extend(value)
    placeholders = ", ".join("?" for _ in value)
    return f" AND {column} IN ({placeholders})"


def _shape_filter(
    shapes: Sequence[str] | None,
    parameters: list[Any],
) -> str:
    """Build a shape filter; an empty input means all shapes."""
    if shapes is None or len(shapes) == 0:
        return ""
    parameters.extend(shapes)
    placeholders = ", ".join("?" for _ in shapes)
    return f" AND f.shape IN ({placeholders})"


def _load_runs(
    run_id_range: Sequence[int] | None,
    number_of_drone_range: Sequence[int] | None,
    shape_range: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Load filtered run configurations in ascending run-ID order."""
    parameters: list[Any] = [str(_RUNS_FILE), str(_FORMATIONS_FILE), str(_PATHS_FILE)]
    filters = "".join(
        (
            _range_filter(run_id_range, "r.runs_id", parameters),
            _range_filter(number_of_drone_range, "f.number_of_drone", parameters),
            _shape_filter(shape_range, parameters),
        )
    )
    query = f"""
        SELECT
            r.runs_id,
            r.leader_failure_enabled,
            r.fail_time,
            r.init_leader,
            r.election_type,
            r.status,
            f.number_of_drone,
            f.shape,
            f.spread,
            f.plane,
            f.leader_index,
            f.leader_position_x,
            f.leader_position_y,
            f.leader_position_z,
            p.path_type,
            p.velocity,
            p.time,
            p.seed,
            p.initial_heading
        FROM read_csv_auto(?) AS r
        INNER JOIN read_csv_auto(?) AS f
            ON r.formation_id = f.formation_id
        INNER JOIN read_csv_auto(?) AS p
            ON r.path_id = p.path_id
        WHERE TRUE
        {filters}
        ORDER BY r.runs_id
    """
    with duckdb.connect() as connection:
        result = connection.execute(query, parameters)
        columns = [description[0] for description in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def _relative_data_file(run_id: int, failure_enabled: bool) -> str:
    category = "fail" if failure_enabled else "nonfail"
    return f"results/data/{category}/{run_id}.parquet"


def _record_result(
    run_id: int,
    result: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    data_file: str,
) -> None:
    """Append simulation metadata to runs_result.csv."""
    result_data_input(
        run_id=run_id,
        outcome=result["outcome"],
        error=result.get("error", "") or None,
        started_at=started_at,
        finished_at=finished_at,
        duration=float(result.get("duration", 0.0)),
        data_file=data_file,
        n_steps=int(result.get("n_steps", 0)),
        promoted_leader=result.get("promoted_leader"),
        promotion_time=result.get("promotion_time"),
    )


def _record_failed_exception(
    run_id: int,
    error: Exception,
    started_at: datetime,
    data_file: str,
) -> None:
    finished_at = datetime.now()
    result_data_input(
        run_id=run_id,
        outcome="failed",
        error=f"{type(error).__name__}: {error}",
        started_at=started_at,
        finished_at=finished_at,
        duration=0.0,
        data_file=data_file,
        n_steps=0,
    )
    edit_run_status(run_id, "failed")


def run_experiments(
    run_id_range: Sequence[int] | None = None,
    number_of_drone_range: Sequence[int] | None = None,
    shape_range: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run selected registry experiments in ascending ``run_id`` order.

    Empty or ``None`` filters select all values for that field. Numeric ranges
    with two values are inclusive; other numeric sequences select exact IDs.
    """
    runs = _load_runs(run_id_range, number_of_drone_range, shape_range)
    recorded_results: list[dict[str, Any]] = []

    for run in runs:
        run_id = int(run["runs_id"])
        print("\n" + "=" * 72)
        if str(run["status"]) == "infeasible":
            print(f"Skipping run {run_id}: already infeasible")
            print("=" * 72)
            continue

        failure_enabled = bool(run["leader_failure_enabled"])
        data_file = _relative_data_file(run_id, failure_enabled)
        trajectory_file = f"results/data/trajectories/{run_id}_path.parquet"
        started_at = datetime.now()
        print(
            f"Running run {run_id} "
            f"(path/formation: {run['path_type']}/{run['shape']}, "
            f"failure: {failure_enabled})"
        )
        edit_run_status(run_id, "running")

        try:
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
                path_seed=(None if run["seed"] in (None, "") else int(run["seed"])),
                leader_failure_time=(
                    None if not failure_enabled else float(run["fail_time"])
                ),
                election_type=cast(Any, str(run["election_type"])),
                sim_rate=100,
                display=False,
                save_data=True,
                save_data_filename=data_file.removeprefix("results/data/"),
                print_path=not failure_enabled,
                print_path_filename=(
                    None if failure_enabled else f"{run_id}_path.parquet"
                ),
            )
            finished_at = datetime.now()
            if not failure_enabled and result["outcome"] == "done":
                feasibility = check_feasibility(
                    f"{run_id}.parquet",
                    f"{run_id}_path.parquet",
                )
                print(f"Run {run_id} feasibility: {feasibility}")
                if feasibility["result"] == "infeasible":
                    feasible_data_input(
                        run_id=run_id,
                        max_position_deviation=feasibility["MAX_POSITION_DEVIATION"],
                        max_deviation_duration=feasibility["MAX_DEVIATION_DURATION"],
                        run_file=feasibility["run_file"],
                        trajectory_file=feasibility["trajectory_file"],
                        reason=feasibility["reason"],
                        drone=feasibility.get("drone"),
                        infeasible_since=feasibility.get("infeasible_since"),
                    )
                    edit_run_status(run_id, "infeasible")
                    result["outcome"] = "infeasible"
                else:
                    edit_run_status(run_id, "done")
            else:
                edit_run_status(run_id, result["outcome"])

            _record_result(run_id, result, started_at, finished_at, data_file)
            print(f"Run {run_id} result: {result['outcome']}")
            print("=" * 72)
            recorded_results.append(result)
        except Exception as error:
            _record_failed_exception(run_id, error, started_at, data_file)
            print(f"Run {run_id} result: failed ({error})")
            print("=" * 72)
            recorded_results.append(
                {
                    "outcome": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return recorded_results


if __name__ == "__main__":
    run_experiments()
