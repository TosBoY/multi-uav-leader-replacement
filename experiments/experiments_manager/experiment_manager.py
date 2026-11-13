"""Run registered experiments and record their outcomes."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

CONCURRENT_SIMULATIONS = 8
PARQUET_MERGE_BATCH_SIZE = 10000


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as days, hours, minutes, and seconds."""
    whole_seconds = int(seconds)
    days, remainder = divmod(whole_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, remaining_seconds = divmod(remainder, 60)

    if days:
        return f"{days}d {hours}h {minutes}m {remaining_seconds}s"
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{seconds:.2f}s"


try:
    from analysis.per_run_analysis.check_feasibility import check_feasibility
    from experiments.setup.common import connect_database
    from experiments.setup.feasible import data_input as feasible_data_input
    from experiments.setup.runs import edit_run_status
    from experiments.setup.runs_result import data_input as result_data_input
    from experiments.experiments_manager.parquet_merger import merge_parquet_files
    from simulation.experiment_maker import make_experiment
except ModuleNotFoundError:
    from pathlib import Path as _Path
    import sys

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from analysis.per_run_analysis.check_feasibility import check_feasibility
    from experiments.setup.common import connect_database
    from experiments.setup.feasible import data_input as feasible_data_input
    from experiments.setup.runs import edit_run_status
    from experiments.setup.runs_result import data_input as result_data_input
    from experiments.experiments_manager.parquet_merger import merge_parquet_files
    from simulation.experiment_maker import make_experiment


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
    parameters: list[Any] = []
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
            r.path_id,
            r.formation_id,
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
        FROM registry.runs AS r
        INNER JOIN registry.formations AS f
            ON r.formation_id = f.formation_id
        INNER JOIN registry.paths AS p
            ON r.path_id = p.path_id
        WHERE TRUE
        {filters}
        ORDER BY r.runs_id
    """
    with connect_database() as connection:
        result = connection.execute(query, parameters)
        columns = [description[0] for description in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def _load_current_statuses(run_ids: Sequence[int]) -> dict[int, str]:
    """Load current statuses for selected runs immediately before batching."""
    if not run_ids:
        return {}
    placeholders = ", ".join("?" for _ in run_ids)
    with connect_database() as connection:
        rows = connection.execute(
            f"""
            SELECT runs_id, status
            FROM registry.runs
            WHERE runs_id IN ({placeholders})
            """,
            list(run_ids),
        ).fetchall()
    return {int(run_id): str(status) for run_id, status in rows}


def _record_result(
    run_id: int,
    result: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Append simulation metadata to the runs-result registry table."""
    result_data_input(
        run_id=run_id,
        outcome=result["outcome"],
        error=result.get("error", "") or None,
        started_at=started_at,
        finished_at=finished_at,
        duration=float(result.get("duration", 0.0)),
        n_steps=int(result.get("n_steps", 0)),
        promoted_leader=result.get("promoted_leader"),
        promotion_time=result.get("promotion_time"),
    )


def _record_failed_exception(
    run_id: int,
    error: Exception,
    started_at: datetime,
) -> None:
    finished_at = datetime.now()
    result_data_input(
        run_id=run_id,
        outcome="failed",
        error=f"{type(error).__name__}: {error}",
        started_at=started_at,
        finished_at=finished_at,
        duration=0.0,
        n_steps=0,
    )
    edit_run_status(run_id, "failed")


def _execute_run(run: dict[str, Any]) -> dict[str, Any]:
    """Run one experiment in a worker process without touching the database."""
    run_id = int(run["runs_id"])
    started_at = datetime.now()
    failure_enabled = bool(run["leader_failure_enabled"])
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
            save_data_filename=(
                f"{'fail' if failure_enabled else 'nonfail'}/{run_id}.parquet"
            ),
            print_path=not failure_enabled,
            print_path_filename=(
                None if failure_enabled else f"{run_id}_path.parquet"
            ),
        )
    except Exception as error:
        result = {
            "outcome": "failed",
            "error": f"{type(error).__name__}: {error}",
            "duration": 0.0,
            "n_steps": 0,
            "promoted_leader": None,
            "promotion_time": None,
        }
    return {
        "run_id": run_id,
        "failure_enabled": failure_enabled,
        "result": result,
        "started_at": started_at,
        "finished_at": datetime.now(),
    }


def run_experiments(
    run_id_range: Sequence[int] | None = None,
    number_of_drone_range: Sequence[int] | None = None,
    shape_range: Sequence[str] | None = None,
    concurrent_simulations: int = CONCURRENT_SIMULATIONS,
) -> list[dict[str, Any]]:
    """Run selected registry experiments in ascending ``run_id`` order.

    Empty or ``None`` filters select all values for that field. Numeric ranges
    with two values are inclusive; other numeric sequences select exact IDs.
    ``concurrent_simulations`` controls the maximum number of worker processes.
    """
    if (
        isinstance(concurrent_simulations, bool)
        or not isinstance(concurrent_simulations, int)
        or concurrent_simulations < 1
    ):
        raise ValueError("concurrent_simulations must be a positive integer.")

    start_time = perf_counter()
    runs = _load_runs(run_id_range, number_of_drone_range, shape_range)
    if runs:
        print(
            f"Running {len(runs)} experiments "
            f"(run IDs {runs[0]['runs_id']}-{runs[-1]['runs_id']}, "
            f"number of drones: {number_of_drone_range or 'all'}, "
            f"shapes: {shape_range or 'all'})"
        )
    else:
        print("No experiments match the selected range.")
    recorded_results: list[dict[str, Any]] = []
    fail_run_ids: list[int] = []
    nonfail_run_ids: list[int] = []
    data_root = Path(__file__).resolve().parents[2] / "results" / "data"

    def merge_fail_data() -> None:
        """Merge and clear the pending failed-run batch."""
        if fail_run_ids:
            merge_parquet_files(data_root / "fail", fail_run_ids)
            fail_run_ids.clear()

    def merge_nonfail_data() -> None:
        """Merge and clear the pending non-failed-run batch."""
        if nonfail_run_ids:
            merge_parquet_files(data_root / "nonfail", nonfail_run_ids)
            merge_parquet_files(data_root / "trajectories", nonfail_run_ids)
            nonfail_run_ids.clear()

    def process_result(worker_result: dict[str, Any]) -> None:
        """Apply one completed worker result in run-ID order."""
        run_id = int(worker_result["run_id"])
        failure_enabled = bool(worker_result["failure_enabled"])
        result = worker_result["result"]
        started_at = cast(datetime, worker_result["started_at"])
        finished_at = cast(datetime, worker_result["finished_at"])

        try:
            if not failure_enabled and result["outcome"] == "done":
                feasibility = check_feasibility(
                    f"{run_id}.parquet",
                    f"{run_id}_path.parquet",
                )
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
                    print(
                        f"Run {run_id} result: infeasible - "
                        f"{feasibility['reason']}"
                    )
                else:
                    edit_run_status(run_id, "done")
            else:
                edit_run_status(run_id, result["outcome"])

            _record_result(run_id, result, started_at, finished_at)
            (fail_run_ids if failure_enabled else nonfail_run_ids).append(run_id)
            if len(fail_run_ids) >= PARQUET_MERGE_BATCH_SIZE:
                merge_fail_data()
            if len(nonfail_run_ids) >= PARQUET_MERGE_BATCH_SIZE:
                merge_nonfail_data()
            if result["outcome"] == "failed":
                print(f"Run {run_id} result: {result['outcome']}")
            recorded_results.append(result)
        except Exception as error:
            _record_failed_exception(run_id, error, started_at)
            print(f"Run {run_id} result: failed ({error})")
            recorded_results.append(
                {
                    "outcome": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    def run_phase(
        phase_runs: Sequence[dict[str, Any]],
        executor: ProcessPoolExecutor,
    ) -> None:
        """Run one phase with a bounded rolling set of worker futures."""
        pending_runs = iter(phase_runs)
        futures: dict[Any, dict[str, Any]] = {}
        pool_error: Exception | None = None

        def submit_next() -> bool:
            nonlocal pool_error
            try:
                run = next(pending_runs)
            except StopIteration:
                return False
            run_id = int(run["runs_id"])
            edit_run_status(run_id, "running")
            try:
                futures[executor.submit(_execute_run, run)] = run
            except Exception as error:
                pool_error = error
                print(f"Failed to submit run {run_id}: {error}")
            return True

        for _ in range(min(concurrent_simulations, len(phase_runs))):
            if not submit_next() or pool_error is not None:
                break

        while futures:
            completed, _ = wait(
                futures,
                return_when=FIRST_COMPLETED,
            )
            for future in as_completed(completed):
                run = futures.pop(future)
                run_id = int(run["runs_id"])
                try:
                    worker_result = future.result()
                except Exception as error:
                    _record_failed_exception(run_id, error, datetime.now())
                    print(f"Run {run_id} result: failed ({error})")
                    recorded_results.append(
                        {
                            "outcome": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                else:
                    process_result(worker_result)
                if pool_error is None:
                    submit_next()

        if pool_error is not None:
            raise pool_error

    with ProcessPoolExecutor(max_workers=concurrent_simulations) as executor:
        current_statuses = _load_current_statuses(
            [int(run["runs_id"]) for run in runs]
        )
        eligible_runs = []
        for run in runs:
            run_id = int(run["runs_id"])
            if current_statuses.get(run_id, str(run["status"])) == "infeasible":
                print(f"Skipping run {run_id}: already infeasible")
                continue
            eligible_runs.append(run)

        baseline_runs = [
            run for run in eligible_runs
            if not bool(run["leader_failure_enabled"])
        ]
        fail_runs = [
            run for run in eligible_runs
            if bool(run["leader_failure_enabled"])
        ]
        run_phase(baseline_runs, executor)

        fail_statuses = _load_current_statuses(
            [int(run["runs_id"]) for run in fail_runs]
        )
        eligible_fail_runs = []
        for run in fail_runs:
            run_id = int(run["runs_id"])
            if fail_statuses.get(run_id, str(run["status"])) == "infeasible":
                print(f"Skipping run {run_id}: already infeasible")
                continue
            eligible_fail_runs.append(run)
        run_phase(eligible_fail_runs, executor)

    merge_fail_data()
    merge_nonfail_data()
    elapsed_time = perf_counter() - start_time
    print(f"Finished experiments in {_format_elapsed_time(elapsed_time)}.")
    return recorded_results


if __name__ == "__main__":
    run_experiments()
