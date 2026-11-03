"""Create and append feasibility-analysis records."""

from pathlib import Path
import csv

try:
    from experiments.setup.common import REGISTRY_DIRECTORY
except ModuleNotFoundError:
    from common import REGISTRY_DIRECTORY


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
DEFAULT_FEASIBLE_FILE = REGISTRY_DIRECTORY / "feasible.csv"


def make_feasible_csv(
    output_path: str | Path = DEFAULT_FEASIBLE_FILE,
) -> Path:
    """Create an empty feasibility registry with its required header."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(FEASIBLE_COLUMNS)
    return destination


def data_input(
    run_id: int,
    max_position_deviation: float,
    max_deviation_duration: float,
    run_file: str,
    trajectory_file: str,
    reason: str = "",
    drone: int | str | None = None,
    infeasible_since: float | None = None,
    output_path: str | Path = DEFAULT_FEASIBLE_FILE,
) -> Path:
    """Append one feasibility result to the registry table."""
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise ValueError("run_id must be a positive integer.")
    if max_position_deviation < 0:
        raise ValueError("max_position_deviation must be non-negative.")
    if max_deviation_duration < 0:
        raise ValueError("max_deviation_duration must be non-negative.")

    destination = Path(output_path)
    if not destination.exists() or destination.stat().st_size == 0:
        make_feasible_csv(destination)

    row = {
        "run_id": run_id,
        "MAX_POSITION_DEVIATION": max_position_deviation,
        "MAX_DEVIATION_DURATION": max_deviation_duration,
        "run_file": run_file,
        "trajectory_file": trajectory_file,
        "reason": reason,
        "drone": "" if drone is None else drone,
        "infeasible_since": "" if infeasible_since is None else infeasible_since,
    }
    with destination.open("a", newline="", encoding="utf-8") as csv_file:
        csv.DictWriter(csv_file, fieldnames=FEASIBLE_COLUMNS).writerow(row)
    return destination


if __name__ == "__main__":
    make_feasible_csv()
