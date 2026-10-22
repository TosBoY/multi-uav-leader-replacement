"""Check whether a recorded run stays close to its intended trajectory."""

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DATA = _PROJECT_ROOT / "results" / "data"
_RUN_DIRECTORY = _RESULTS_DATA / "nonfail"
_TRAJECTORY_DIRECTORY = _RESULTS_DATA / "trajectories"

MAX_POSITION_DEVIATION = 2.0
MAX_DEVIATION_DURATION = 1.5
_POSITION_COLUMNS = ["timestamp", "drone", "x", "y", "z"]


def _load_table(filename: str | Path, directory: Path) -> pd.DataFrame:
    """Load a named Parquet or CSV table from a registry directory."""
    source = directory / Path(filename).name
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported file type: {source.suffix}")


def _validate_position_columns(data: pd.DataFrame, label: str) -> None:
    """Ensure a table contains the position columns required for comparison."""
    missing = [column for column in _POSITION_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def check_feasibility(
    run_filename: str | Path,
    trajectory_filename: str | Path,
) -> dict[str, Any]:
    """Compare actual run positions with intended trajectory positions.

    A run is infeasible when any drone remains more than
    ``MAX_POSITION_DEVIATION`` metres from its intended position for longer
    than ``MAX_DEVIATION_DURATION`` seconds continuously.
    """
    run_data = _load_table(run_filename, _RUN_DIRECTORY)
    trajectory_data = _load_table(trajectory_filename, _TRAJECTORY_DIRECTORY)
    _validate_position_columns(run_data, "run data")
    _validate_position_columns(trajectory_data, "trajectory data")

    run_data = run_data[_POSITION_COLUMNS].copy()
    trajectory_data = trajectory_data[_POSITION_COLUMNS].copy()
    run_data["timestamp"] = pd.to_numeric(run_data["timestamp"]).round(9)
    trajectory_data["timestamp"] = pd.to_numeric(
        trajectory_data["timestamp"]
    ).round(9)
    merged = run_data.merge(
        trajectory_data,
        on=["timestamp", "drone"],
        how="outer",
        suffixes=("_actual", "_intended"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        return {
            "result": "infeasible",
            "run_file": Path(run_filename).name,
            "trajectory_file": Path(trajectory_filename).name,
            "reason": "Run and trajectory timestamps or drones do not match.",
        }

    merged = merged.sort_values(["drone", "timestamp"])
    actual = merged[["x_actual", "y_actual", "z_actual"]].to_numpy(dtype=float)
    intended = merged[["x_intended", "y_intended", "z_intended"]].to_numpy(dtype=float)
    merged["deviation"] = np.linalg.norm(actual - intended, axis=1)
    merged["exceeds_limit"] = (
        ~np.isfinite(merged["deviation"])
        | (merged["deviation"] > MAX_POSITION_DEVIATION)
    )

    for _, drone_data in merged.groupby("drone", sort=False):
        drone_data = drone_data.sort_values("timestamp")
        exceeded = drone_data["exceeds_limit"].to_numpy(dtype=bool)
        timestamps = drone_data["timestamp"].to_numpy(dtype=float)
        start_time: float | None = None
        for index, is_exceeded in enumerate(exceeded):
            if is_exceeded and start_time is None:
                start_time = timestamps[index]
            elif not is_exceeded and start_time is not None:
                if timestamps[index] - start_time >= MAX_DEVIATION_DURATION:
                    return {
                        "result": "infeasible",
                        "run_file": Path(run_filename).name,
                        "trajectory_file": Path(trajectory_filename).name,
                        "reason": "A drone exceeded the deviation limit for too long.",
                        "drone": drone_data.iloc[0]["drone"],
                    }
                start_time = None
        if start_time is not None and timestamps[-1] - start_time >= MAX_DEVIATION_DURATION:
            return {
                "result": "infeasible",
                "run_file": Path(run_filename).name,
                "trajectory_file": Path(trajectory_filename).name,
                "reason": "A drone exceeded the deviation limit for too long.",
                "drone": drone_data.iloc[0]["drone"],
            }

    return {
        "result": "feasible",
        "run_file": Path(run_filename).name,
        "trajectory_file": Path(trajectory_filename).name,
    }


def main() -> None:
    """Run the checker from the command line."""
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python check_feasibility.py RUN_FILENAME TRAJECTORY_FILENAME"
        )
    result = check_feasibility(sys.argv[1], sys.argv[2])
    print(result)


if __name__ == "__main__":
    main()
