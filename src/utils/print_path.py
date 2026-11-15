"""Persist intended controller paths and leader trajectory data as Parquet."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd


Drone = TypeVar("Drone")
_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "results" / "data" / "trajectories"
)


def print_path(
    intended_positions: np.ndarray,
    times: Sequence[float] | np.ndarray,
    drones: Sequence[Drone],
    output_path: str | Path | None = None,
    output_filename: str | Path | None = None,
    leader_path_history: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Save controller targets and leader trajectory data to Parquet."""
    if intended_positions.shape != (len(times), len(drones), 3):
        raise ValueError("intended_positions must have shape (time, drone, 3).")
    if leader_path_history is not None and len(leader_path_history) != len(times):
        raise ValueError("leader_path_history must contain one entry per timestamp.")
    if leader_path_history is not None and not leader_path_history:
        raise ValueError("leader_path_history must not be empty.")

    if output_path is not None and output_filename is not None:
        raise ValueError("Specify either output_path or output_filename, not both.")
    if output_filename is not None:
        destination = _DATA_DIRECTORY / Path(output_filename)
    elif output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = _DATA_DIRECTORY / f"simulation_path_{timestamp}.parquet"
    else:
        destination = Path(output_path)
    if destination.suffix.lower() != ".parquet":
        destination = destination.with_suffix(".parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)

    time_values = np.asarray(times, dtype=float)
    drone_values = np.asarray(drones)
    flat_positions = np.asarray(intended_positions, dtype=float).reshape(-1, 3)
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(time_values, len(drone_values)),
            "drone": np.tile(drone_values, len(time_values)),
            "x": flat_positions[:, 0],
            "y": flat_positions[:, 1],
            "z": flat_positions[:, 2],
        }
    )
    if leader_path_history is not None:
        trajectory_fields = set(leader_path_history[0])
        if not trajectory_fields:
            raise ValueError("leader_path_history entries must not be empty.")
        if any(set(entry) != trajectory_fields for entry in leader_path_history):
            raise ValueError("leader_path_history entries must have the same fields.")

        for field in sorted(trajectory_fields):
            if field == "x":
                continue
            values = np.asarray(
                [entry[field] for entry in leader_path_history],
                dtype=float,
            )
            if values.ndim == 1:
                frame[f"leader_path_{field}"] = np.repeat(
                    values,
                    len(drone_values),
                )
            elif values.ndim == 2 and values.shape[1] == 3:
                repeated_values = np.repeat(values, len(drone_values), axis=0)
                for index, axis in enumerate(("x", "y", "z")):
                    frame[f"leader_path_{field}_{axis}"] = repeated_values[:, index]
            else:
                raise ValueError(
                    f"leader_path_history field {field!r} must be a scalar or 3D vector."
                )
    frame.to_parquet(destination, index=False)
    return destination
