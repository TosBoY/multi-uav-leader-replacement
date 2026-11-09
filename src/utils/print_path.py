"""Persist intended controller paths as CSV data."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar

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
) -> Path:
    """Save controller target positions for every drone and timestamp to Parquet."""
    if intended_positions.shape != (len(times), len(drones), 3):
        raise ValueError("intended_positions must have shape (time, drone, 3).")

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
    frame.to_parquet(destination, index=False)
    return destination
