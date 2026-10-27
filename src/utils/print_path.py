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

    rows = []
    for time_index, timestamp in enumerate(times):
        for drone_index, drone in enumerate(drones):
            position = intended_positions[time_index, drone_index]
            rows.append(
                {
                    "timestamp": float(timestamp),
                    "drone": drone,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                }
            )
    pd.DataFrame(rows).to_parquet(destination, index=False)
    return destination
