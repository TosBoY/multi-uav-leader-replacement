"""Persist intended controller paths as CSV data."""

from collections.abc import Sequence
import csv
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import numpy as np


Drone = TypeVar("Drone")
_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "results" / "data" / "trajectories"
)


def print_path(
    intended_positions: np.ndarray,
    times: Sequence[float] | np.ndarray,
    drones: Sequence[Drone],
    output_path: str | Path | None = None,
) -> Path:
    """Save controller target positions for every drone and timestamp to CSV."""
    if intended_positions.shape != (len(times), len(drones), 3):
        raise ValueError("intended_positions must have shape (time, drone, 3).")

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = _DATA_DIRECTORY / f"simulation_path_{timestamp}.csv"
    else:
        destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "drone", "x", "y", "z"])
        for time_index, timestamp in enumerate(times):
            for drone_index, drone in enumerate(drones):
                position = intended_positions[time_index, drone_index]
                writer.writerow([
                    float(timestamp),
                    drone,
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ])
    return destination
