"""Persist simulation histories as Parquet data."""

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "results" / "data"


def save_simulation_data(
    results: Mapping[str, Any],
    output_path: str | Path | None = None,
    output_filename: str | Path | None = None,
) -> Path:
    """Save one row per drone and timestamp to a Parquet file.

    Args:
        results: Results dictionary returned by ``run_simulation``.
        output_path: Optional output file. If omitted, a timestamped file is
            created in ``results/data`` in the project root.
        output_filename: Optional filename inside ``results/data``. If omitted,
            a timestamped filename is used.

    Returns:
        The path of the written Parquet file.
    """
    if output_path is not None and output_filename is not None:
        raise ValueError("Specify either output_path or output_filename, not both.")

    drones = results["drones"]
    times = results["time"]
    positions = results["positions"]
    rotations = results["rotations"]
    wind = results["wind"]
    travel_distance_history = results["travel_distances"]
    leader_history = results["leader_history"]
    failed_history = results["failed_history"]
    promotion_time = results["promotion_time"]

    time_values = np.asarray(times, dtype=float)
    drone_values = np.asarray(drones)
    timestamps = np.repeat(time_values, len(drone_values))
    repeated_leaders = np.repeat(np.asarray(leader_history), len(drone_values))
    tiled_drones = np.tile(drone_values, len(time_values))
    flat_positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    flat_rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    flat_wind = np.asarray(wind, dtype=float).reshape(-1, 3)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "drone": tiled_drones,
            "is_leader": tiled_drones == repeated_leaders,
            "failed": np.asarray(failed_history, dtype=bool).reshape(-1),
            "x": flat_positions[:, 0],
            "y": flat_positions[:, 1],
            "z": flat_positions[:, 2],
            "rotation_00": flat_rotations[:, 0, 0],
            "rotation_01": flat_rotations[:, 0, 1],
            "rotation_02": flat_rotations[:, 0, 2],
            "rotation_10": flat_rotations[:, 1, 0],
            "rotation_11": flat_rotations[:, 1, 1],
            "rotation_12": flat_rotations[:, 1, 2],
            "rotation_20": flat_rotations[:, 2, 0],
            "rotation_21": flat_rotations[:, 2, 1],
            "rotation_22": flat_rotations[:, 2, 2],
            "wind_x": flat_wind[:, 0],
            "wind_y": flat_wind[:, 1],
            "wind_z": flat_wind[:, 2],
            "travel_distance": np.asarray(
                travel_distance_history, dtype=float
            ).reshape(-1),
            "promotion_time": promotion_time,
        }
    )

    if output_filename is not None:
        destination = _DATA_DIRECTORY / Path(output_filename)
    elif output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = _DATA_DIRECTORY / f"simulation_data_{timestamp}.parquet"
    else:
        destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination
