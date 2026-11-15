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
    velocities = results["velocities"]
    angular_velocities = results["angular_velocities"]
    quaternions = results["quaternions"]
    travel_distance_history = results["travel_distances"]
    leader_history = results["leader_history"]
    failed_history = results["failed_history"]

    time_values = np.asarray(times, dtype=float)
    drone_values = np.asarray(drones)
    timestamps = np.repeat(time_values, len(drone_values))
    repeated_leaders = np.repeat(np.asarray(leader_history), len(drone_values))
    tiled_drones = np.tile(drone_values, len(time_values))
    flat_positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    flat_velocities = np.asarray(velocities, dtype=float).reshape(-1, 3)
    flat_angular_velocities = np.asarray(
        angular_velocities,
        dtype=float,
    ).reshape(-1, 3)
    flat_quaternions = np.asarray(quaternions, dtype=float).reshape(-1, 4)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "drone": tiled_drones,
            "is_leader": tiled_drones == repeated_leaders,
            "failed": np.asarray(failed_history, dtype=bool).reshape(-1),
            "x": flat_positions[:, 0],
            "y": flat_positions[:, 1],
            "z": flat_positions[:, 2],
            "v_x": flat_velocities[:, 0],
            "v_y": flat_velocities[:, 1],
            "v_z": flat_velocities[:, 2],
            "w_x": flat_angular_velocities[:, 0],
            "w_y": flat_angular_velocities[:, 1],
            "w_z": flat_angular_velocities[:, 2],
            "q_x": flat_quaternions[:, 0],
            "q_y": flat_quaternions[:, 1],
            "q_z": flat_quaternions[:, 2],
            "q_w": flat_quaternions[:, 3],
            "travel_distance": np.asarray(
                travel_distance_history, dtype=float
            ).reshape(-1),
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
