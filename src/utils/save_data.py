"""Persist simulation histories as Parquet data."""

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


_DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "results" / "data"


def save_simulation_data(
    results: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Save one row per drone and timestamp to a Parquet file.

    Args:
        results: Results dictionary returned by ``run_simulation``.
        output_path: Optional output file. If omitted, a timestamped file is
            created in ``results/data`` in the project root.

    Returns:
        The path of the written Parquet file.
    """
    drones = results["drones"]
    times = results["time"]
    positions = results["positions"]
    rotations = results["rotations"]
    wind = results["wind"]
    leader_history = results["leader_history"]
    failed_history = results["failed_history"]
    promotion_time = results["promotion_time"]

    rows: list[dict[str, Any]] = []
    for time_index, timestamp in enumerate(times):
        current_leader = leader_history[time_index]
        for drone_index, drone in enumerate(drones):
            position = positions[time_index, drone_index]
            rotation = rotations[time_index, drone_index]
            wind_vector = wind[time_index, drone_index]
            rows.append(
                {
                    "timestamp": float(timestamp),
                    "drone": drone,
                    "is_leader": drone == current_leader,
                    "failed": bool(failed_history[time_index, drone_index]),
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                    "rotation_00": float(rotation[0, 0]),
                    "rotation_01": float(rotation[0, 1]),
                    "rotation_02": float(rotation[0, 2]),
                    "rotation_10": float(rotation[1, 0]),
                    "rotation_11": float(rotation[1, 1]),
                    "rotation_12": float(rotation[1, 2]),
                    "rotation_20": float(rotation[2, 0]),
                    "rotation_21": float(rotation[2, 1]),
                    "rotation_22": float(rotation[2, 2]),
                    "wind_x": float(wind_vector[0]),
                    "wind_y": float(wind_vector[1]),
                    "wind_z": float(wind_vector[2]),
                    "promotion_time": promotion_time,
                }
            )

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = _DATA_DIRECTORY / f"simulation_data_{timestamp}.parquet"
    else:
        destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(destination, index=False)
    return destination
