"""Compare two saved simulation Parquet files."""

from pathlib import Path
import argparse
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "timestamp",
    "drone",
    "is_leader",
    "failed",
    "x",
    "y",
    "z",
    "wind_x",
    "wind_y",
    "wind_z",
}


def _read_simulation_data(path: str | Path) -> pd.DataFrame:
    """Read and validate one saved simulation file."""
    frame = pd.read_parquet(path)
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _leader_changes(frame: pd.DataFrame) -> int:
    """Count leader changes in one simulation frame."""
    leaders = (
        frame[frame["is_leader"]]
        .sort_values("timestamp")
        .groupby("timestamp", as_index=True)["drone"]
        .first()
    )
    return int((leaders != leaders.shift()).iloc[1:].sum())


def compare_simulation_data(
    first_path: str | Path,
    second_path: str | Path,
) -> dict[str, Any]:
    """Compare two simulation data files and print a summary.

    Rows are matched by ``timestamp`` and ``drone``. Position and wind errors
    are calculated only for matched rows with finite values in both files.
    """
    first = _read_simulation_data(first_path)
    second = _read_simulation_data(second_path)
    keys = ["timestamp", "drone"]
    merged = first.merge(
        second,
        on=keys,
        how="outer",
        suffixes=("_first", "_second"),
        indicator=True,
    )
    matched = merged[merged["_merge"] == "both"]
    position_columns = ["x", "y", "z"]
    wind_columns = ["wind_x", "wind_y", "wind_z"]

    position_first = matched[[f"{column}_first" for column in position_columns]].to_numpy()
    position_second = matched[[f"{column}_second" for column in position_columns]].to_numpy()
    wind_first = matched[[f"{column}_first" for column in wind_columns]].to_numpy()
    wind_second = matched[[f"{column}_second" for column in wind_columns]].to_numpy()

    position_valid = np.isfinite(position_first).all(axis=1) & np.isfinite(position_second).all(axis=1)
    wind_valid = np.isfinite(wind_first).all(axis=1) & np.isfinite(wind_second).all(axis=1)
    position_errors = np.linalg.norm(
        position_first[position_valid] - position_second[position_valid],
        axis=1,
    )
    wind_errors = np.linalg.norm(
        wind_first[wind_valid] - wind_second[wind_valid],
        axis=1,
    )

    summary: dict[str, Any] = {
        "first_file": str(first_path),
        "second_file": str(second_path),
        "first_rows": len(first),
        "second_rows": len(second),
        "matched_rows": len(matched),
        "only_in_first_rows": int((merged["_merge"] == "left_only").sum()),
        "only_in_second_rows": int((merged["_merge"] == "right_only").sum()),
        "position_comparison_rows": len(position_errors),
        "mean_position_error": float(position_errors.mean()) if len(position_errors) else None,
        "max_position_error": float(position_errors.max()) if len(position_errors) else None,
        "wind_comparison_rows": len(wind_errors),
        "mean_wind_error": float(wind_errors.mean()) if len(wind_errors) else None,
        "max_wind_error": float(wind_errors.max()) if len(wind_errors) else None,
        "first_leader_changes": _leader_changes(first),
        "second_leader_changes": _leader_changes(second),
        "first_failed_rows": int(first["failed"].sum()),
        "second_failed_rows": int(second["failed"].sum()),
    }

    print(f"First file: {summary['first_file']}")
    print(f"Second file: {summary['second_file']}")
    print(f"Matched rows: {summary['matched_rows']}")
    print(f"Only in first: {summary['only_in_first_rows']}")
    print(f"Only in second: {summary['only_in_second_rows']}")
    print(f"Mean position error: {summary['mean_position_error']}")
    print(f"Maximum position error: {summary['max_position_error']}")
    print(f"Mean wind error: {summary['mean_wind_error']}")
    print(f"Maximum wind error: {summary['max_wind_error']}")
    print(f"Leader changes: {summary['first_leader_changes']} -> {summary['second_leader_changes']}")
    print(f"Failed rows: {summary['first_failed_rows']} -> {summary['second_failed_rows']}")
    return summary


def main() -> None:
    """Compare two files provided as arguments or entered interactively."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_file", nargs="?", help="First Parquet file")
    parser.add_argument("second_file", nargs="?", help="Second Parquet file")
    args = parser.parse_args()
    first_file = args.first_file or input("First simulation data file: ").strip()
    second_file = args.second_file or input("Second simulation data file: ").strip()
    compare_simulation_data(first_file, second_file)


if __name__ == "__main__":
    main()
