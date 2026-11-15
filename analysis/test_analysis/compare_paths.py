"""Compare two path Parquet files row by row.

Each row is identified by its ``timestamp`` and ``drone`` values. Rows with
missing keys or different positions are printed with their time and drone.

Usage:
    python compare_paths.py first.parquet second.parquet
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


_KEY_COLUMNS = ["timestamp", "drone"]
_POSITION_COLUMNS = ["x", "y", "z"]
_REQUIRED_COLUMNS = set(_KEY_COLUMNS + _POSITION_COLUMNS)
_RESULTS_DIRECTORY = Path(__file__).resolve().parents[2] / "results"


def _read_path(path: str | Path) -> pd.DataFrame:
    """Read one path file and validate its required columns."""
    frame = pd.read_parquet(path)
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.duplicated(_KEY_COLUMNS).any():
        raise ValueError(
            f"{path} contains duplicate timestamp/drone rows."
        )
    return frame[_KEY_COLUMNS + _POSITION_COLUMNS].copy()


def _positions_match(
    first: np.ndarray,
    second: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> np.ndarray:
    """Return whether each pair of position rows matches."""
    return np.asarray(
        np.isclose(
            first,
            second,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            equal_nan=True,
        ).all(axis=1),
        dtype=bool,
    )


def compare_paths(
    first_path: str | Path,
    second_path: str | Path,
    *,
    absolute_tolerance: float = 0.1,
    relative_tolerance: float = 0.0,
) -> int:
    """Compare two path files and print the time/drone of every mismatch.

    Returns:
        The number of rows that do not match.
    """
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Tolerances must not be negative.")

    first = _read_path(first_path).rename(
        columns={column: f"{column}_first" for column in _POSITION_COLUMNS}
    )
    second = _read_path(second_path).rename(
        columns={column: f"{column}_second" for column in _POSITION_COLUMNS}
    )
    merged = first.merge(
        second,
        on=_KEY_COLUMNS,
        how="outer",
        indicator=True,
        sort=True,
    )

    missing = merged["_merge"] != "both"
    mismatches = pd.Series(False, index=merged.index)
    matched = ~missing
    if matched.any():
        first_positions = merged.loc[
            matched, [f"{column}_first" for column in _POSITION_COLUMNS]
        ].to_numpy(dtype=float)
        second_positions = merged.loc[
            matched, [f"{column}_second" for column in _POSITION_COLUMNS]
        ].to_numpy(dtype=float)
        mismatches.loc[matched] = ~_positions_match(
            first_positions,
            second_positions,
            absolute_tolerance,
            relative_tolerance,
        )

    unmatched = merged[missing | mismatches]
    report_lines = [
        f"First file: {first_path}",
        f"Second file: {second_path}",
        f"Absolute tolerance: {absolute_tolerance}",
        f"Relative tolerance: {relative_tolerance}",
        "",
    ]
    for _, row in unmatched.iterrows():
        if row["_merge"] != "both":
            reason = (
                "missing from first file"
                if row["_merge"] == "right_only"
                else "missing from second file"
            )
            report_lines.append(
                f"time={row['timestamp']}, drone={row['drone']}: {reason}"
            )
        else:
            differences = [
                row[f"{column}_first"] - row[f"{column}_second"]
                for column in _POSITION_COLUMNS
            ]
            distance = np.linalg.norm(differences)
            report_lines.append(
                f"time={row['timestamp']}, drone={row['drone']}: "
                f"dx={differences[0]:.5f}, dy={differences[1]:.5f}, "
                f"dz={differences[2]:.5f}, distance={distance:.5f}"
            )

    if unmatched.empty:
        report_lines.append("All path rows match.")
    else:
        report_lines.append(f"{len(unmatched)} row(s) do not match.")

    report = "\n".join(report_lines)
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report_path = _RESULTS_DIRECTORY / (
        f"path_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
    )
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport saved to {report_path}")
    return len(unmatched)


def main() -> None:
    """Compare two Parquet files supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_file", nargs="?", help="First path Parquet file")
    parser.add_argument("second_file", nargs="?", help="Second path Parquet file")
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=0.1,
        help="Allowed absolute position difference (default: 0.1)",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.0,
        help="Allowed relative position difference (default: exact match)",
    )
    args = parser.parse_args()
    first_file = args.first_file or input("First path Parquet file: ").strip()
    second_file = args.second_file or input("Second path Parquet file: ").strip()
    compare_paths(
        first_file,
        second_file,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )


if __name__ == "__main__":
    main()
