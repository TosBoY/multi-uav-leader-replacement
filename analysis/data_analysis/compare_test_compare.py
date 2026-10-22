"""Measure how a leader failure moves each drone away from a no-failure run.

Usage:
    python analyse_leader_failure.py baseline.parquet failure.parquet
"""

from pathlib import Path
import argparse
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {"timestamp", "drone", "is_leader", "failed", "x", "y", "z"}
_POSITION_COLUMNS = ["x", "y", "z"]
# This file lives in project/analysis/, so the project root is two levels up.
# Anchoring to the file (not the working directory) means the report always
# lands in project/result/analysis no matter where the script is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "analysis"


def _read_simulation_data(path: str | Path) -> pd.DataFrame:
    """Read one saved simulation file and validate its columns."""
    frame = pd.read_parquet(path)
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    # Rounding stops tiny float differences from breaking the timestamp merge.
    frame["timestamp"] = frame["timestamp"].astype(float).round(9)
    return frame


def _find_leader_change(frame: pd.DataFrame) -> tuple[float, Any, Any]:
    """Return (time of first leader change, failed leader, promoted leader)."""
    leaders = (
        frame[frame["is_leader"]]
        .groupby("timestamp")["drone"]
        .first()
        .sort_index()
    )
    if leaders.empty:
        raise ValueError("No leader rows found in the failure file.")
    initial_leader = leaders.iloc[0]
    changed = leaders[leaders != initial_leader]
    if changed.empty:
        raise ValueError("The leader never changes in the failure file.")
    return float(changed.index[0]), initial_leader, changed.iloc[0]


def _position_errors(
    baseline: pd.DataFrame,
    failure: pd.DataFrame,
    failed_leader: Any | None = None,
    promoted_leader: Any | None = None,
    change_time: float | None = None,
) -> pd.DataFrame:
    """Per-drone position error, with the promoted leader using the old leader's baseline.

    All drones are matched by their own ID before the leader change. After
    the change, the promoted leader is compared with the failed leader's
    no-failure positions, which represents the route it would have inherited.
    """
    columns = ["timestamp", "drone", *_POSITION_COLUMNS]
    baseline = baseline[columns].copy()
    if (
        failed_leader is not None
        and promoted_leader is not None
        and change_time is not None
    ):
        old_leader_rows = baseline[baseline["drone"] == failed_leader].copy()
        promoted_baseline = old_leader_rows[
            old_leader_rows["timestamp"] >= change_time
        ].copy()
        promoted_baseline["drone"] = promoted_leader
        baseline = pd.concat(
            [
                baseline[
                    (baseline["drone"] != promoted_leader)
                    | (baseline["timestamp"] < change_time)
                ],
                promoted_baseline,
            ],
            ignore_index=True,
        )

    merged = baseline[columns].merge(
        failure[columns],
        on=["timestamp", "drone"],
        suffixes=("_base", "_fail"),
    )
    base = merged[[f"{c}_base" for c in _POSITION_COLUMNS]].to_numpy(dtype=float)
    fail = merged[[f"{c}_fail" for c in _POSITION_COLUMNS]].to_numpy(dtype=float)
    merged["error"] = np.linalg.norm(base - fail, axis=1)
    return merged[["timestamp", "drone", "error"]].sort_values(["drone", "timestamp"])


def _first_matched_index(
    within: np.ndarray,
    hold_steps: int,
    until_end: bool,
) -> int | None:
    """Index where the error first drops below tolerance AND stays there.

    A single sample under the tolerance is not enough. The error must stay under
    it for ``hold_steps`` consecutive samples, or for the whole rest of the run
    when ``until_end`` is true.
    """
    if within.size == 0:
        return None

    if until_end:
        # suffix_ok[i] is True only if within[i:] is entirely True.
        suffix_ok = np.logical_and.accumulate(within[::-1])[::-1]
        hits = np.flatnonzero(suffix_ok)
    else:
        if within.size < hold_steps:
            return None
        window_sums = np.convolve(
            within.astype(int),
            np.ones(hold_steps, dtype=int),
            mode="valid",
        )
        hits = np.flatnonzero(window_sums == hold_steps)

    return int(hits[0]) if hits.size else None


def _integrate(times: np.ndarray, errors: np.ndarray) -> float:
    """Trapezoid area under the error curve (metre-seconds)."""
    if len(times) < 2:
        return 0.0
    return float(np.nansum(0.5 * (errors[1:] + errors[:-1]) * np.diff(times)))


def _analyse_after_change(
    drone: Any,
    role: str,
    times: np.ndarray,
    errors: np.ndarray,
    change_time: float,
    tolerance: float,
    hold_steps: int,
    until_end: bool,
) -> dict[str, Any]:
    """Measure re-matching behaviour for one drone after the leader change."""
    row: dict[str, Any] = {
        "drone": drone,
        "role": role,
        "status": "",
        "time_to_match_s": np.nan,
        "mean_error_until_match": np.nan,
        "max_error_until_match": np.nan,
        "error_integral_until_match": np.nan,
        "final_error": np.nan,
        "max_error_after_match": np.nan,
        "excursions_after_match": np.nan,
    }

    finite = np.isfinite(errors)
    if not finite.any():
        row["status"] = "no data (failed)"
        return row

    within = finite & (errors <= tolerance)
    index = _first_matched_index(within, hold_steps, until_end)
    row["final_error"] = float(errors[-1])

    if index is None:
        # Never settled: describe the whole post-change window instead.
        row["status"] = "never matched"
        row["mean_error_until_match"] = float(np.nanmean(errors))
        row["max_error_until_match"] = float(np.nanmax(errors))
        row["error_integral_until_match"] = _integrate(times, errors)
        return row

    until_match = errors[: index + 1]
    after_match = within[index:]
    row["time_to_match_s"] = float(times[index] - change_time)
    row["mean_error_until_match"] = float(np.nanmean(until_match))
    row["max_error_until_match"] = float(np.nanmax(until_match))
    row["error_integral_until_match"] = _integrate(times[: index + 1], until_match)
    row["max_error_after_match"] = float(np.nanmax(errors[index:]))
    row["excursions_after_match"] = int(
        np.sum(after_match[:-1] & ~after_match[1:])
    )
    row["status"] = (
        "matched (stable)"
        if after_match.all()
        else "matched, later left tolerance"
    )
    return row


def analyse_leader_failure(
    baseline_path: str | Path,
    failure_path: str | Path,
    tolerance: float = 0.05,
    hold_time: float = 1.0,
    until_end: bool = False,
    save: bool = True,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Compare a no-failure run with a leader-failure run.

    Args:
        baseline_path: Run with no leader failure.
        failure_path: Run with a leader failure.
        tolerance: Position error (metres) counted as "matched".
        hold_time: Seconds the error must stay within tolerance to count as
            matched. This is what separates a real match from a fluke
            where the error only dips under the tolerance for a moment.
        until_end: If true, the error must stay within tolerance for the
            whole rest of the run instead of just ``hold_time`` seconds.
        save: If true (the default), also write the printed report to a text
            file.
        output_dir: Folder for the report (default ``<project>/result/analysis``,
            where ``<project>`` is the parent of the folder holding this file).
            It is created if it does not exist.
    """
    if tolerance <= 0 or hold_time <= 0:
        raise ValueError("tolerance and hold_time must be greater than zero.")

    baseline = _read_simulation_data(baseline_path)
    failure = _read_simulation_data(failure_path)
    change_time, failed_leader, promoted_leader = _find_leader_change(failure)
    errors = _position_errors(
        baseline,
        failure,
        failed_leader=failed_leader,
        promoted_leader=promoted_leader,
        change_time=change_time,
    )

    all_times = np.sort(errors["timestamp"].unique())
    if len(all_times) < 2:
        raise ValueError("Need at least two matched timestamps.")
    dt = float(np.median(np.diff(all_times)))
    hold_steps = max(1, int(round(hold_time / dt)))

    before_rows = []
    after_rows = []
    for drone, group in errors.groupby("drone", sort=True):
        times = group["timestamp"].to_numpy()
        values = group["error"].to_numpy()
        if drone == failed_leader:
            role = "failed leader"
        elif drone == promoted_leader:
            role = "promoted leader"
        else:
            role = "follower"

        before = values[times < change_time]
        before = before[np.isfinite(before)]
        before_rows.append(
            {
                "drone": drone,
                "role": role,
                "samples": len(before),
                "mean_error": float(before.mean()) if len(before) else np.nan,
                "max_error": float(before.max()) if len(before) else np.nan,
            }
        )

        after = times >= change_time
        after_rows.append(
            _analyse_after_change(
                drone,
                role,
                times[after],
                values[after],
                change_time,
                tolerance,
                hold_steps,
                until_end,
            )
        )

    before_change = pd.DataFrame(before_rows)
    after_change = pd.DataFrame(after_rows)

    criterion = (
        "stay within tolerance until the end of the run"
        if until_end
        else f"stay within tolerance for {hold_time:g} s ({hold_steps} samples)"
    )
    def _format(table: pd.DataFrame) -> str:
        return table.to_string(index=False, float_format=lambda v: f"{v:.4g}")

    report = "\n".join(
        [
            "LEADER FAILURE ANALYSIS",
            "=======================",
            f"Generated:             {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Baseline (no failure): {baseline_path}",
            f"Failure run:           {failure_path}",
            f"Leader change at t = {change_time:.2f} s: "
            f"UAV {failed_leader} failed, UAV {promoted_leader} promoted",
            f"Match = error <= {tolerance:g} m and must {criterion}",
            "",
            "BEFORE leader change (position difference between the two runs)",
            "---------------------------------------------------------------",
            _format(before_change),
            "",
            "AFTER leader change",
            "-------------------",
            _format(after_change),
        ]
    )
    print(report)

    report_path = None
    if save:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        report_path = folder / (
            f"{Path(failure_path).stem}_vs_{Path(baseline_path).stem}.txt"
        )
        report_path.write_text(report + "\n", encoding="utf-8")
        print(f"\nReport saved to {report_path}")

    return {
        "change_time": change_time,
        "failed_leader": failed_leader,
        "promoted_leader": promoted_leader,
        "before_change": before_change,
        "after_change": after_change,
        "report_path": report_path,
    }


def main() -> None:
    """Run the analysis from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_file", nargs="?", help="Parquet file without leader failure")
    parser.add_argument("failure_file", nargs="?", help="Parquet file with leader failure")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Position error in metres counted as matched (default 0.05)")
    parser.add_argument("--hold-time", type=float, default=1.0,
                        help="Seconds the error must stay within tolerance (default 1.0)")
    parser.add_argument("--until-end", action="store_true",
                        help="Require the match to hold for the rest of the run")
    parser.add_argument("--no-save", action="store_true",
                        help="Print the report without writing a text file")
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR),
                        help=f"Folder for the saved report (default {_DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    baseline_file = args.baseline_file or input("No-failure data file: ").strip()
    failure_file = args.failure_file or input("Leader-failure data file: ").strip()
    analyse_leader_failure(
        baseline_file,
        failure_file,
        tolerance=args.tolerance,
        hold_time=args.hold_time,
        until_end=args.until_end,
        save=not args.no_save,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()