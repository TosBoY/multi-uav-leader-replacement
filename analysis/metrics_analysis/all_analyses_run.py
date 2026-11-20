"""Run all metrics analyses sequentially."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "analysis"
ANALYSIS_MODULES = (
    "analysis.metrics_analysis.mission_success_analysis",
    "analysis.metrics_analysis.recovery_performance_analysis",
    "analysis.metrics_analysis.robustness_analysis",
    "analysis.metrics_analysis.safety_performance_analysis",
    "analysis.metrics_analysis.trajectory_deviation_analysis",
)


def main() -> None:
    """Run each metrics-analysis module in a fixed order."""
    if ANALYSIS_OUTPUT_DIRECTORY.exists():
        print(
            f"Removing previous analysis results from "
            f"{ANALYSIS_OUTPUT_DIRECTORY}...",
            flush=True,
        )
        shutil.rmtree(ANALYSIS_OUTPUT_DIRECTORY)
    ANALYSIS_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for module in ANALYSIS_MODULES:
        print(f"\nRunning {module}...", flush=True)
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print("\nAll metrics analyses completed successfully.")


if __name__ == "__main__":
    main()
