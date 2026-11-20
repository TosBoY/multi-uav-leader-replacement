"""Run raw-data analysis steps in dependency order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_MODULES = (
    "analysis.raw_data_analysis.trajectory_difficulty_calculate",
    "analysis.raw_data_analysis.metrics_calculate",
)


def main() -> None:
    """Calculate trajectory difficulty before run-level metrics."""
    for module in ANALYSIS_MODULES:
        print(f"\nRunning {module}...", flush=True)
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print("\nAll raw-data analyses completed successfully.")


if __name__ == "__main__":
    main()
