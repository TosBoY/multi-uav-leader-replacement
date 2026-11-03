"""Editable launcher for registered experiments."""

from pathlib import Path
import sys

try:
    from .experiment_manager import run_experiments
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.experiments_manager.experiment_manager import run_experiments


RUN_ID_RANGE = [1, 10]
NUMBER_OF_DRONE_RANGE = []
SHAPE_RANGE = []

"""
RUN_ID_RANGE = [1, 100]
NUMBER_OF_DRONE_RANGE = [5, 6]
SHAPE_RANGE = ["line_x", "sphere"]
"""


if __name__ == "__main__":
    run_experiments(
        run_id_range=RUN_ID_RANGE,
        number_of_drone_range=NUMBER_OF_DRONE_RANGE,
        shape_range=SHAPE_RANGE,
    )
