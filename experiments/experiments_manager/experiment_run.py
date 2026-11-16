"""Editable launcher for registered experiments."""

from pathlib import Path
import sys

try:
    from .experiment_manager import run_experiments
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.experiments_manager.experiment_manager import run_experiments


# RUN_ID_RANGE = [1, 10]
RUN_ID_RANGE = [11, 20]
PATH_ID_RANGE = []
NUMBER_OF_DRONE_RANGE = []
SHAPE_RANGE = []
PLANE_RANGE = []
SPREAD_RANGE = []
NONFAIL_ONLY = False

"""
RUN_ID_RANGE = [1, 200]
PATH_ID_RANGE = [1, 10]
NUMBER_OF_DRONE_RANGE = [5, 6]
SHAPE_RANGE = ["line_x", "sphere"]
PLANE_RANGE = ["xy"]
SPREAD_RANGE = [1.0, 2.0]
NONFAIL_ONLY = False
"""


if __name__ == "__main__":
    run_experiments(
        run_id_range=RUN_ID_RANGE,
        path_id_range=PATH_ID_RANGE,
        number_of_drone_range=NUMBER_OF_DRONE_RANGE,
        shape_range=SHAPE_RANGE,
        plane_range=PLANE_RANGE,
        spread_range=SPREAD_RANGE,
        nonfail_only=NONFAIL_ONLY,
    )
