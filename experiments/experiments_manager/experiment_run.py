"""Editable launcher for registered experiments."""

from pathlib import Path
import sys
from time import perf_counter

try:
    from .experiment_manager import run_experiments
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.experiments_manager.experiment_manager import run_experiments


# RUN_ID_RANGE = [1, 10]
RUN_ID_RANGE = [1, 10000]
RUN_ID_BATCH_LIMIT = 2000
PATH_ID_RANGE = []
NUMBER_OF_DRONE_RANGE = []
SHAPE_RANGE = []
PLANE_RANGE = []
SPREAD_RANGE = []
NONFAIL_ONLY = False


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as days, hours, minutes, and seconds."""
    whole_seconds = int(seconds)
    days, remainder = divmod(whole_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, remaining_seconds = divmod(remainder, 60)

    if days:
        return f"{days}d {hours}h {minutes}m {remaining_seconds}s"
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{seconds:.2f}s"


def _split_run_id_ranges(
    run_id_range: list[int] | None,
    batch_limit: int,
) -> list[list[int] | None]:
    """Split an inclusive run-ID range into sequential bounded ranges."""
    if batch_limit < 1:
        raise ValueError("RUN_ID_BATCH_LIMIT must be positive.")
    if run_id_range is None or len(run_id_range) != 2:
        return [run_id_range]

    start, end = sorted(run_id_range)
    return [
        [batch_start, min(batch_start + batch_limit - 1, end)]
        for batch_start in range(start, end + 1, batch_limit)
    ]


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
    start_time = perf_counter()
    try:
        for run_id_range in _split_run_id_ranges(
            RUN_ID_RANGE,
            RUN_ID_BATCH_LIMIT,
        ):
            run_experiments(
                run_id_range=run_id_range,
                path_id_range=PATH_ID_RANGE,
                number_of_drone_range=NUMBER_OF_DRONE_RANGE,
                shape_range=SHAPE_RANGE,
                plane_range=PLANE_RANGE,
                spread_range=SPREAD_RANGE,
                nonfail_only=NONFAIL_ONLY,
            )
    finally:
        elapsed_time = perf_counter() - start_time
        print(
            f"Finished all experiment batches in "
            f"{_format_elapsed_time(elapsed_time)}."
        )
