"""Generate and edit the experiment paths setup table."""

from collections.abc import Sequence
from pathlib import Path

try:
    from experiments.setup.common import (
        read_csv_rows,
        setup_path,
        validate_choice,
        validate_columns,
        write_csv_rows,
    )
except ModuleNotFoundError:
    from common import (
        read_csv_rows,
        setup_path,
        validate_choice,
        validate_columns,
        write_csv_rows,
    )


PATH_COLUMNS = [
    "path_id",
    "path_type",
    "first_waypoint_x",
    "first_waypoint_y",
    "first_waypoint_z",
    "velocity",
    "time",
    "seed",
    "initial_heading",
]
PATH_TYPES = ("random", "line")
DEFAULT_PATH_COUNT = 100
DEFAULT_SEED_START = 1
DEFAULT_FIRST_WAYPOINT = (0.0, 0.0, 1.0)
DEFAULT_VELOCITY = 1.0
DEFAULT_TIME = 10.0
DEFAULT_INITIAL_HEADING = 0.0
DEFAULT_PATHS_FILE = setup_path("paths.csv")


def generate_paths(
    number_of_paths: int = DEFAULT_PATH_COUNT,
    seed_start: int = DEFAULT_SEED_START,
    path_types: Sequence[str] = PATH_TYPES,
) -> list[dict[str, object]]:
    """Generate path setup rows with deterministic default values.

    Path types are assigned in order, so the default alternates ``random`` and
    ``line`` while seeds increase consecutively.
    """
    if isinstance(number_of_paths, bool) or number_of_paths < 1:
        raise ValueError("number_of_paths must be a positive integer.")
    if isinstance(seed_start, bool) or seed_start < 1:
        raise ValueError("seed_start must be a positive integer.")
    if not path_types:
        raise ValueError("path_types must not be empty.")
    for path_type in path_types:
        validate_choice(path_type, PATH_TYPES, "path_type")

    rows: list[dict[str, object]] = []
    for index in range(number_of_paths):
        rows.append(
            {
                "path_id": index + 1,
                "path_type": path_types[index % len(path_types)],
                "first_waypoint_x": DEFAULT_FIRST_WAYPOINT[0],
                "first_waypoint_y": DEFAULT_FIRST_WAYPOINT[1],
                "first_waypoint_z": DEFAULT_FIRST_WAYPOINT[2],
                "velocity": DEFAULT_VELOCITY,
                "time": DEFAULT_TIME,
                "seed": seed_start + index,
                "initial_heading": DEFAULT_INITIAL_HEADING,
            }
        )
    return rows


def make_paths_csv(
    output_path: str | Path = DEFAULT_PATHS_FILE,
    number_of_paths: int = DEFAULT_PATH_COUNT,
    seed_start: int = DEFAULT_SEED_START,
    path_types: Sequence[str] = PATH_TYPES,
) -> Path:
    """Generate and write the paths setup CSV."""
    return write_csv_rows(
        output_path,
        PATH_COLUMNS,
        generate_paths(number_of_paths, seed_start, path_types),
    )


def edit_path(
    path_id: int,
    output_path: str | Path = DEFAULT_PATHS_FILE,
    **updates: object,
) -> Path:
    """Edit one path row and rewrite the paths CSV.

    Editable fields are ``path_type``, ``first_waypoint_x``,
    ``first_waypoint_y``, ``first_waypoint_z``, ``velocity``, ``time``,
    ``seed``, and ``initial_heading``. The path ID cannot be changed.
    """
    fieldnames, rows = read_csv_rows(output_path)
    validate_columns(fieldnames, PATH_COLUMNS)
    if not isinstance(path_id, int) or isinstance(path_id, bool):
        raise TypeError("path_id must be an integer.")
    if any(field not in PATH_COLUMNS[1:] for field in updates):
        raise ValueError("Updates contain an unknown or non-editable field.")
    if "path_type" in updates:
        validate_choice(str(updates["path_type"]), PATH_TYPES, "path_type")

    for row in rows:
        if int(row["path_id"]) == path_id:
            row.update({field: str(value) for field, value in updates.items()})
            break
    else:
        raise ValueError(f"No path with path_id={path_id} exists.")

    return write_csv_rows(output_path, PATH_COLUMNS, rows)


if __name__ == "__main__":
    make_paths_csv()
