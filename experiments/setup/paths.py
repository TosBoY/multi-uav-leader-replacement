"""Generate and edit the experiment paths setup table."""

from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path

try:
    from experiments.setup.common import (
        DATABASE_PATH,
        connect_database,
        validate_choice,
    )
except ModuleNotFoundError:
    from common import (
        DATABASE_PATH,
        connect_database,
        validate_choice,
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
PATH_TYPES = ("line", "random")

# Values to combine for each column. Edit these (or pass overrides to
# generate_paths) to widen the grid. Every column must have at least one value.
DEFAULT_PATH_VALUES: dict[str, Sequence[object]] = {
    "first_waypoint_x": (0.0,),
    "first_waypoint_y": (0.0,),
    "first_waypoint_z": (1.0,),
    "velocity": (1.0,),
    "time": (10.0,),
    "seed": tuple(range(1, 101)), # ----------> Adjust number of seed here (max + 1) <----------
    "initial_heading": (0.0,),
}
PATHS_TABLE = "registry.paths"


def _validated_values(
    path_values: Mapping[str, Sequence[object]] | None,
) -> dict[str, Sequence[object]]:
    """Merge overrides into the defaults and check every value list."""
    values = dict(DEFAULT_PATH_VALUES)
    if path_values:
        unknown = set(path_values) - set(values)
        if unknown:
            raise ValueError(
                f"Unknown or non-editable columns: {sorted(unknown)}. "
                f"Valid columns are {sorted(values)}."
            )
        values.update(path_values)

    for column, options in values.items():
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
            raise TypeError(
                f"Values for {column!r} must be a list or tuple, "
                f"e.g. ({options!r},)."
            )
        if len(options) == 0:
            raise ValueError(f"Values for {column!r} must not be empty.")
    return values


def generate_paths(
    path_values: Mapping[str, Sequence[object]] | None = None,
    path_types: Sequence[str] = PATH_TYPES,
) -> list[dict[str, object]]:
    """Generate one row for every combination of the column values.

    Path types are generated in the requested order (``line`` first by
    default). Within each type, columns are ordered by importance from left to
    right: the leftmost column changes slowest and the rightmost changes
    fastest. Line paths have no seed, so their seed field stays blank and the
    seed values are not multiplied into the line rows.

    Args:
        path_values: Optional overrides for ``DEFAULT_PATH_VALUES``, mapping a
            column name to the values to combine, e.g.
            ``{"velocity": (0.5, 1.0), "seed": (1, 2, 3)}``.
        path_types: Path types to generate, in order.
    """
    if not path_types:
        raise ValueError("path_types must not be empty.")
    if len(set(path_types)) != len(path_types):
        raise ValueError("path_types must not contain duplicates.")
    for path_type in path_types:
        validate_choice(path_type, PATH_TYPES, "path_type")

    values = _validated_values(path_values)
    value_columns = PATH_COLUMNS[2:]  # everything except path_id and path_type

    rows: list[dict[str, object]] = []
    for path_type in path_types:
        columns = [
            column
            for column in value_columns
            if not (path_type == "line" and column == "seed")
        ]
        for combination in product(*(values[column] for column in columns)):
            row: dict[str, object] = {
                "path_id": len(rows) + 1,
                "path_type": path_type,
            }
            row.update(zip(columns, combination))
            row.setdefault("seed", "")
            rows.append(row)
    return rows


def make_paths_table(
    output_path: str | Path = DATABASE_PATH,
    path_values: Mapping[str, Sequence[object]] | None = None,
    path_types: Sequence[str] = PATH_TYPES,
) -> Path:
    """Generate and replace the paths registry table."""
    rows = generate_paths(path_values, path_types)
    with connect_database() as connection:
        connection.execute("DROP TABLE IF EXISTS registry.paths")
        connection.execute(
            """
            CREATE TABLE registry.paths (
                path_id BIGINT,
                path_type VARCHAR,
                first_waypoint_x DOUBLE,
                first_waypoint_y DOUBLE,
                first_waypoint_z DOUBLE,
                velocity DOUBLE,
                time DOUBLE,
                seed BIGINT,
                initial_heading DOUBLE
            )
            """
        )
        connection.executemany(
            "INSERT INTO registry.paths VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row[column] if row[column] != "" else None for column in PATH_COLUMNS)
             for row in rows],
        )
    return Path(output_path)


def edit_path(
    path_id: int,
    output_path: str | Path = DATABASE_PATH,
    **updates: object,
) -> Path:
    """Edit one path row in the paths registry table.

    Editable fields are ``path_type``, ``first_waypoint_x``,
    ``first_waypoint_y``, ``first_waypoint_z``, ``velocity``, ``time``,
    ``seed``, and ``initial_heading``. The path ID cannot be changed.
    """
    if not isinstance(path_id, int) or isinstance(path_id, bool):
        raise TypeError("path_id must be an integer.")
    if any(field not in PATH_COLUMNS[1:] for field in updates):
        raise ValueError("Updates contain an unknown or non-editable field.")
    if "path_type" in updates:
        validate_choice(str(updates["path_type"]), PATH_TYPES, "path_type")

    with connect_database() as connection:
        if connection.execute(
            "SELECT 1 FROM registry.paths WHERE path_id = ?", [path_id]
        ).fetchone() is None:
            raise ValueError(f"No path with path_id={path_id} exists.")
        for field, value in updates.items():
            connection.execute(
                f'UPDATE registry.paths SET "{field}" = ? WHERE path_id = ?',
                [value, path_id],
            )
    return Path(output_path)


if __name__ == "__main__":
    make_paths_table()


make_paths_csv = make_paths_table