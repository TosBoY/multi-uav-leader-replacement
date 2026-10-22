"""Generate and edit the experiment formations setup table."""

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


FORMATION_COLUMNS = [
    "formation_id",
    "number_of_drone",
    "shape",
    "spread",
    "dimension",
    "plane",
    "leader_index",
    "leader_position_x",
    "leader_position_y",
    "leader_position_z",
]
FORMATION_SHAPES = (
    "line_x",
    "line_y",
    "line_z",
    "arrow",
    "circle",
    "grid",
)
FORMATION_PLANES = ("xy", "xz", "yz")
DEFAULT_NUMBER_OF_DRONES = tuple(range(5, 9))
DEFAULT_SPREADS = (1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_LEADER_INDEX = 0
DEFAULT_LEADER_POSITION = (0.0, 0.0, 1.0)
DEFAULT_FORMATIONS_FILE = setup_path("formations.csv")


def generate_formations(
    number_of_drones: Sequence[int] = DEFAULT_NUMBER_OF_DRONES,
    shapes: Sequence[str] = FORMATION_SHAPES,
    spreads: Sequence[float] = DEFAULT_SPREADS,
    planes: Sequence[str] = FORMATION_PLANES,
) -> list[dict[str, object]]:
    """Generate every requested valid formation configuration."""
    if not number_of_drones:
        raise ValueError("number_of_drones must not be empty.")
    if not shapes:
        raise ValueError("shapes must not be empty.")
    if not spreads:
        raise ValueError("spreads must not be empty.")
    if not planes:
        raise ValueError("planes must not be empty.")

    for count in number_of_drones:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("number_of_drones must contain positive integers.")
    for shape in shapes:
        validate_choice(shape, FORMATION_SHAPES, "shape")
    for plane in planes:
        validate_choice(plane, FORMATION_PLANES, "plane")
    for spread in spreads:
        if spread <= 0 or spread not in DEFAULT_SPREADS:
            raise ValueError("spreads must use values from 1.0 to 3.0 in 0.5 steps.")

    rows: list[dict[str, object]] = []
    formation_id = 1
    for count in number_of_drones:
        for shape in shapes:
            dimensions = (2, 3) if shape == "circle" else (2,)
            for spread in spreads:
                for dimension in dimensions:
                    for plane in planes:
                        rows.append(
                            {
                                "formation_id": formation_id,
                                "number_of_drone": count,
                                "shape": shape,
                                "spread": spread,
                                "dimension": dimension,
                                "plane": plane,
                                "leader_index": DEFAULT_LEADER_INDEX,
                                "leader_position_x": DEFAULT_LEADER_POSITION[0],
                                "leader_position_y": DEFAULT_LEADER_POSITION[1],
                                "leader_position_z": DEFAULT_LEADER_POSITION[2],
                            }
                        )
                        formation_id += 1
    return rows


def make_formations_csv(
    output_path: str | Path = DEFAULT_FORMATIONS_FILE,
    number_of_drones: Sequence[int] = DEFAULT_NUMBER_OF_DRONES,
    shapes: Sequence[str] = FORMATION_SHAPES,
    spreads: Sequence[float] = DEFAULT_SPREADS,
    planes: Sequence[str] = FORMATION_PLANES,
) -> Path:
    """Generate and write the formations setup CSV."""
    return write_csv_rows(
        output_path,
        FORMATION_COLUMNS,
        generate_formations(number_of_drones, shapes, spreads, planes),
    )


def edit_formation(
    formation_id: int,
    output_path: str | Path = DEFAULT_FORMATIONS_FILE,
    **updates: object,
) -> Path:
    """Edit one formation row and rewrite the formations CSV."""
    fieldnames, rows = read_csv_rows(output_path)
    validate_columns(fieldnames, FORMATION_COLUMNS)
    if not isinstance(formation_id, int) or isinstance(formation_id, bool):
        raise TypeError("formation_id must be an integer.")
    if any(field not in FORMATION_COLUMNS[1:] for field in updates):
        raise ValueError("Updates contain an unknown or non-editable field.")
    if "shape" in updates:
        validate_choice(str(updates["shape"]), FORMATION_SHAPES, "shape")
    if "plane" in updates:
        validate_choice(str(updates["plane"]), FORMATION_PLANES, "plane")

    for row in rows:
        if int(row["formation_id"]) == formation_id:
            row.update({field: str(value) for field, value in updates.items()})
            break
    else:
        raise ValueError(f"No formation with formation_id={formation_id} exists.")

    return write_csv_rows(output_path, FORMATION_COLUMNS, rows)


if __name__ == "__main__":
    make_formations_csv()
