"""Generate and edit the experiment formations setup table."""

from collections.abc import Sequence
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


FORMATION_COLUMNS = [
    "formation_id",
    "number_of_drone",
    "shape",
    "spread",
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
    "sphere",
    "grid",
)
FORMATION_SHAPES_WITHOUT_PLANE = ("line_x", "line_y", "line_z", "sphere")
FORMATION_PLANES = ("xy", "xz", "yz")
DEFAULT_NUMBER_OF_DRONES = tuple(range(5, 9))
DEFAULT_SPREADS = (1.0, 2.0, 3.0)
DEFAULT_LEADER_INDEX = 0
DEFAULT_LEADER_POSITION = (0.0, 0.0, 1.0)
FORMATIONS_TABLE = "registry.formations"


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
        if isinstance(spread, bool) or not isinstance(spread, (int, float)):
            raise TypeError("spreads must contain numbers.")
        if spread <= 0:
            raise ValueError("spreads must contain positive values.")

    rows: list[dict[str, object]] = []
    formation_id = 1
    for count in number_of_drones:
        for shape in shapes:
            for spread in spreads:
                formation_planes = (
                    (None,) if shape in FORMATION_SHAPES_WITHOUT_PLANE else planes
                )
                for plane in formation_planes:
                    rows.append(
                        {
                            "formation_id": formation_id,
                            "number_of_drone": count,
                            "shape": shape,
                            "spread": spread,
                            "plane": plane,
                            "leader_index": DEFAULT_LEADER_INDEX,
                            "leader_position_x": DEFAULT_LEADER_POSITION[0],
                            "leader_position_y": DEFAULT_LEADER_POSITION[1],
                            "leader_position_z": DEFAULT_LEADER_POSITION[2],
                        }
                    )
                    formation_id += 1
    return rows


def make_formations_table(
    output_path: str | Path = DATABASE_PATH,
    number_of_drones: Sequence[int] = DEFAULT_NUMBER_OF_DRONES,
    shapes: Sequence[str] = FORMATION_SHAPES,
    spreads: Sequence[float] = DEFAULT_SPREADS,
    planes: Sequence[str] = FORMATION_PLANES,
) -> Path:
    """Generate and replace the formations registry table."""
    rows = generate_formations(number_of_drones, shapes, spreads, planes)
    with connect_database() as connection:
        connection.execute("DROP TABLE IF EXISTS registry.formations")
        connection.execute(
            """
            CREATE TABLE registry.formations (
                formation_id BIGINT,
                number_of_drone BIGINT,
                shape VARCHAR,
                spread DOUBLE,
                plane VARCHAR,
                leader_index BIGINT,
                leader_position_x DOUBLE,
                leader_position_y DOUBLE,
                leader_position_z DOUBLE
            )
            """
        )
        connection.executemany(
            "INSERT INTO registry.formations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row[column] if row[column] != "" else None for column in FORMATION_COLUMNS)
             for row in rows],
        )
    return Path(output_path)


def edit_formation(
    formation_id: int,
    output_path: str | Path = DATABASE_PATH,
    **updates: object,
) -> Path:
    """Edit one formation row in the formations registry table."""
    if not isinstance(formation_id, int) or isinstance(formation_id, bool):
        raise TypeError("formation_id must be an integer.")
    if any(field not in FORMATION_COLUMNS[1:] for field in updates):
        raise ValueError("Updates contain an unknown or non-editable field.")
    if "shape" in updates:
        validate_choice(str(updates["shape"]), FORMATION_SHAPES, "shape")
    if "plane" in updates and "shape" not in updates:
        with connect_database() as connection:
            current_shape = connection.execute(
                "SELECT shape FROM registry.formations WHERE formation_id = ?",
                [formation_id],
            ).fetchone()
        if current_shape is not None and current_shape[0] in FORMATION_SHAPES_WITHOUT_PLANE:
            updates["plane"] = None
    if updates.get("shape") in FORMATION_SHAPES_WITHOUT_PLANE:
        updates["plane"] = None
    if "plane" in updates and updates["plane"] is not None:
        validate_choice(str(updates["plane"]), FORMATION_PLANES, "plane")

    with connect_database() as connection:
        if connection.execute(
            "SELECT 1 FROM registry.formations WHERE formation_id = ?",
            [formation_id],
        ).fetchone() is None:
            raise ValueError(f"No formation with formation_id={formation_id} exists.")
        for field, value in updates.items():
            connection.execute(
                f'UPDATE registry.formations SET "{field}" = ? WHERE formation_id = ?',
                [value, formation_id],
            )
    return Path(output_path)


if __name__ == "__main__":
    make_formations_table()


make_formations_csv = make_formations_table
