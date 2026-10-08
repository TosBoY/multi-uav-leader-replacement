"""Generate initial drone formations."""

from collections.abc import Sequence
from typing import Literal

import numpy as np


FormationType = Literal["leader_follower"]
FormationShape = Literal[
    "line_x",
    "line_y",
    "line_z",
    "arrow",
    "circle",
    "grid",
]
FormationPlane = Literal["xy", "xz", "yz"]


def _validate_inputs(
    number_of_drones: int,
    spread: float,
    dimension: int,
    plane: FormationPlane,
    leader_index: int,
    leader_position: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Validate common formation inputs and return the leader position."""
    if isinstance(number_of_drones, bool) or not isinstance(number_of_drones, int):
        raise TypeError("number_of_drones must be an integer.")

    if number_of_drones < 1:
        raise ValueError("number_of_drones must be at least one.")

    if not np.isfinite(spread) or spread <= 0:
        raise ValueError("spread must be finite and greater than zero.")

    if dimension not in {2, 3}:
        raise ValueError("dimension must be either 2 or 3.")

    if plane not in {"xy", "xz", "yz"}:
        raise ValueError('plane must be "xy", "xz", or "yz".')

    if not 0 <= leader_index < number_of_drones:
        raise ValueError("leader_index must identify one of the drones.")

    position = np.asarray(leader_position, dtype=float)

    if position.shape != (3,):
        raise ValueError("leader_position must contain exactly three values.")

    if not np.all(np.isfinite(position)):
        raise ValueError("leader_position must contain only finite values.")

    return position


def _arrow_offsets(number_of_drones: int, spread: float) -> list[np.ndarray]:
    """Create offsets with the leader at the arrow tip."""
    offsets = [np.zeros(3)]

    for follower_rank in range(1, number_of_drones):
        row = (follower_rank + 1) // 2
        side = -1.0 if follower_rank % 2 else 1.0

        offsets.append(
            np.array([-row * spread, side * row * spread, 0.0])
        )

    return offsets


def _line_offsets(
    number_of_drones: int,
    spread: float,
    axis: Literal["x", "y", "z"],
) -> list[np.ndarray]:
    """Create evenly spaced offsets behind the leader along one axis."""
    offsets = []

    for index in range(number_of_drones):
        distance = -index * spread

        if axis == "x":
            offset = np.array([distance, 0.0, 0.0])
        elif axis == "y":
            offset = np.array([0.0, distance, 0.0])
        else:
            offset = np.array([0.0, 0.0, distance])

        offsets.append(offset)

    return offsets


def _circle_offsets(
    number_of_drones: int,
    spread: float,
    dimension: int,
) -> list[np.ndarray]:
    """Create a leader-centered circle or sphere of follower offsets."""
    if number_of_drones == 1:
        return [np.zeros(3)]

    offsets = [np.zeros(3)]
    follower_count = number_of_drones - 1
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for index in range(follower_count):
        if dimension == 2:
            angle = 2.0 * np.pi * index / follower_count
            offset = np.array(
                [
                    spread * np.cos(angle),
                    spread * np.sin(angle),
                    0.0,
                ]
            )
        else:
            # Use the Fibonacci sphere algorithm to evenly distribute points on a sphere.
            angle = golden_angle * index
            vertical = 1.0 - 2.0 * (index + 0.5) / follower_count
            radius = np.sqrt(max(0.0, 1.0 - vertical**2))

            offset = spread * np.array(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    vertical,
                ]
            )

        offsets.append(offset)

    return offsets


def _grid_offsets(
    number_of_drones: int,
    spread: float,
) -> list[np.ndarray]:
    """Create a square grid."""
    side_length = int(np.ceil(np.sqrt(number_of_drones)))
    center = (side_length - 1) / 2.0

    offsets = []

    for index in range(number_of_drones):
        row, column = divmod(index, side_length)

        offsets.append(
            np.array(
                [
                    (column - center) * spread,
                    (row - center) * spread,
                    0.0,
                ]
            )
        )

    leader_offset = offsets[0].copy()

    return [offset - leader_offset for offset in offsets]


def make_formation(
    number_of_drones: int,
    formation_type: FormationType = "leader_follower",
    shape: FormationShape = "arrow",
    spread: float = 1.0,
    dimension: int = 2,
    plane: FormationPlane = "xy",
    leader_index: int = 0,
    leader_position: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
) -> dict[int, np.ndarray]:
    """Generate initial positions for a drone formation.

    Args:
        number_of_drones: Total number of drones, including the leader.
        formation_type: Formation behavior. Currently ``"leader_follower"``.
        shape: ``"line_x"``, ``"line_y"``, ``"line_z"``,
            ``"arrow"``, ``"circle"``, or ``"grid"``.
        spread: Distance between neighboring formation elements in meters.
        dimension: Arrange the formation in 2D or 3D.
        plane: For 2D formations, determines the two world axes used
            by ``arrow``, ``circle``, and ``grid``.
        leader_index: Drone identifier to place at the formation origin.
        leader_position: World position of the leader.

    Returns:
        A mapping from drone identifiers to 3D NumPy position arrays.
    """
    position = _validate_inputs(
        number_of_drones,
        spread,
        dimension,
        plane,
        leader_index,
        leader_position,
    )

    if formation_type != "leader_follower":
        raise ValueError('formation_type must be "leader_follower".')

    # Line formations explicitly define their own axis.
    # They are not affected by the plane parameter.
    if shape == "line_x":
        offsets = _line_offsets(number_of_drones, spread, "x")
        apply_plane = False

    elif shape == "line_y":
        offsets = _line_offsets(number_of_drones, spread, "y")
        apply_plane = False

    elif shape == "line_z":
        offsets = _line_offsets(number_of_drones, spread, "z")
        apply_plane = False

    elif shape == "arrow":
        offsets = _arrow_offsets(number_of_drones, spread)
        apply_plane = True

    elif shape == "circle":
        offsets = _circle_offsets(
            number_of_drones,
            spread,
            dimension,
        )
        apply_plane = True

    elif shape == "grid":
        offsets = _grid_offsets(number_of_drones, spread)
        apply_plane = True

    else:
        raise ValueError(
            'shape must be "line_x", "line_y", "line_z", '
            '"arrow", "circle", or "grid".'
        )

    # Move the formation so that the selected leader is at the origin.
    leader_offset = offsets[leader_index].copy()
    offsets = [offset - leader_offset for offset in offsets]

    # Only apply the plane constraint to formations that use a 2D plane.
    if dimension == 2 and apply_plane:
        if plane == "xy":
            offsets = [
                np.array([offset[0], offset[1], 0.0])
                for offset in offsets
            ]

        elif plane == "xz":
            offsets = [
                np.array([offset[0], 0.0, offset[1]])
                for offset in offsets
            ]

        elif plane == "yz":
            offsets = [
                np.array([0.0, offset[0], offset[1]])
                for offset in offsets
            ]

    return {
        drone: position + offset
        for drone, offset in enumerate(offsets)
    }