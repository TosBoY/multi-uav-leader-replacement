"""Furthest-drone leader election for a collection of drones."""

from collections.abc import Mapping, Sequence
import math
from typing import TypeVar


Drone = TypeVar("Drone")


def elect_furthest_leader(
    drones: Sequence[Drone],
    positions: Mapping[Drone, Sequence[float]],
    current_leader: Drone,
) -> Drone:
    """Return the active drone furthest from the current leader.

    Args:
        drones: Ordered active drone identifiers, including the current leader.
        positions: Mapping from every drone identifier to its position. Positions
            must contain exactly three finite coordinates.
        current_leader: Identifier of the drone whose furthest neighbor is chosen.

    Returns:
        The furthest drone other than ``current_leader``. Equal distances are
        resolved by preserving the order in ``drones``.

    Raises:
        ValueError: If the inputs are empty, incomplete, invalid, or contain no
            drone available for promotion.
    """
    drones = list(drones)
    if not drones:
        raise ValueError("At least one drone is required.")
    if len(set(drones)) != len(drones):
        raise ValueError("Drone identifiers must be unique.")
    if current_leader not in drones:
        raise ValueError("The current leader must be included in drones.")

    coordinates: dict[Drone, tuple[float, float, float]] = {}
    for drone in drones:
        if drone not in positions:
            raise ValueError(f"Missing position for drone: {drone}")
        try:
            position = tuple(float(value) for value in positions[drone])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid position for drone: {drone}") from None
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError(
                f"Position for drone {drone} must contain three finite values."
            )
        coordinates[drone] = position

    candidates = [drone for drone in drones if drone != current_leader]
    if not candidates:
        raise ValueError("At least one drone other than the current leader is required.")

    leader_position = coordinates[current_leader]
    squared_distances = {
        drone: sum(
            (candidate - leader) ** 2
            for candidate, leader in zip(coordinates[drone], leader_position)
        )
        for drone in candidates
    }

    for drone in candidates:
        print(
            f"Distance from leader {current_leader} to drone {drone}: "
            f"{math.sqrt(squared_distances[drone]):.6f}"
        )

    return max(candidates, key=squared_distances.__getitem__)
