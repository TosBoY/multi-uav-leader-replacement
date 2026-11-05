"""Central-drone leader election for a collection of drones."""

from collections.abc import Mapping, Sequence
import math
from typing import TypeVar


Drone = TypeVar("Drone")


def elect_central_leader(
    drones: Sequence[Drone],
    positions: Mapping[Drone, Sequence[float]],
    current_leader: Drone,
    print_election: bool = False,
) -> Drone:
    """Return the active drone closest to all other active drones.

    The failed/current leader is excluded from both the candidates and the
    distance totals. Ties preserve the input order of ``drones``.
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
        raise ValueError("At least one active drone is required for promotion.")

    total_distances: dict[Drone, float] = {}
    for candidate in candidates:
        candidate_position = coordinates[candidate]
        total_distances[candidate] = sum(
            math.sqrt(
                sum(
                    (candidate_coordinate - other_coordinate) ** 2
                    for candidate_coordinate, other_coordinate in zip(
                        candidate_position,
                        coordinates[other_drone],
                    )
                )
            )
            for other_drone in candidates
            if other_drone != candidate
        )

    if print_election:
        print(f"Current leader: {current_leader}")
        for drone in candidates:
            print(
                f"Total distance from drone {drone} to active drones: "
                f"{total_distances[drone]:.6f}"
            )

    return min(candidates, key=total_distances.__getitem__)
