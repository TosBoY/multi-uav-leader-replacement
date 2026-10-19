"""Minimum-travel leader election for a collection of drones."""

from collections.abc import Sequence
import math
from typing import TypeVar


Drone = TypeVar("Drone")
TravelDistance = tuple[Drone, float]


def elect_minimum_travel_leader(
    travel_distances: Sequence[TravelDistance],
    old_leader: Drone,
    print_election: bool = False,
) -> Drone:
    """Return the active drone that has traveled the shortest distance.

    Args:
        travel_distances: Pairs containing a drone identifier and its
            cumulative traveled distance.
        old_leader: Identifier of the failed leader, excluded from candidates.
        print_election: Print all distances and the elected leader when true.

    Returns:
        The candidate with the shortest cumulative travel distance. Ties are
        resolved by preserving the input order.
    """
    if not travel_distances:
        raise ValueError("At least one travel distance is required.")

    seen_drones: set[Drone] = set()
    candidates: list[TravelDistance] = []
    for drone, distance in travel_distances:
        if drone in seen_drones:
            raise ValueError("Drone identifiers must be unique.")
        seen_drones.add(drone)
        if isinstance(distance, bool) or not math.isfinite(distance) or distance < 0:
            raise ValueError(
                f"Travel distance for drone {drone} must be finite and non-negative."
            )
        if drone != old_leader:
            candidates.append((drone, float(distance)))

    if not candidates:
        raise ValueError("At least one active drone is required for promotion.")

    elected_drone, elected_distance = min(candidates, key=lambda item: item[1])
    if print_election:
        for drone, distance in travel_distances:
            print(f"Travel distance for drone {drone}: {distance:.6f}")
        print(
            f"Minimum-travel elected leader: {elected_drone} "
            f"({elected_distance:.6f})"
        )
    return elected_drone
