"""Smallest-ID leader election for a collection of drones."""

from collections.abc import Sequence
from typing import TypeVar


Drone = TypeVar("Drone")


def elect_smallest_id_leader(
    drones: Sequence[Drone],
    old_leader: Drone,
) -> Drone:
    """Return the active drone with the smallest comparable identifier."""
    drones = list(drones)
    if not drones:
        raise ValueError("At least one drone is required.")
    if len(set(drones)) != len(drones):
        raise ValueError("Drone identifiers must be unique.")
    if old_leader not in drones:
        raise ValueError("The old leader must be included in drones.")

    candidates = [drone for drone in drones if drone != old_leader]
    if not candidates:
        raise ValueError("At least one active drone is required for promotion.")
    try:
        return min(candidates)
    except TypeError as error:
        raise ValueError("Drone identifiers must be comparable.") from error
