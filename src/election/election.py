"""Leader-election dispatcher for drone teams."""

from collections.abc import Mapping, Sequence
import random
from typing import Literal, TypeVar

from election.centroid_election import elect_closest_leader
from election.furthest_election import elect_furthest_leader
from election.random_election import elect_random_leader


Drone = TypeVar("Drone")
ElectionType = Literal["random", "closest", "furthest"]


def elect_new_leader(
    drones: Sequence[Drone],
    positions: Mapping[Drone, Sequence[float]],
    election_type: ElectionType,
    failed_drone: Drone,
    rng: random.Random | None = None,
) -> Drone:
    """Elect a replacement leader after a drone failure.

    Args:
        drones: Identifiers of all drones participating in the election.
        positions: Current position of every drone, keyed by drone identifier.
        election_type: Strategy to use: ``"random"``, ``"closest"``, or
            ``"furthest"``.
        failed_drone: Identifier of the drone that failed and cannot be elected.
        rng: Optional seeded random generator used by the random strategy.

    Returns:
        The identifier of the newly elected leader.

    Raises:
        ValueError: If the inputs are invalid or no drone can be elected.
    """
    drones = list(drones)
    if not drones:
        raise ValueError("At least one drone is required.")
    if len(set(drones)) != len(drones):
        raise ValueError("Drone identifiers must be unique.")
    if failed_drone not in drones:
        raise ValueError("The failed drone must be included in drones.")
    if any(drone not in positions for drone in drones):
        raise ValueError("A current position is required for every drone.")

    active_drones = [drone for drone in drones if drone != failed_drone]
    if not active_drones:
        raise ValueError("At least one active drone is required for promotion.")

    if election_type == "random":
        elected_drone = elect_random_leader(active_drones, rng)
    elif election_type == "closest":
        elected_drone = elect_closest_leader(
            drones,
            positions,
            failed_drone,
        )
    elif election_type == "furthest":
        elected_drone = elect_furthest_leader(
            drones,
            positions,
            failed_drone,
        )
    else:
        raise ValueError(
            'election_type must be "random", "closest", or "furthest".'
        )

    if elected_drone not in active_drones:
        raise ValueError("The election strategy returned an unavailable drone.")
    return elected_drone
