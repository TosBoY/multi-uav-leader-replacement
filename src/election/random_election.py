"""Random leader election for a collection of drones."""

from collections.abc import Sequence
import random
from typing import TypeVar


Drone = TypeVar("Drone")


def elect_random_leader(
    drones: Sequence[Drone],
    rng: random.Random | None = None,
) -> Drone:
    """Return one randomly selected drone from the current drone list.

    Args:
        drones: A non-empty sequence containing the available drones. The items
            may be drone IDs, drone objects, or any other value identifying a
            candidate.
        rng: Optional random-number generator. Supplying a ``random.Random``
            instance makes the election reproducible when it has been seeded.
            When omitted, Python's module-level random generator is used.

    Returns:
        One member of ``drones``, selected with equal probability.

    Raises:
        ValueError: If ``drones`` is empty, because there is no candidate that
            can become leader.
    """
    if not drones:
        raise ValueError("Cannot elect a leader from an empty drone list.")

    selector = rng if rng is not None else random
    return selector.choice(drones)
