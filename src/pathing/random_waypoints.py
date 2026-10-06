"""Random but reproducible three-dimensional waypoint generation."""

from collections.abc import Sequence

import numpy as np


def generate_random_waypoints(
    first_waypoint: Sequence[float] | np.ndarray,
    velocity: float | Sequence[float] | np.ndarray,
    time: float,
    seed: int | None = None,
    print_waypoints: bool = False,
    initial_heading: float = 0.0,
) -> np.ndarray:
    """Generate a smooth random walk for a velocity and duration.

    Args:
        first_waypoint: The first waypoint as ``[x, y, z]``.
        velocity: Travel speed as a scalar or a three-dimensional velocity
            vector. A vector's magnitude is used to calculate travel distance.
        time: Travel duration in seconds.
        seed: Optional local random seed. The seed makes the returned waypoints
            reproducible without changing NumPy's global random-number state.
        print_waypoints: If ``True``, print the array length and generated
            waypoint array.
        initial_heading: Initial horizontal movement heading in radians. Zero
            points along positive x, and ``pi / 2`` points along positive y.

    Returns:
        An array with shape ``(length, 3)``. The length is randomly selected
        from the calculated travel distance. Each waypoint is generated from
        the previous waypoint.

    Raises:
        ValueError: If an input is invalid or negative.
        TypeError: If ``print_waypoints`` is not a Boolean.
    """
    if not isinstance(print_waypoints, bool):
        raise TypeError("print_waypoints must be a Boolean.")
    if isinstance(initial_heading, bool) or not np.isfinite(initial_heading):
        raise ValueError("initial_heading must be a finite angle in radians.")

    starting_point = np.asarray(first_waypoint, dtype=float)
    if starting_point.shape != (3,):
        raise ValueError("first_waypoint must contain exactly three values.")
    if not np.all(np.isfinite(starting_point)):
        raise ValueError("first_waypoint must contain only finite values.")

    velocity_array = np.asarray(velocity, dtype=float)
    if velocity_array.ndim == 0:
        speed = float(velocity_array)
    elif velocity_array.shape == (3,):
        speed = float(np.linalg.norm(velocity_array))
    else:
        raise ValueError("velocity must be a scalar or contain three values.")
    if not np.isfinite(speed) or speed < 0:
        raise ValueError("velocity must be finite and non-negative.")
    if isinstance(time, bool) or not np.isfinite(time) or time < 0:
        raise ValueError("time must be finite and non-negative.")

    travel_distance = speed * time
    if travel_distance == 0:
        waypoints = starting_point.reshape(1, 3).copy()
        if print_waypoints:
            print(f"Waypoint length: {len(waypoints)}")
            print(waypoints)
        return waypoints

    generator = np.random.default_rng(seed)
    minimum_steps = 2
    maximum_steps = max(minimum_steps, int(np.ceil(travel_distance)))
    step_count = int(
        generator.integers(minimum_steps, maximum_steps + 1)
    )
    length = step_count + 1
    path_distance = travel_distance * generator.uniform(1.0, 1.25)
    step_length = path_distance / step_count
    maximum_turn = np.pi / np.sqrt(step_count)

    waypoints = np.empty((length, 3), dtype=float)
    waypoints[0] = starting_point

    direction = np.array(
        [
            np.cos(initial_heading),
            np.sin(initial_heading),
            0.0,
        ]
    )

    for index in range(1, length):
        if index == 1:
            waypoints[index] = waypoints[index - 1] + step_length * direction
            continue

        perpendicular = generator.normal(size=3)
        perpendicular -= np.dot(perpendicular, direction) * direction
        perpendicular_norm = np.linalg.norm(perpendicular)
        if perpendicular_norm < 1e-12:
            perpendicular = np.cross(direction, np.array([1.0, 0.0, 0.0]))
            perpendicular_norm = np.linalg.norm(perpendicular)
            if perpendicular_norm < 1e-12:
                perpendicular = np.cross(direction, np.array([0.0, 1.0, 0.0]))
                perpendicular_norm = np.linalg.norm(perpendicular)
        perpendicular /= perpendicular_norm

        turn_angle = generator.uniform(-maximum_turn, maximum_turn)
        direction = (
            np.cos(turn_angle) * direction
            + np.sin(turn_angle) * perpendicular
        )
        direction /= np.linalg.norm(direction)
        waypoints[index] = waypoints[index - 1] + step_length * direction

    if print_waypoints:
        print(f"Waypoint length: {len(waypoints)}")
        print(waypoints)

    return waypoints
