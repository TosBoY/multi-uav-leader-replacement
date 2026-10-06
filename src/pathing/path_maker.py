"""Factories for leader trajectories used by the simulation runner."""

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.trajectories.speed_traj import ConstantSpeed

from pathing.random_waypoints import generate_random_waypoints


PathType = Literal["random", "line"]


def _speed_from_velocity(
    velocity: float | Sequence[float] | np.ndarray,
) -> float:
    """Return a finite non-negative speed from a scalar or velocity vector."""
    velocity_array = np.asarray(velocity, dtype=float)
    if velocity_array.ndim == 0:
        speed = float(velocity_array)
    elif velocity_array.shape == (3,):
        speed = float(np.linalg.norm(velocity_array))
    else:
        raise ValueError("velocity must be a scalar or contain three values.")
    if not np.isfinite(speed) or speed < 0:
        raise ValueError("velocity must be finite and non-negative.")
    return speed


def _line_axis(initial_heading: float) -> int:
    """Return the ConstantSpeed axis for a cardinal initial heading."""
    heading = initial_heading % (2.0 * np.pi)
    if np.isclose(np.sin(heading), 0.0):
        if np.cos(heading) < 0:
            raise ValueError("line paths do not support negative x direction.")
        return 0
    if np.isclose(np.cos(heading), 0.0):
        if np.sin(heading) < 0:
            raise ValueError("line paths do not support negative y direction.")
        return 1
    raise ValueError(
        "line paths require an initial_heading along positive x or positive y."
    )


def make_leader_path(
    path_type: PathType,
    first_waypoint: Sequence[float] | np.ndarray,
    velocity: float | Sequence[float] | np.ndarray,
    time: float,
    seed: int | None = None,
    print_waypoints: bool = False,
    initial_heading: float = 0.0,
) -> Any:
    """Create a leader path compatible with ``run_simulation``.

    Args:
        path_type: ``"random"`` for a MinSnap path through generated
            waypoints, or ``"line"`` for a ConstantSpeed path.
        first_waypoint: Starting position as ``[x, y, z]``.
        velocity: Scalar speed or three-dimensional velocity vector.
        time: Intended path duration in seconds.
        seed: Seed used by the random waypoint path.
        print_waypoints: Print generated waypoints for a random path.
        initial_heading: Initial horizontal heading in radians.

    Returns:
        A RotorPy trajectory object exposing ``update(time)``.

    Raises:
        ValueError: If the path type or path parameters are invalid.
    """
    if not isinstance(path_type, str):
        raise TypeError("path_type must be a string.")
    if not isinstance(print_waypoints, bool):
        raise TypeError("print_waypoints must be a Boolean.")
    if isinstance(initial_heading, bool) or not np.isfinite(initial_heading):
        raise ValueError("initial_heading must be a finite angle in radians.")
    if isinstance(time, bool) or not np.isfinite(time) or time < 0:
        raise ValueError("time must be finite and non-negative.")

    starting_point = np.asarray(first_waypoint, dtype=float)
    if starting_point.shape != (3,):
        raise ValueError("first_waypoint must contain exactly three values.")
    if not np.all(np.isfinite(starting_point)):
        raise ValueError("first_waypoint must contain only finite values.")

    speed = _speed_from_velocity(velocity)
    if path_type == "random":
        waypoints = generate_random_waypoints(
            first_waypoint=starting_point,
            velocity=velocity,
            time=time,
            seed=seed,
            print_waypoints=print_waypoints,
            initial_heading=initial_heading,
        )
        return MinSnap(
            waypoints,
            v_avg=max(1, int(speed)),
            verbose=False,
        )

    if path_type == "line":
        if speed == 0:
            raise ValueError("line paths require a positive velocity.")
        line_trajectory = cast(Any, ConstantSpeed)
        return line_trajectory(
            init_pos=starting_point,
            dist=speed * time,
            speed=speed,
            axis=_line_axis(initial_heading),
            yaw_traj="forward",
        )

    raise ValueError('path_type must be either "random" or "line".')
