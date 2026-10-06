"""Reusable RotorPy simulation runner for leader/follower drone teams."""

from collections.abc import Mapping, Sequence
import random
from typing import Any, Literal, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.wind.default_winds import NoWind

from election.election import ElectionType, elect_new_leader
from simulation.display import display_simulation


Drone = TypeVar("Drone")


def _make_initial_state(
	position: Sequence[float] | np.ndarray,
) -> dict[str, np.ndarray]:
	"""Create the initial RotorPy state for one drone."""
	return {
		"x": np.array(position, dtype=float),
		"v": np.zeros(3),
		"q": np.array([0.0, 0.0, 0.0, 1.0]),
		"w": np.zeros(3),
		"wind": np.zeros(3),
		"rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
	}


def _make_flat_output(path_flat: Mapping[str, Any], position: np.ndarray) -> dict[str, Any]:
	"""Copy a path output while replacing its desired position."""
	return {
		"x": position,
		"x_dot": path_flat["x_dot"],
		"x_ddot": path_flat["x_ddot"],
		"x_dddot": path_flat["x_dddot"],
		"x_ddddot": path_flat["x_ddddot"],
		"yaw": path_flat["yaw"],
		"yaw_dot": path_flat["yaw_dot"],
		"yaw_ddot": path_flat["yaw_ddot"],
	}


def _route_frame(
	velocity: Sequence[float],
	fallback: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return forward, lateral, and vertical unit vectors for the route."""
	direction = np.asarray(velocity, dtype=float).copy()
	speed = np.linalg.norm(direction)
	if speed < 1e-9:
		if fallback is not None:
			return fallback[0].copy(), fallback[1].copy(), fallback[2].copy()
		forward = np.array([1.0, 0.0, 0.0])
	else:
		forward = direction / speed

	world_up = np.array([0.0, 0.0, 1.0])
	lateral = np.cross(world_up, forward)
	lateral_speed = np.linalg.norm(lateral)
	if lateral_speed < 1e-9:
		lateral = np.cross(np.array([1.0, 0.0, 0.0]), forward)
		lateral_speed = np.linalg.norm(lateral)
	lateral /= lateral_speed
	vertical = np.cross(forward, lateral)
	return forward, lateral, vertical


def _route_aligned_offset(
	velocity: Sequence[float],
	formation_coordinates: tuple[float, float, float],
	fallback_frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
	"""Convert route-relative coordinates into a world offset."""
	forward, lateral, vertical = _route_frame(velocity, fallback_frame)
	behind, lateral_distance, vertical_distance = formation_coordinates
	return (
		-behind * forward
		+ lateral_distance * lateral
		+ vertical_distance * vertical
	)


def _normalise_positions(
	drones: Sequence[Drone],
	positions: Mapping[Drone, Sequence[float]] | Sequence[Sequence[float]],
) -> dict[Drone, np.ndarray]:
	"""Convert either supported position format into a drone-keyed mapping."""
	if isinstance(positions, Mapping):
		missing = [drone for drone in drones if drone not in positions]
		if missing:
			raise ValueError(f"Missing initial positions for drones: {missing}")
		return {
			drone: np.asarray(positions[drone], dtype=float).copy()
			for drone in drones
		}

	if len(positions) != len(drones):
		raise ValueError("The number of positions must match the number of drones.")
	return {
		drone: np.asarray(position, dtype=float).copy()
		for drone, position in zip(drones, positions)
	}


def run_simulation(
	drones: Sequence[Drone],
	initial_leader: Drone,
	initial_positions: Mapping[Drone, Sequence[float]] | Sequence[Sequence[float]],
	leader_path: Any,
	leader_failure_time: float | None,
	election_type: ElectionType = "random",
	*,
	formation_offsets: Mapping[Drone, Sequence[float]] | None = None,
	route_aligned_formation: bool = False,
	sim_rate: int = 100,
	t_final: float = 10.0,
	random_seed: int | None = None,
	display: bool | str = "no",
) -> dict[str, Any]:
	"""Run a configurable RotorPy leader/follower simulation.

	Args:
		drones: Ordered drone identifiers. Identifiers must be hashable because
			they are used as keys in the returned histories and state mappings.
		initial_leader: Identifier of the drone that starts as leader.
		initial_positions: Either a mapping from drone identifiers to ``[x, y,
			z]`` positions or a sequence in the same order as ``drones``.
		leader_path: RotorPy trajectory object. It must provide ``update(t)`` and
			return the flat-output dictionary expected by ``SE3Control``.
		leader_failure_time: Time at which the current leader fails. Pass
			``None`` to run without a leader failure or election.
		election_type: Election strategy to use: ``"random"`` or ``"closest"``.
		formation_offsets: Optional fixed offsets from the leader for each drone.
			If omitted, offsets are inferred from the initial positions. The
			promoted leader receives the zero offset; other drones retain theirs.
		route_aligned_formation: If ``True``, rotate each formation offset with
			the leader's current three-dimensional direction of travel.
		sim_rate: Simulation frequency in Hz.
		t_final: Simulation duration in seconds.
		random_seed: Optional seed passed to the random election strategy.
		display: Use ``"yes"`` to show the 3D plot and animation after the
			simulation, or ``"no"`` to return results without opening a window.
			Boolean values are also accepted.

	Returns:
		A dictionary containing time, positions, rotations, wind, leader history,
		failure history, promotion time, and the ordered drone identifiers.

	Raises:
		ValueError: If the inputs are inconsistent or the election returns an
			unavailable drone.
	"""
	drones = list(drones)
	if not drones:
		raise ValueError("At least one drone is required.")
	if len(set(drones)) != len(drones):
		raise ValueError("Drone identifiers must be unique.")
	if initial_leader not in drones:
		raise ValueError("The initial leader must be included in drones.")
	if sim_rate <= 0:
		raise ValueError("sim_rate must be greater than zero.")
	if t_final < 0:
		raise ValueError("t_final cannot be negative.")
	if leader_failure_time is not None and leader_failure_time < 0:
		raise ValueError("leader_failure_time cannot be negative.")
	if isinstance(display, str):
		display_value = display.strip().lower()
		if display_value not in {"yes", "no"}:
			raise ValueError('display must be either "yes" or "no".')
		show_display = display_value == "yes"
	elif isinstance(display, bool):
		show_display = display
	else:
		raise TypeError('display must be either "yes", "no", or a boolean.')

	positions = _normalise_positions(drones, initial_positions)
	if any(position.shape != (3,) for position in positions.values()):
		raise ValueError("Every initial position must contain exactly three values.")

	leader_position = positions[initial_leader]
	if formation_offsets is None:
		offsets = {
			drone: positions[drone] - leader_position
			for drone in drones
		}
	else:
		missing = [drone for drone in drones if drone not in formation_offsets]
		if missing:
			raise ValueError(f"Missing formation offsets for drones: {missing}")
		offsets = {
			drone: np.asarray(formation_offsets[drone], dtype=float).copy()
			for drone in drones
		}

	if any(offset.shape != (3,) for offset in offsets.values()):
		raise ValueError("Every formation offset must contain exactly three values.")

	formation_coordinates: dict[Drone, tuple[float, float, float]] = {}
	formation_frame = None
	if route_aligned_formation:
		initial_flat = leader_path.update(0.0)
		formation_frame = _route_frame(initial_flat["x_dot"])
		initial_forward, initial_lateral, initial_vertical = formation_frame
		formation_coordinates = {
			drone: (
				-np.dot(offsets[drone], initial_forward),
				np.dot(offsets[drone], initial_lateral),
				np.dot(offsets[drone], initial_vertical),
			)
			for drone in drones
		}

	vehicles = {}
	controllers = {}
	states = {}
	for drone in drones:
		state = _make_initial_state(positions[drone])
		vehicles[drone] = Multirotor(quad_params, initial_state=state)
		controllers[drone] = SE3Control(quad_params)
		states[drone] = state

	print(
		f"Created {len(drones)} UAVs. "
		f"UAV {initial_leader} is the initial leader."
	)
	if leader_failure_time is not None:
		print(
			f"Leader failure scheduled at t = {leader_failure_time:.2f} s."
		)

	num_steps = int(t_final * sim_rate) + 1
	dt = 1.0 / sim_rate
	time_history = np.zeros(num_steps)
	position_history = np.zeros((num_steps, len(drones), 3))
	quat_history = np.zeros((num_steps, len(drones), 4))
	wind_history = np.zeros((num_steps, len(drones), 3))
	leader_history = np.empty(num_steps, dtype=object)
	failed_history = np.zeros((num_steps, len(drones)), dtype=bool)

	drone_indices = {drone: index for index, drone in enumerate(drones)}
	for drone in drones:
		index = drone_indices[drone]
		position_history[0, index] = states[drone]["x"]
		quat_history[0, index] = states[drone]["q"]
		wind_history[0, index] = states[drone]["wind"]

	active_drones = list(drones)
	failed_drones: set[Drone] = set()
	active_leader = initial_leader
	election_rng = random.Random(random_seed)
	promotion_time = None
	leader_history[0] = active_leader
	wind_profile = NoWind()

	for step in range(1, num_steps):
		time = step * dt
		time_history[step] = time

		if (
			leader_failure_time is not None
			and promotion_time is None
			and time >= leader_failure_time
		):
			failed_leader = active_leader
			failed_drones.add(active_leader)
			active_drones.remove(active_leader)
			if not active_drones:
				raise ValueError("The failed leader was the only active drone.")

			current_positions = {
				drone: states[drone]["x"].copy()
				for drone in drones
			}
			promoted_leader = elect_new_leader(
				drones=drones,
				positions=current_positions,
				election_type=election_type,
				failed_drone=failed_leader,
				rng=election_rng,
			)
			if promoted_leader not in active_drones:
				raise ValueError(
					"The election function must return an active drone."
				)
			active_leader = promoted_leader
			active_drones.remove(active_leader)
			active_drones.insert(0, active_leader)
			offsets[active_leader] = np.zeros(3)
			promotion_time = time
			print(
				f"UAV {failed_leader} failed at t = {time:.2f} s. "
				f"UAV {active_leader} promoted to leader."
			)

		leader_history[step] = active_leader
		for failed_drone in failed_drones:
			failed_history[step, drone_indices[failed_drone]] = True

		path_flat = leader_path.update(time)
		if route_aligned_formation:
			formation_frame = _route_frame(
				path_flat["x_dot"],
				formation_frame,
			)
		leader_position = states[active_leader]["x"].copy()
		controls = {}

		for drone in active_drones:
			if drone == active_leader:
				flat_output = path_flat
			else:
				if route_aligned_formation:
					desired_offset = _route_aligned_offset(
					path_flat["x_dot"],
					formation_coordinates[drone],
					formation_frame,
				)
				else:
					desired_offset = offsets[drone]
				desired_position = leader_position + desired_offset
				flat_output = _make_flat_output(path_flat, desired_position)
			controls[drone] = controllers[drone].update(
				time,
				states[drone],
				flat_output,
			)

		for drone in drones:
			index = drone_indices[drone]
			if drone in failed_drones:
				position_history[step, index] = np.nan
				quat_history[step, index] = quat_history[step - 1, index]
				wind_history[step, index] = wind_history[step - 1, index]
				continue

			states[drone]["wind"] = wind_profile.update(
				time,
				states[drone]["x"],
			)
			states[drone] = vehicles[drone].step(
				states[drone],
				controls[drone],
				dt,
			)
			position_history[step, index] = states[drone]["x"]
			quat_history[step, index] = states[drone]["q"]
			wind_history[step, index] = states[drone]["wind"]

	rotation_history = (
		Rotation.from_quat(quat_history.reshape(-1, 4))
		.as_matrix()
		.reshape(num_steps, len(drones), 3, 3)
	)

	results = {
		"drones": drones,
		"time": time_history,
		"positions": position_history,
		"rotations": rotation_history,
		"wind": wind_history,
		"leader_history": leader_history,
		"failed_history": failed_history,
		"promotion_time": promotion_time,
	}

	print("Simulation finished.")
	for drone in drones:
		index = drone_indices[drone]
		print(
			f"UAV {drone}: start {position_history[0, index]} -> "
			f"end {position_history[-1, index]}"
		)
	print(f"Time steps: {len(time_history)}")

	if show_display:
		print("Opening leader-promotion animation...")
		display_simulation(results)

	return results
