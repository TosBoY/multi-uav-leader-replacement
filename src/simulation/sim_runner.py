"""Reusable RotorPy simulation runner for leader/follower drone teams."""

from collections.abc import Callable, Mapping, Sequence
import random
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.utils.animate import animate
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.wind.default_winds import NoWind
from rotorpy.world import World

from election.random_election import elect_random_leader


Drone = TypeVar("Drone")
ElectionFunction = Callable[[Sequence[Drone], random.Random | None], Drone]


def _make_initial_state(position: Sequence[float]) -> dict[str, np.ndarray]:
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


def _display_simulation(results: Mapping[str, Any]) -> Any:
	"""Display recorded drone paths and animate the completed simulation."""
	world = World.empty((-8, 4, -3, 3, -2, 3))
	drones = results["drones"]
	positions = results["positions"]

	fig_3d = plt.figure("Leader Promotion Simulation")
	ax = fig_3d.add_subplot(projection="3d")
	world.draw(ax)
	for index, drone in enumerate(drones):
		path = positions[:, index, :]
		ax.plot3D(
			path[:, 0],
			path[:, 1],
			path[:, 2],
			".",
			label=f"Drone {drone}",
		)

	leader_history = results["leader_history"]
	promotion_steps = np.flatnonzero(leader_history != leader_history[0])
	if promotion_steps.size:
		promotion_step = promotion_steps[0]
		promoted_leader = leader_history[promotion_step]
		promoted_index = drones.index(promoted_leader)
		ax.scatter(
			*positions[promotion_step, promoted_index],
			color="green",
			marker="*",
			s=100,
			label=f"Drone {promoted_leader} promoted",
		)
	ax.legend()

	animation = animate(
		results["time"],
		positions,
		results["rotations"],
		results["wind"],
		animate_wind=False,
		world=world,
	)
	plt.show()
	return animation


def run_simulation(
	drones: Sequence[Drone],
	initial_leader: Drone,
	initial_positions: Mapping[Drone, Sequence[float]] | Sequence[Sequence[float]],
	leader_path: Any,
	leader_failure_time: float | None,
	election_type: ElectionFunction = elect_random_leader,
	*,
	formation_offsets: Mapping[Drone, Sequence[float]] | None = None,
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
		election_type: Election function receiving the active drone list and a
			random generator. It must return one active drone. The default is
			``elect_random_leader``.
		formation_offsets: Optional fixed offsets from the leader for each drone.
			If omitted, offsets are inferred from the initial positions. The
			promoted leader receives the zero offset; other drones retain theirs.
		sim_rate: Simulation frequency in Hz.
		t_final: Simulation duration in seconds.
		random_seed: Optional seed passed to the election function.
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

			promoted_leader = election_type(active_drones, election_rng)
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
		leader_position = states[active_leader]["x"].copy()
		controls = {}

		for drone in active_drones:
			if drone == active_leader:
				flat_output = path_flat
			else:
				desired_position = leader_position + offsets[drone]
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
		_display_simulation(results)

	return results
