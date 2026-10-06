"""Visualization helpers for completed drone simulations."""

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rotorpy.utils.animate import animate
from rotorpy.world import World


def _scene_bounds(positions: np.ndarray) -> tuple[float, float, float, float, float, float]:
	"""Calculate padded world bounds from all finite recorded positions."""
	finite_positions = positions[np.isfinite(positions).all(axis=2)]
	if finite_positions.size == 0:
		raise ValueError("Simulation results contain no finite drone positions.")

	minimum = np.min(finite_positions, axis=0)
	maximum = np.max(finite_positions, axis=0)
	span = maximum - minimum
	padding = np.maximum(span * 0.1, 1.0)
	minimum -= padding
	maximum += padding
	return (
		float(minimum[0]),
		float(maximum[0]),
		float(minimum[1]),
		float(maximum[1]),
		float(minimum[2]),
		float(maximum[2]),
	)


def display_simulation(results: Mapping[str, Any]) -> Any:
	"""Display recorded drone paths and animate the completed simulation."""
	drones = results["drones"]
	positions = results["positions"]
	scene_bounds = _scene_bounds(positions)
	world = World.empty(scene_bounds)

	fig_3d = plt.figure("Leader Promotion Simulation")
	ax = fig_3d.add_subplot(projection="3d")
	world.draw(ax)
	ax.set_xlim(scene_bounds[0], scene_bounds[1])
	ax.set_ylim(scene_bounds[2], scene_bounds[3])
	ax.set_zlim(scene_bounds[4], scene_bounds[5])
	ax.set_box_aspect(
		(
			scene_bounds[1] - scene_bounds[0],
			scene_bounds[3] - scene_bounds[2],
			scene_bounds[5] - scene_bounds[4],
		)
	)
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
