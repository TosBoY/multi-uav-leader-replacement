"""Visualization helpers for completed drone simulations."""

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rotorpy.utils.animate import animate
from rotorpy.world import World


def display_simulation(results: Mapping[str, Any]) -> Any:
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
