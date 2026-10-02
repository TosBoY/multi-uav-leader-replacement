"""Simulate random leader promotion using the shared election function.

This file keeps the same straight-line leader-promotion simulation as
`test_rotorpy_leader_promotion_line_commented.py`, but delegates the random
leader selection to `src.election.random_election.elect_random_leader`.
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.utils.animate import animate
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.wind.default_winds import NoWind
from rotorpy.world import World

# When this file is executed directly, Python places `thesis_simulation` on the
# import path instead of the project root. Add the sibling `src` directory so
# the shared election module can be imported in both execution styles.
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from election.random_election import elect_random_leader

# -----------------------------------------------------------------------------
# Simulation configuration.
# These constants define the time discretization, total simulation duration, and
# the logic used to trigger a leadership handover.
# -----------------------------------------------------------------------------
SIM_RATE = 100
DT = 1.0 / SIM_RATE
T_FINAL = 10.0
NUM_UAVS = 5
LEADER_FAILURE_TIME = T_FINAL / 2.0
RANDOM_SEED = 42

# The leader starts at a point slightly in front of the formation. The follower
# offsets create an arrow-like arrangement around the current leader.
LEADER_INITIAL_POSITION = np.array([-2.0, 0.0, 1.0])
FOLLOWER_SPACING = 1.0
FORMATION_POSITIONS = (
    (0.0, 0.0),   # Leader at the arrow tip.
    (0.8, -1.0),  # Near left wing.
    (0.8, 1.0),   # Near right wing.
    (-0.8, -1.0),  # Rear left wing.
    (-0.8, 1.0),   # Rear right wing.
)
LINE_DISTANCE = 5.0
LINE_SPEED = 0.5
LINE_AXIS = 0


def make_initial_state(position):
    """Create the initial RotorPy state dictionary for one UAV."""
    return {
        "x": np.array(position, dtype=float),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }


def make_leader_trajectory():
    """Build the constant-speed straight-line trajectory for the leader."""
    return ConstantSpeed(
        init_pos=LEADER_INITIAL_POSITION,
        dist=LINE_DISTANCE,
        speed=LINE_SPEED,
        axis=LINE_AXIS,
        yaw_traj="forward",
    )


def make_path_flat_output(path_flat, position):
    """Create a controller target using a path output and a new position."""
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


def make_following_flat_output(leader_flat, leader_position, rank):
    """Compute a follower's controller target from its formation slot."""
    behind, lateral = FORMATION_POSITIONS[rank]
    desired_position = leader_position + np.array(
        [-behind * FOLLOWER_SPACING, lateral * FOLLOWER_SPACING, 0.0]
    )
    return make_path_flat_output(leader_flat, desired_position)


def run_simulation():
    """Run the simulation and return the recorded UAV state histories."""
    wind_profile = NoWind()
    leader_trajectory = make_leader_trajectory()

    initial_positions = [
        LEADER_INITIAL_POSITION
        + np.array(
            [
                -FORMATION_POSITIONS[i][0] * FOLLOWER_SPACING,
                FORMATION_POSITIONS[i][1] * FOLLOWER_SPACING,
                0.0,
            ]
        )
        for i in range(NUM_UAVS)
    ]

    vehicles = []
    controllers = []
    states = []
    for position in initial_positions:
        initial_state = make_initial_state(position)
        vehicles.append(Multirotor(quad_params, initial_state=initial_state))
        controllers.append(SE3Control(quad_params))
        states.append(initial_state)

    num_steps = int(T_FINAL * SIM_RATE) + 1
    time_history = np.zeros(num_steps)
    position_history = np.zeros((num_steps, NUM_UAVS, 3))
    quat_history = np.zeros((num_steps, NUM_UAVS, 4))
    wind_history = np.zeros((num_steps, NUM_UAVS, 3))
    leader_history = np.full(num_steps, -1, dtype=int)
    failed_history = np.zeros((num_steps, NUM_UAVS), dtype=bool)

    for i in range(NUM_UAVS):
        position_history[0, i] = states[i]["x"]
        quat_history[0, i] = states[i]["q"]
        wind_history[0, i] = states[i]["wind"]

    active_order = list(range(NUM_UAVS))
    formation_ranks = {i: i for i in range(NUM_UAVS)}
    failed_drones = set()
    active_leader = 0
    promotion_rng = random.Random(RANDOM_SEED)
    promotion_time = None
    leader_history[0] = active_leader

    print(f"Created {NUM_UAVS} UAVs. UAV 0 is the initial leader.")
    print(f"Leader failure scheduled at t = {LEADER_FAILURE_TIME:.2f} s.")

    for step in range(1, num_steps):
        t = step * DT
        time_history[step] = t

        if active_leader == 0 and t >= LEADER_FAILURE_TIME:
            failed_drones.add(active_leader)
            active_order.remove(active_leader)

            # The election module receives the surviving drone list and returns
            # one randomly selected candidate to become the new leader. Passing
            # the seeded generator preserves the original reproducible behavior.
            active_leader = elect_random_leader(active_order)

            active_order.remove(active_leader)
            active_order.insert(0, active_leader)
            # Only the promoted drone takes the leader slot. Every other
            # survivor keeps its original formation slot.
            formation_ranks[active_leader] = 0
            promotion_time = t
            print(
                f"UAV 0 failed at t = {t:.2f} s. "
                f"UAV {active_leader} promoted to leader."
            )

        leader_history[step] = active_leader
        for failed_index in failed_drones:
            failed_history[step, failed_index] = True

        path_flat = leader_trajectory.update(t)
        leader_position = states[active_leader]["x"].copy()
        controls = [None] * NUM_UAVS

        for i in active_order:
            if i == active_leader:
                flat_output = path_flat
            else:
                flat_output = make_following_flat_output(
                    path_flat,
                    leader_position,
                    formation_ranks[i],
                )
            controls[i] = controllers[i].update(t, states[i], flat_output)

        for i in range(NUM_UAVS):
            if i in failed_drones:
                position_history[step, i] = np.nan
                quat_history[step, i] = quat_history[step - 1, i]
                wind_history[step, i] = wind_history[step - 1, i]
                continue

            states[i]["wind"] = wind_profile.update(t, states[i]["x"])
            states[i] = vehicles[i].step(states[i], controls[i], DT)
            position_history[step, i] = states[i]["x"]
            quat_history[step, i] = states[i]["q"]
            wind_history[step, i] = states[i]["wind"]

    print("Simulation finished.")
    for i in range(NUM_UAVS):
        print(
            f"UAV {i}: start {position_history[0, i]} -> "
            f"end {position_history[-1, i]}"
        )

    rotation_history = (
        Rotation.from_quat(quat_history.reshape(-1, 4))
        .as_matrix()
        .reshape(num_steps, NUM_UAVS, 3, 3)
    )

    return {
        "time": time_history,
        "positions": position_history,
        "rotations": rotation_history,
        "wind": wind_history,
        "leader_history": leader_history,
        "failed_history": failed_history,
        "promotion_time": promotion_time,
    }


def display_results(results):
    """Show the formation paths and leader-promotion animation."""
    world = World.empty((-8, 4, -3, 3, -2, 3))

    fig_3d = plt.figure("Leader Promotion On Straight Path")
    ax = fig_3d.add_subplot(projection="3d")
    world.draw(ax)
    for i in range(NUM_UAVS):
        path = results["positions"][:, i, :]
        label = "UAV 0 (failed leader)" if i == 0 else f"UAV {i}"
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], ".", label=label)

    promotion_step = np.flatnonzero(
        results["leader_history"] != results["leader_history"][0]
    )[0]
    promoted_leader = results["leader_history"][promotion_step]
    promotion_position = results["positions"][promotion_step, promoted_leader]
    ax.scatter(
        *promotion_position,
        color="green",
        marker="*",
        s=100,
        label=f"UAV {promoted_leader} promoted",
    )
    ax.legend()

    ani = animate(
        results["time"],
        results["positions"],
        results["rotations"],
        results["wind"],
        animate_wind=False,
        world=world,
    )

    plt.show()
    return ani


def main():
    """Run the simulation and open its animation."""
    results = run_simulation()
    print(f"Time steps: {len(results['time'])}")
    print("Opening leader-promotion animation...")
    display_results(results)


if __name__ == "__main__":
    main()
