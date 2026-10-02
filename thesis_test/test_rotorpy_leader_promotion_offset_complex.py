"""Promote a leader onto an offset copy of a complex route with RotorPy."""

import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.utils.animate import animate
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.wind.default_winds import NoWind
from rotorpy.world import World

SIM_RATE = 100
DT = 1.0 / SIM_RATE
T_FINAL = 10.0
NUM_UAVS = 5
LEADER_FAILURE_TIME = T_FINAL / 2.0
RANDOM_SEED = 42

LEADER_INITIAL_POSITION = np.array([0.0, 0.0, 1.0])
FOLLOWER_SPACING = 1.0
FORMATION_POSITIONS = (
    (0.0, 0.0),
    (0.8, -1.0),
    (0.8, 1.0),
    (1.6, -2.0),
    (1.6, 2.0),
)
PATH_X_AMPLITUDE = 2.5
PATH_Y_AMPLITUDE = 1.5
PATH_X_FREQUENCY = 0.6
PATH_Y_FREQUENCY = 1.2
PATH_HEIGHT = 1.0


def make_initial_state(position):
    return {
        "x": np.array(position, dtype=float),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }


def make_leader_trajectory():
    return TwoDLissajous(
        A=PATH_X_AMPLITUDE,
        B=PATH_Y_AMPLITUDE,
        a=PATH_X_FREQUENCY,
        b=PATH_Y_FREQUENCY,
        delta=0.0,
        x_offset=LEADER_INITIAL_POSITION[0],
        y_offset=LEADER_INITIAL_POSITION[1],
        height=PATH_HEIGHT,
        yaw_traj="forward",
    )


def get_route_offset(leader_velocity, rank):
    """Return a route-aligned arrow offset for a formation rank."""
    direction = np.array(leader_velocity, dtype=float)
    direction[2] = 0.0
    speed = np.linalg.norm(direction)
    if speed < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= speed

    normal = np.array([-direction[1], direction[0], 0.0])
    behind, lateral = FORMATION_POSITIONS[rank]
    return -behind * FOLLOWER_SPACING * direction + lateral * FOLLOWER_SPACING * normal


def make_path_flat_output(path_flat, position, velocity=None):
    flat_output = {
        "x": position,
        "x_dot": path_flat["x_dot"],
        "x_ddot": path_flat["x_ddot"],
        "x_dddot": path_flat["x_dddot"],
        "x_ddddot": path_flat["x_ddddot"],
        "yaw": path_flat["yaw"],
        "yaw_dot": path_flat["yaw_dot"],
        "yaw_ddot": path_flat["yaw_ddot"],
    }
    if velocity is not None:
        flat_output["x_dot"] = np.array(velocity, dtype=float)
    return flat_output


def make_offset_path_flat_output(path_flat, path_offset):
    """Track the original Lissajous path translated by the computed offset."""
    return make_path_flat_output(path_flat, path_flat["x"] + path_offset)


def make_following_flat_output(path_flat, leader_position, leader_velocity, rank):
    """Follow the active leader while retaining the assigned arrow slot."""
    desired_position = leader_position + get_route_offset(leader_velocity, rank)
    return make_path_flat_output(
        path_flat,
        desired_position,
        velocity=leader_velocity,
    )


def run_simulation():
    wind_profile = NoWind()
    leader_trajectory = make_leader_trajectory()
    leader_start = leader_trajectory.update(0.0)["x"]
    initial_velocity = leader_trajectory.update(0.0)["x_dot"]
    initial_positions = [leader_start]

    for rank in range(1, NUM_UAVS):
        initial_positions.append(
            leader_start + get_route_offset(initial_velocity, rank)
        )

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
    promoted_path_offset = np.zeros(3)
    leader_history[0] = active_leader

    print(f"Created {NUM_UAVS} UAVs. UAV 0 is the initial leader.")
    print(f"Leader failure scheduled at t = {LEADER_FAILURE_TIME:.2f} s.")

    for step in range(1, num_steps):
        t = step * DT
        time_history[step] = t

        if active_leader == 0 and t >= LEADER_FAILURE_TIME:
            old_leader_position = states[active_leader]["x"].copy()
            failed_drones.add(active_leader)
            active_order.remove(active_leader)
            active_leader = promotion_rng.choice(active_order)
            promoted_path_offset = (
                states[active_leader]["x"].copy() - old_leader_position
            )
            active_order.remove(active_leader)
            active_order.insert(0, active_leader)
            formation_ranks[active_leader] = 0
            promotion_time = t
            print(
                f"UAV 0 failed at t = {t:.2f} s. "
                f"UAV {active_leader} promoted with path offset "
                f"{promoted_path_offset}."
            )

        leader_history[step] = active_leader
        for failed_index in failed_drones:
            failed_history[step, failed_index] = True

        path_flat = leader_trajectory.update(t)
        active_leader_position = states[active_leader]["x"].copy()
        active_leader_velocity = states[active_leader]["v"].copy()
        controls = [None] * NUM_UAVS

        for i in active_order:
            if i == active_leader:
                if active_leader == 0:
                    flat_output = path_flat
                else:
                    flat_output = make_offset_path_flat_output(
                        path_flat,
                        promoted_path_offset,
                    )
            else:
                flat_output = make_following_flat_output(
                    path_flat,
                    active_leader_position,
                    active_leader_velocity,
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
        "promoted_path_offset": promoted_path_offset,
    }


def display_results(results):
    """Show the complex offset-path promotion animation."""
    world = World.empty((-6, 6, -5, 5, -3, 3))

    fig_3d = plt.figure("Complex Offset Leader Promotion")
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
    results = run_simulation()
    print(f"Time steps: {len(results['time'])}")
    print("Opening complex offset-path promotion animation...")
    display_results(results)


if __name__ == "__main__":
    main()
