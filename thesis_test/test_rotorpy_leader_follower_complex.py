"""Simulate a complex-route leader-follower formation with RotorPy."""

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

LEADER_INITIAL_POSITION = np.array([0.0, 0.0, 1.0])
FOLLOWER_SPACING = 1.0
FORMATION_POSITIONS = (
    (0.0, 0.0),   # Leader at the arrow tip.
    (0.8, -1.0),  # Near left wing.
    (0.8, 1.0),   # Near right wing.
    (1.6, -2.0),  # Rear left wing.
    (1.6, 2.0),   # Rear right wing.
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


def get_formation_offset(leader_velocity, follower_index):
    """Return a route-aligned offset for the arrow formation."""
    direction = np.array(leader_velocity, dtype=float)
    direction[2] = 0.0
    speed = np.linalg.norm(direction)
    if speed < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= speed
    normal = np.array([-direction[1], direction[0], 0.0])
    behind, lateral = FORMATION_POSITIONS[follower_index]
    return -behind * FOLLOWER_SPACING * direction + lateral * FOLLOWER_SPACING * normal


def make_follower_flat_output(leader_flat, leader_position, follower_index):
    offset = get_formation_offset(leader_flat["x_dot"], follower_index)
    return {
        "x": leader_position + offset,
        "x_dot": leader_flat["x_dot"],
        "x_ddot": leader_flat["x_ddot"],
        "x_dddot": leader_flat["x_dddot"],
        "x_ddddot": leader_flat["x_ddddot"],
        "yaw": leader_flat["yaw"],
        "yaw_dot": leader_flat["yaw_dot"],
        "yaw_ddot": leader_flat["yaw_ddot"],
    }


def run_simulation():
    wind_profile = NoWind()
    leader_trajectory = make_leader_trajectory()
    leader_start = leader_trajectory.update(0.0)["x"]
    initial_positions = [leader_start]

    initial_velocity = leader_trajectory.update(0.0)["x_dot"]
    for i in range(1, NUM_UAVS):
        initial_positions.append(
            leader_start + get_formation_offset(initial_velocity, i)
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

    for i in range(NUM_UAVS):
        position_history[0, i] = states[i]["x"]
        quat_history[0, i] = states[i]["q"]
        wind_history[0, i] = states[i]["wind"]

    print(f"Created {NUM_UAVS} UAVs. UAV 0 is the leader.")
    print("Running complex-route leader-follower simulation...")

    for step in range(1, num_steps):
        t = step * DT
        time_history[step] = t

        leader_flat = leader_trajectory.update(t)
        leader_position = states[0]["x"].copy()
        controls = []

        for i in range(NUM_UAVS):
            if i == 0:
                flat_output = leader_flat
            else:
                flat_output = make_follower_flat_output(
                    leader_flat,
                    leader_position,
                    i,
                )
            controls.append(controllers[i].update(t, states[i], flat_output))

        for i in range(NUM_UAVS):
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
    }


def display_results(results):
    """Show the complex leader-follower animation and path plot."""
    world = World.empty((-6, 6, -4, 4, -1, 4))

    fig_3d = plt.figure("Complex Leader-Follower Path")
    ax = fig_3d.add_subplot(projection="3d")
    world.draw(ax)
    for i in range(NUM_UAVS):
        path = results["positions"][:, i, :]
        label = "Leader" if i == 0 else f"Follower {i}"
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], ".", label=label)
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
    print("Opening complex leader-follower animation...")
    display_results(results)


if __name__ == "__main__":
    main()
