"""Simulate a straight-line leader-follower formation with RotorPy."""

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

SIM_RATE = 100
DT = 1.0 / SIM_RATE
T_FINAL = 10.0
NUM_UAVS = 5

LEADER_INITIAL_POSITION = np.array([-2.0, 0.0, 1.0])
FOLLOWER_SPACING = 1.0
FORMATION_OFFSETS = np.array(
    [[-i * FOLLOWER_SPACING, 0.0, 0.0] for i in range(NUM_UAVS)]
)
LINE_DISTANCE = 5.0
LINE_SPEED = 0.5
LINE_AXIS = 0


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
    return ConstantSpeed(
        init_pos=LEADER_INITIAL_POSITION,
        dist=LINE_DISTANCE,
        speed=LINE_SPEED,
        axis=LINE_AXIS,
        yaw_traj="forward",
    )


def make_follower_flat_output(leader_flat, leader_position, offset):
    """Track the leader's measured position while keeping a fixed offset."""
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
    vehicles = []
    controllers = []
    states = []

    for offset in FORMATION_OFFSETS:
        initial_state = make_initial_state(LEADER_INITIAL_POSITION + offset)
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
    print("Running straight-line leader-follower simulation...")

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
                    FORMATION_OFFSETS[i],
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
    """Show the leader-follower animation and path plot."""
    world = World.empty((-8, 4, -3, 3, -1, 4))

    fig_3d = plt.figure("Leader-Follower Straight Path")
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
    print("Opening leader-follower animation...")
    display_results(results)


if __name__ == "__main__":
    main()
