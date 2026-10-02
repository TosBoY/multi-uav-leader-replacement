"""Simulate an arrow formation after the complex-route leader fails."""

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
    initial_velocity = leader_trajectory.update(0.0)["x_dot"]
    initial_positions = [leader_start]

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
    leader_failed_history = np.zeros(num_steps, dtype=bool)
    last_follower_outputs = {}
    leader_failed = False

    for i in range(NUM_UAVS):
        position_history[0, i] = states[i]["x"]
        quat_history[0, i] = states[i]["q"]
        wind_history[0, i] = states[i]["wind"]

    print(f"Created {NUM_UAVS} UAVs. UAV 0 is the leader.")
    print(f"Leader failure scheduled at t = {LEADER_FAILURE_TIME:.2f} s.")

    for step in range(1, num_steps):
        t = step * DT
        time_history[step] = t

        if not leader_failed and t >= LEADER_FAILURE_TIME:
            leader_failed = True
            print(f"Leader failure injected at t = {t:.2f} s.")

        leader_flat = leader_trajectory.update(t)
        leader_position = states[0]["x"].copy()
        controls = []

        for i in range(NUM_UAVS):
            if i == 0:
                flat_output = leader_flat
            elif leader_failed:
                # The followers have lost the leader and keep their last
                # known formation targets instead of tracking stale data.
                flat_output = last_follower_outputs[i].copy()
                flat_output["x_dot"] = np.zeros(3)
                flat_output["x_ddot"] = np.zeros(3)
                flat_output["x_dddot"] = np.zeros(3)
                flat_output["x_ddddot"] = np.zeros(3)
            else:
                flat_output = make_follower_flat_output(
                    leader_flat,
                    leader_position,
                    i,
                )
                last_follower_outputs[i] = flat_output
            controls.append(controllers[i].update(t, states[i], flat_output))

        if leader_failed:
            leader_failed_history[step] = True

        for i in range(NUM_UAVS):
            if i == 0 and leader_failed:
                # A failed leader is no longer observable in the simulation.
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
        "leader_failed": leader_failed_history,
    }


def display_results(results):
    """Show the failed-leader formation animation and path plot."""
    # Keep the camera focused on the route and formation instead of the
    # leader's long free-fall after the motor failure.
    world = World.empty((-6, 6, -5, 5, -3, 3))

    fig_3d = plt.figure("Complex Route With Leader Failure")
    ax = fig_3d.add_subplot(projection="3d")
    world.draw(ax)
    for i in range(NUM_UAVS):
        path = results["positions"][:, i, :]
        label = "Leader" if i == 0 else f"Follower {i}"
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], ".", label=label)
    ax.legend()

    failure_step = np.flatnonzero(results["leader_failed"])[0]
    failure_position = results["positions"][failure_step - 1, 0]
    ax.scatter(
        *failure_position,
        color="red",
        marker="x",
        s=80,
        label="Leader failure",
    )

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
    print("Opening failed-leader animation...")
    display_results(results)


if __name__ == "__main__":
    main()
