"""Simulate five drones in one synchronized RotorPy loop."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.utils.animate import animate
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.wind.default_winds import NoWind
from rotorpy.world import World

SIM_RATE = 100
DT = 1.0 / SIM_RATE
T_FINAL = 10.0
NUM_UAVS = 5

# Start the five drones in a line along X at 1 m altitude.
INITIAL_POSITIONS = [
    np.array([2.0, 0.0, 1.0]),
    np.array([1.0, 0.0, 1.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([-1.0, 0.0, 1.0]),
    np.array([-2.0, 0.0, 1.0]),
]


def make_initial_state(position):
    return {
        "x": np.array(position, dtype=float),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }


def make_trajectory(center_xy):
    """Each drone flies a circle around its own start XY at z = 1 m."""
    return ThreeDCircularTraj(
        center=np.array([center_xy[0], center_xy[1], 1.0]),
        radius=np.array([1.5, 1.5, 0.0]),
        freq=np.array([0.1, 0.1, 0.0]),
        yaw_traj="forward",
    )


def run_simulation():
    wind_profile = NoWind()
    vehicles = []
    controllers = []
    trajectories = []
    states = []

    for position in INITIAL_POSITIONS:
        initial_state = make_initial_state(position)
        vehicles.append(Multirotor(quad_params, initial_state=initial_state))
        controllers.append(SE3Control(quad_params))
        trajectories.append(make_trajectory(position))
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

    print(f"Created {NUM_UAVS} RotorPy vehicles.")
    print("Running synchronized 5-drone simulation...")

    for step in range(1, num_steps):
        t = step * DT
        time_history[step] = t

        controls = []
        for i in range(NUM_UAVS):
            flat_output = trajectories[i].update(t)
            control = controllers[i].update(t, states[i], flat_output)
            controls.append(control)

        for i in range(NUM_UAVS):
            states[i]["wind"] = wind_profile.update(t, states[i]["x"])
            states[i] = vehicles[i].step(states[i], controls[i], DT)
            position_history[step, i] = states[i]["x"]
            quat_history[step, i] = states[i]["q"]
            wind_history[step, i] = states[i]["wind"]

    print("Simulation finished.")
    for i in range(NUM_UAVS):
        print(
            f"UAV {i}: start {INITIAL_POSITIONS[i]} -> "
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
    """Show RotorPy's 3D quadrotor animation and path plot."""
    world = World.empty((-6, 6, -6, 6, -1, 4))

    fig_3d = plt.figure("3D Path")
    ax = fig_3d.add_subplot(projection="3d")
    world.draw(ax)
    for i in range(NUM_UAVS):
        path = results["positions"][:, i, :]
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], ".", label=f"UAV {i}")
    ax.legend()

    # Keep this reference so matplotlib does not garbage-collect the animation.
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
    print("Opening RotorPy animation...")
    display_results(results)


if __name__ == "__main__":
    main()
