"""Simulate five drones following synchronized Lissajous paths with RotorPy."""

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

PATH_X_AMPLITUDE = 2.5
PATH_Y_AMPLITUDE = 1.5
PATH_X_FREQUENCY = 0.6
PATH_Y_FREQUENCY = 1.2
PATH_HEIGHT = 1.0

# Each UAV is a different distance along the same geometric path.
INITIAL_PHASES = np.array([0.0, -0.8, -1.6, -2.4, -3.2])


class PhaseShiftedTrajectory:
    """Evaluate one shared trajectory at a phase-shifted time."""

    def __init__(self, trajectory, phase):
        self.trajectory = trajectory
        self.phase = phase

    def update(self, t):
        return self.trajectory.update(t + self.phase)


def make_initial_state(phase):
    position = make_lissajous_trajectory(phase).update(0.0)["x"]
    return {
        "x": np.array(position, dtype=float),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }


def make_lissajous_trajectory(phase):
    """Each drone follows the same 2:1 XY pattern at a different phase."""
    trajectory = TwoDLissajous(
        A=PATH_X_AMPLITUDE,
        B=PATH_Y_AMPLITUDE,
        a=PATH_X_FREQUENCY,
        b=PATH_Y_FREQUENCY,
        delta=0.0,
        x_offset=0.0,
        y_offset=0.0,
        height=PATH_HEIGHT,
        yaw_traj="forward",
    )
    return PhaseShiftedTrajectory(trajectory, phase)


def run_simulation():
    wind_profile = NoWind()
    vehicles = []
    controllers = []
    trajectories = []
    states = []

    for phase in INITIAL_PHASES:
        initial_state = make_initial_state(phase)
        vehicles.append(Multirotor(quad_params, initial_state=initial_state))
        controllers.append(SE3Control(quad_params))
        trajectories.append(make_lissajous_trajectory(phase))
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
    print("Running synchronized 5-drone Lissajous simulation...")

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
    """Show RotorPy's 3D quadrotor animation and path plot."""
    world = World.empty((-8, 8, -4, 4, -1, 4))

    fig_3d = plt.figure("3D Path")
    ax = fig_3d.add_subplot(projection="3d")
    world.draw(ax)
    for i in range(NUM_UAVS):
        path = results["positions"][:, i, :]
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], ".", label=f"UAV {i}")
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
    print("Opening RotorPy animation...")
    display_results(results)


if __name__ == "__main__":
    main()
