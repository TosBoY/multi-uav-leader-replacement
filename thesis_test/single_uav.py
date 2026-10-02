import numpy as np

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.wind.default_winds import NoWind


def main():
    # Create the UAV
    vehicle = Multirotor(quad_params)

    # Create the controller
    controller = SE3Control(quad_params)

    # Create a simple hover trajectory
    trajectory = HoverTraj()

    # Create the RotorPy simulation environment
    sim = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=trajectory,
        wind_profile=NoWind(),
        sim_rate=100,
    )

    # Initial UAV state
    sim.vehicle.initial_state = {
        "x": np.array([0.0, 0.0, 0.0]),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.array(
            [1788.53, 1788.53, 1788.53, 1788.53]
        ),
    }

    # Run the simulation
    results = sim.run(
        t_final=10,
        plot=True,
        animate_bool=True,
        verbose=True,
    )

    print("Simulation finished.")
    print(f"Number of time steps: {len(results['time'])}")


if __name__ == "__main__":
    main()