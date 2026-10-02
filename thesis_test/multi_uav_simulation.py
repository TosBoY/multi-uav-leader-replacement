import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.wind.default_winds import NoWind

from follower_trajectory import FollowerTrajectory


def create_uav(initial_position, trajectory):
    initial_state = {
        "x": np.array(initial_position, dtype=float),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.array(
            [1788.53, 1788.53, 1788.53, 1788.53]
        ),
    }

    vehicle = Multirotor(
        quad_params,
        initial_state=initial_state,
    )

    controller = SE3Control(quad_params)

    sim = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=trajectory,
        wind_profile=NoWind(),
        sim_rate=100,
    )

    return sim


def animate_formation(results):
    """
    Animate all UAVs together in one 3D window.
    """

    # Extract position histories.
    positions = [
        result["state"]["x"]
        for result in results
    ]

    # All simulations should have the same number of frames.
    num_frames = min(len(p) for p in positions)

    positions = [
        p[:num_frames]
        for p in positions
    ]

    # Combine all positions to determine plot limits.
    all_positions = np.concatenate(positions, axis=0)

    min_xyz = np.min(all_positions, axis=0)
    max_xyz = np.max(all_positions, axis=0)

    # Add some padding around the formation.
    padding = 1.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("5-UAV Leader-Follower Formation")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_xlim(
        min_xyz[0] - padding,
        max_xyz[0] + padding
    )

    ax.set_ylim(
        min_xyz[1] - padding,
        max_xyz[1] + padding
    )

    ax.set_zlim(
        min_xyz[2] - padding,
        max_xyz[2] + padding
    )

    # Create one point for each UAV.
    points = []

    for i in range(len(positions)):
        point, = ax.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            label=f"UAV {i}"
        )

        points.append(point)

    # Create trails showing where each UAV has flown.
    trails = []

    for i in range(len(positions)):
        trail, = ax.plot(
            [],
            [],
            [],
            linewidth=1
        )

        trails.append(trail)

    ax.legend()

    def update(frame):
        for i in range(len(positions)):
            x = positions[i][frame, 0]
            y = positions[i][frame, 1]
            z = positions[i][frame, 2]

            points[i].set_data([x], [y])
            points[i].set_3d_properties([z])

            # Show the path travelled so far.
            trails[i].set_data(
                positions[i][:frame + 1, 0],
                positions[i][:frame + 1, 1]
            )

            trails[i].set_3d_properties(
                positions[i][:frame + 1, 2]
            )

        ax.set_title(
            f"5-UAV Leader-Follower Formation "
            f"| Time: {frame / 100:.2f} s"
        )

        return points + trails

    animation = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=10,
        blit=False,
        repeat=False,
    )

    plt.show()


def main():

    formation_positions = [
        [2.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 1.0],
    ]

    # Leader trajectory.
    leader_trajectory = ThreeDCircularTraj(
        center=np.array([0.0, 0.0, 1.0]),
        radius=np.array([2.0, 2.0, 0.0]),
        freq=np.array([0.1, 0.1, 0.0]),
        yaw_traj="forward",
    )

    # Create the five UAV simulations.
    uavs = []

    for i, position in enumerate(formation_positions):

        if i == 0:
            # UAV 0 is the leader.
            trajectory = leader_trajectory

        else:
            # Followers stay at fixed offsets behind the leader.
            offset = np.array(
                [-float(i), 0.0, 0.0]
            )

            trajectory = FollowerTrajectory(
                leader_trajectory,
                offset=offset
            )

        uavs.append(
            create_uav(position, trajectory)
        )

    print(f"Created {len(uavs)} UAVs.")

    # Run each RotorPy simulation.
    results = []

    for i, uav in enumerate(uavs):

        print(f"Simulating UAV {i}...")

        result = uav.run(
            t_final=10,
            plot=False,
            animate_bool=False,
            verbose=False,
        )

        results.append(result)

        final_position = result["state"]["x"][-1]

        print(
            f"UAV {i} finished: "
            f"{len(result['time'])} time steps"
        )

        print(
            f"  Initial position: "
            f"{formation_positions[i]}"
        )

        print(
            f"  Final position:   "
            f"{final_position}"
        )

    print("All UAV simulations finished.")

    # Open one animation containing all five UAVs.
    print("Opening formation animation...")

    animate_formation(results)


if __name__ == "__main__":
    main()