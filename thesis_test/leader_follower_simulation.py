import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj


# ============================================================
# Simulation settings
# ============================================================

SIM_RATE = 100
DT = 1.0 / SIM_RATE
T_FINAL = 10.0

NUM_UAVS = 5


# ============================================================
# Formation
# ============================================================

FORMATION_OFFSETS = [
    np.array([0.0, 0.0, 0.0]),     # Leader
    np.array([-1.0, -1.0, 0.0]),    # Follower 1
    np.array([-2.0, -2.0, 0.0]),    # Follower 2
    np.array([1.0, 1.0, 0.0]),    # Follower 3
    np.array([2.0, 2.0, 0.0]),    # Follower 4
]


# ============================================================
# Create UAV
# ============================================================

def create_uav(initial_position):
    """
    Create one RotorPy vehicle and controller.

    The state is stored separately because Multirotor does not
    expose the state through a .state attribute.
    """

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

    return vehicle, controller, initial_state


# ============================================================
# Leader trajectory
# ============================================================

def create_leader_trajectory():
    """
    Create the planned trajectory for the leader.
    """

    return ThreeDCircularTraj(
        center=np.array([0.0, 0.0, 2.0]),
        radius=np.array([5.0, 5.0, 0.0]),
        freq=np.array([0.1, 0.1, 0.0]),
        yaw_traj="forward",
    )


# ============================================================
# Simulation
# ============================================================

def run_simulation():
    """
    Run all UAVs in one synchronized simulation loop.
    """

    # --------------------------------------------------------
    # Initial formation
    # --------------------------------------------------------

    leader_start = np.array(
        [5.0, 0.0, 2.0]
    )

    vehicles = []
    controllers = []
    states = []

    for i in range(NUM_UAVS):

        initial_position = (
            leader_start
            + FORMATION_OFFSETS[i]
        )

        vehicle, controller, initial_state = create_uav(
            initial_position
        )

        vehicles.append(vehicle)
        controllers.append(controller)
        states.append(initial_state)

    leader_trajectory = create_leader_trajectory()

    # --------------------------------------------------------
    # History arrays
    # --------------------------------------------------------

    num_steps = int(T_FINAL * SIM_RATE) + 1

    time_history = np.zeros(num_steps)

    position_history = np.zeros(
        (num_steps, NUM_UAVS, 3)
    )

    velocity_history = np.zeros(
        (num_steps, NUM_UAVS, 3)
    )

    # Store initial state.
    for i in range(NUM_UAVS):

        position_history[0, i] = states[i]["x"]

        velocity_history[0, i] = states[i]["v"]

    print(f"Created {NUM_UAVS} UAVs.")
    print(
        "Starting synchronized leader-follower simulation..."
    )

    # ========================================================
    # Main simulation loop
    # ========================================================

    for step in range(1, num_steps):

        t = step * DT

        time_history[step] = t

        # ----------------------------------------------------
        # 1. Get leader's planned trajectory.
        # ----------------------------------------------------

        leader_flat = leader_trajectory.update(t)

        # ----------------------------------------------------
        # 2. Read the leader's ACTUAL simulated state.
        # ----------------------------------------------------

        leader_position = states[0]["x"].copy()

        leader_velocity = states[0]["v"].copy()

        # ----------------------------------------------------
        # 3. Calculate controller commands.
        # ----------------------------------------------------

        controls = []

        for i in range(NUM_UAVS):

            if i == 0:

                # ============================================
                # LEADER
                # ============================================

                flat_output = leader_flat

            else:

                # ============================================
                # FOLLOWER
                # ============================================

                # Desired position is based on the leader's
                # ACTUAL current position.
                desired_position = (
                    leader_position
                    + FORMATION_OFFSETS[i]
                )

                # Desired velocity follows the leader.
                desired_velocity = leader_velocity

                flat_output = {
                    "x": desired_position,
                    "x_dot": desired_velocity,
                    "x_ddot": leader_flat["x_ddot"],
                    "x_dddot": leader_flat["x_dddot"],
                    "x_ddddot": leader_flat["x_ddddot"],
                    "yaw": leader_flat["yaw"],
                    "yaw_dot": leader_flat["yaw_dot"],
                    "yaw_ddot": leader_flat["yaw_ddot"],
                }

            # Controller calculates motor commands.
            control = controllers[i].update(
                t,
                states[i],
                flat_output,
            )

            controls.append(control)

        # ----------------------------------------------------
        # 4. Advance ALL UAV dynamics.
        # ----------------------------------------------------

        for i in range(NUM_UAVS):

            states[i] = vehicles[i].step(
                states[i],
                controls[i],
                DT,
            )

        # ----------------------------------------------------
        # 5. Save states.
        # ----------------------------------------------------

        for i in range(NUM_UAVS):

            position_history[step, i] = states[i]["x"]

            velocity_history[step, i] = states[i]["v"]

    print("Simulation finished.")

    # --------------------------------------------------------
    # Final positions
    # --------------------------------------------------------

    for i in range(NUM_UAVS):

        print(
            f"UAV {i} final position: "
            f"{position_history[-1, i]}"
        )

    return {
        "time": time_history,
        "positions": position_history,
        "velocities": velocity_history,
    }


# ============================================================
# Formation metrics
# ============================================================

def calculate_formation_error(results):
    """
    Calculate formation tracking error.

    Each follower's desired position is defined relative
    to the ACTUAL leader position.
    """

    positions = results["positions"]

    # Actual leader position over the whole simulation.
    leader_positions = positions[:, 0, :]

    errors = []

    for i in range(1, NUM_UAVS):

        # Desired follower position.
        desired_positions = (
            leader_positions
            + FORMATION_OFFSETS[i]
        )

        # Actual follower position.
        actual_positions = positions[:, i, :]

        # Position error vector.
        error = (
            actual_positions
            - desired_positions
        )

        errors.append(error)

    errors = np.array(errors)

    # --------------------------------------------------------
    # Convert XYZ error to Euclidean error magnitude.
    # --------------------------------------------------------

    error_magnitude = np.linalg.norm(
        errors,
        axis=2
    )

    # --------------------------------------------------------
    # Overall RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        np.mean(error_magnitude ** 2)
    )

    # --------------------------------------------------------
    # Maximum error
    # --------------------------------------------------------

    max_error = np.max(
        error_magnitude
    )

    # --------------------------------------------------------
    # Final error
    # --------------------------------------------------------

    final_error = error_magnitude[:, -1]

    print()
    print(
        "================ FORMATION METRICS ================"
    )

    print(
        f"Overall formation RMSE: "
        f"{rmse:.4f} m"
    )

    print(
        f"Maximum formation error: "
        f"{max_error:.4f} m"
    )

    print(
        f"Final formation error: "
        f"{np.mean(final_error):.4f} m"
    )

    # --------------------------------------------------------
    # Individual follower RMSE
    # --------------------------------------------------------

    for i in range(NUM_UAVS - 1):

        follower_rmse = np.sqrt(
            np.mean(
                error_magnitude[i] ** 2
            )
        )

        print(
            f"Follower {i + 1} RMSE: "
            f"{follower_rmse:.4f} m"
        )

    print(
        "==================================================="
    )
    print()


# ============================================================
# Animation
# ============================================================

def animate_results(results):
    """
    Display all UAVs simultaneously in one 3D animation.
    """

    time = results["time"]

    positions = results["positions"]

    # --------------------------------------------------------
    # Determine plot limits.
    # --------------------------------------------------------

    all_positions = positions.reshape(
        -1,
        3
    )

    min_xyz = np.min(
        all_positions,
        axis=0
    )

    max_xyz = np.max(
        all_positions,
        axis=0
    )

    padding = 1.0

    # --------------------------------------------------------
    # Create figure.
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_xlim(
        min_xyz[0] - padding,
        max_xyz[0] + padding,
    )

    ax.set_ylim(
        min_xyz[1] - padding,
        max_xyz[1] + padding,
    )

    ax.set_zlim(
        min_xyz[2] - padding,
        max_xyz[2] + padding,
    )

    # --------------------------------------------------------
    # Create UAV markers and trails.
    # --------------------------------------------------------

    points = []
    trails = []

    for i in range(NUM_UAVS):

        label = (
            "Leader"
            if i == 0
            else f"Follower {i}"
        )

        point, = ax.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            label=label,
        )

        trail, = ax.plot(
            [],
            [],
            [],
            linewidth=1,
        )

        points.append(point)
        trails.append(trail)

    ax.legend()

    # --------------------------------------------------------
    # Animation update function.
    # --------------------------------------------------------

    def update(frame):

        for i in range(NUM_UAVS):

            current = positions[
                frame,
                i
            ]

            # UAV marker.
            points[i].set_data(
                [current[0]],
                [current[1]],
            )

            points[i].set_3d_properties(
                [current[2]]
            )

            # Flight trail.
            path = positions[
                :frame + 1,
                i
            ]

            trails[i].set_data(
                path[:, 0],
                path[:, 1],
            )

            trails[i].set_3d_properties(
                path[:, 2]
            )

        ax.set_title(
            "Leader-Follower Formation "
            f"| Time: {time[frame]:.2f} s"
        )

        return points + trails

    # --------------------------------------------------------
    # Create animation.
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=len(time),
        interval=10,
        blit=False,
        repeat=False,
    )

    # Keep animation object alive.
    _ = animation

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    results = run_simulation()

    calculate_formation_error(results)

    print(
        "Opening formation animation..."
    )

    animate_results(results)


if __name__ == "__main__":
    main()