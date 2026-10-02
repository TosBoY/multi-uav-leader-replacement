"""Simulate random leader promotion after a straight-line leader failure.

This file is a deliberately commented copy of the original simulation logic in
`test_rotorpy_leader_promotion_line.py`. The purpose of this version is not to
change the dynamics or the controller setup, but to make the full workflow easier
for a reader to follow.

The complete scenario is:
1. A team of UAVs flies in a formation along a straight-line reference path.
2. The original leader is taken out at a fixed time.
3. One surviving UAV is selected randomly to become the new leader.
4. The remaining followers keep their relative formation slots instead of
   collapsing into the former leader's old location.

This detailed walkthrough is intended to explain both the mathematical and the
programmatic intent behind each piece of the script.
"""

import random

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

# -----------------------------------------------------------------------------
# Simulation configuration.
# These constants define the time discretization, total simulation duration, and
# the logic used to trigger a leadership handover. No functional behavior is
# changed by the comments; they are here to explain why each value is chosen.
# -----------------------------------------------------------------------------
SIM_RATE = 100
DT = 1.0 / SIM_RATE
T_FINAL = 10.0
NUM_UAVS = 5
LEADER_FAILURE_TIME = T_FINAL / 2.0
RANDOM_SEED = 42

# The leader starts at a point slightly in front of the formation and then moves
# forward along the x-axis. The follower offsets are defined in a local frame
# relative to the current leader, so that the formation behaves like an arrow or
# V-shaped flight arrangement while still being controlled by a single leader.
#
# In this setup:
# - the leader occupies the front-most position;
# - the first two drones sit near the side wings;
# - the final two drones sit farther back and out to the sides;
# - every follower maintains a fixed role relative to the leader instead of
#   reordering itself into the leader's former spot.
LEADER_INITIAL_POSITION = np.array([-2.0, 0.0, 1.0])
FOLLOWER_SPACING = 1.0
'''
FORMATION_POSITIONS = (
    (0.0, 0.0),   # Leader at the arrow tip.
    (0.8, -1.0),  # Near left wing.
    (0.8, 1.0),   # Near right wing.
    (1.6, -2.0),  # Rear left wing.
    (1.6, 2.0),   # Rear right wing.
)
'''
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
    """Create the initial RotorPy state dictionary for one UAV.

    RotorPy uses a dictionary-based state object rather than a custom class. Each
    entry describes a physical quantity that the simulator can advance step by
    step. This helper keeps all UAVs in a consistent starting configuration so we
    do not accidentally mix state layouts between vehicles.

    The important fields are:
    - "x": position in world coordinates
    - "v": linear velocity
    - "q": orientation quaternion
    - "w": angular velocity
    - "wind": current local wind vector
    - "rotor_speeds": motor speeds required by the multirotor model
    """
    return {
        # Position is a 3D vector [x, y, z].
        "x": np.array(position, dtype=float),
        # Linear velocity starts at zero before the controller begins tracking.
        "v": np.zeros(3),
        # The UAV starts level with a unit quaternion [x, y, z, w].
        # A quaternion of [0, 0, 0, 1] represents zero rotation.
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        # Angular velocity starts from rest.
        "w": np.zeros(3),
        # Environment wind is zero for this scenario.
        "wind": np.zeros(3),
        # Rotor speeds are initialized at a nominal hover value for the
        # Crazyflie-like vehicle used by RotorPy.
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }


def make_leader_trajectory():
    """Build the trajectory object that defines the leader's motion.

    This function returns the path generator used by whoever is currently acting as
    the active leader. In this experiment, the leader reference is always a
    straight line along the x-axis with a constant forward speed.

    The important parameters are:
    - init_pos: where the path begins
    - dist: total distance the leader travels
    - speed: linear motion rate along the path
    - axis: which axis defines the path direction
    - yaw_traj: how yaw changes while following the route

    This object produces data like position, velocity, acceleration, and yaw, all
    in the format expected by the SE3 controller.
    """
    return ConstantSpeed(
        init_pos=LEADER_INITIAL_POSITION,
        dist=LINE_DISTANCE,
        speed=LINE_SPEED,
        axis=LINE_AXIS,
        yaw_traj="forward",
    )


def make_path_flat_output(path_flat, position):
    """Convert a trajectory's flat output into a desired target state.

    RotorPy's trajectory objects often expose a flat-output dictionary containing
    the desired motion variables for a controller. This helper keeps the same
    derivatives and yaw information as the leader path, but swaps in a different
    target location. In other words, the follower is not asked to follow the
    exact leader state; instead, it is told to track a nearby offset point while
    still inheriting the leader's velocity and heading behavior.

    The returned dictionary is shaped exactly like the controller expects:
    - x: desired position
    - x_dot: desired velocity
    - x_ddot: desired acceleration
    - x_dddot: desired jerk
    - x_ddddot: desired snap
    - yaw / yaw_dot / yaw_ddot: orientation trajectory
    """
    return {
        # Replace the leader position with the follower's desired offset target.
        "x": position,
        # Retain the leader's translational derivatives so the follower tracks the
        # same underlying motion style, just from a shifted point.
        "x_dot": path_flat["x_dot"],
        "x_ddot": path_flat["x_ddot"],
        "x_dddot": path_flat["x_dddot"],
        "x_ddddot": path_flat["x_ddddot"],
        # Keep the same desired heading profile as the leader.
        "yaw": path_flat["yaw"],
        "yaw_dot": path_flat["yaw_dot"],
        "yaw_ddot": path_flat["yaw_ddot"],
    }


def make_following_flat_output(leader_flat, leader_position, rank):
    """Compute a follower's target position relative to the current leader.

    Each follower has a fixed formation role (`rank`) that defines where it should
    sit relative to the leader. The leader is the front reference point, while the
    followers are offset behind it and to the left/right. This preserves a stable
    formation instead of creating independent trajectories for each aircraft.

    The geometry is encoded in `FORMATION_POSITIONS`, where the first value is the
    forward/backward offset and the second value is the lateral offset. We then
    translate the leader's current position by that offset to get the target spot
    for the follower.
    """
    behind, lateral = FORMATION_POSITIONS[rank]
    desired_position = leader_position + np.array(
        [-behind * FOLLOWER_SPACING, lateral * FOLLOWER_SPACING, 0.0]
    )
    return make_path_flat_output(leader_flat, desired_position)


def run_simulation():
    """Run the leader failover simulation and return recorded state history.

    This function is the heart of the experiment. It initializes every UAV,
    assigns the initial leader, runs the time-stepping loop, and handles the
    transition when the leader fails. The end result is a structured dictionary
    containing the trajectories, quaternion history, leader transitions, and the
    exact time at which promotion happened.

    The central idea is that a single global leader path is used for the entire
    mission, while each aircraft is assigned an offset relative to the current
    leader. When the leader fails, the code does not rebuild the formation from
    scratch; instead, it promotes a surviving drone and keeps the others at their
    original formation slots.
    """
    # No external wind is used in this experiment; the environment is calm.
    # The wind profile object is still included to keep the simulation realistic
    # and to match the interface expected by RotorPy's vehicle stepping function.
    wind_profile = NoWind()
    # One path describes the motion of the current leader, regardless of which
    # drone physically carries the leader role. This is important because the
    # formation should follow the same route even after the identity of the leader
    # changes.
    leader_trajectory = make_leader_trajectory()

    # Each UAV starts in a position based on its formation rank. The leader is
    # placed at the front, while followers are offset behind and to the sides.
    # Note that this initial offset is computed relative to `LEADER_INITIAL_POSITION`
    # and not relative to a unique drone-specific path, so all UAVs share a common
    # formation reference frame.
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

    # Create one rotorpy vehicle and one controller per UAV. The controller is
    # responsible for mapping the desired flat-output trajectory into rotor
    # commands, while the vehicle integrates the dynamics forward in time.
    vehicles = []
    controllers = []
    states = []
    for position in initial_positions:
        initial_state = make_initial_state(position)
        vehicles.append(Multirotor(quad_params, initial_state=initial_state))
        controllers.append(SE3Control(quad_params))
        states.append(initial_state)

    # Allocate arrays to store full trajectory histories for time, positions,
    # orientations, wind, and the leader/failure events. These arrays are used
    # later to animate the flight and inspect the result.
    num_steps = int(T_FINAL * SIM_RATE) + 1
    time_history = np.zeros(num_steps)
    position_history = np.zeros((num_steps, NUM_UAVS, 3))
    quat_history = np.zeros((num_steps, NUM_UAVS, 4))
    wind_history = np.zeros((num_steps, NUM_UAVS, 3))
    leader_history = np.full(num_steps, -1, dtype=int)
    failed_history = np.zeros((num_steps, NUM_UAVS), dtype=bool)

    # Store the initial state for all agents before the simulation loop begins.
    # This guarantees that the first timestep contains the known initial values
    # rather than an uninitialized state.
    for i in range(NUM_UAVS):
        position_history[0, i] = states[i]["x"]
        quat_history[0, i] = states[i]["q"]
        wind_history[0, i] = states[i]["wind"]

    # active_order tracks the current alive UAVs in their leader/follower order.
    # formation_ranks maps each UAV to its permanent formation slot.
    #
    # This distinction matters: `active_order` determines which drones are still
    # included in the control loop, while `formation_ranks` tells each drone where
    # it should sit relative to the leader. The latter is intentionally preserved
    # across promotion so the formation does not reconfigure or shuffle.
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

        # -----------------------------------------------------------------
        # Leader promotion logic.
        # A leader failover occurs at the fixed time threshold. The initial leader
        # is removed from the active set and one of the surviving vehicles is
        # selected at random as the new leader. This random selection is seeded so
        # the scenario remains reproducible across runs.
        #
        # The purpose of this logic is not to teleport the formation or to assign
        # the promoted drone to the former leader slot; instead, the promoted drone
        # simply becomes the new head of the formation while the others remain in
        # their original relative positions.
        # -----------------------------------------------------------------
        if active_leader == 0 and t >= LEADER_FAILURE_TIME:
            failed_drones.add(active_leader)
            active_order.remove(active_leader)
            active_leader = promotion_rng.choice(active_order)
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

        # Record the active leader at each timestep so the final result can be
        # analyzed after the mission is complete.
        leader_history[step] = active_leader
        for failed_index in failed_drones:
            failed_history[step, failed_index] = True

        # The leader's trajectory is evaluated for the current time. The leader
        # is always commanded to follow this trajectory, regardless of which UAV
        # currently holds the leader role. This keeps the path reference globally
        # consistent and makes the role swap a matter of control assignment, not a
        # path redesign.
        path_flat = leader_trajectory.update(t)
        leader_position = states[active_leader]["x"].copy()
        controls = [None] * NUM_UAVS

        # Control the active vehicles. Each surviving drone either follows the
        # current leader trajectory directly (if it is the leader) or tracks a
        # formation offset relative to the leader position.
        #
        # This is the key formation behavior: followers do not each compute their
        # own unique path; they all hold a consistent offset relative to the head
        # of the formation. This is what preserves the arrow-like shape during the
        # mission and after promotion.
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

        # Advance each UAV one timestep. Failed drones are not integrated; they
        # are simply marked as missing in the history arrays. This means the old
        # leader remains visible in the recorded data as a failed aircraft rather
        # than being physically simulated after its loss.
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

    # Convert quaternion arrays into rotation matrices for animation and output.
    # This is necessary because the animation utility expects rotation matrices,
    # while the simulation state stores quaternions internally.
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
    """Visualize the motion with a 3D plot and animation.

    This function is not part of the dynamic simulation itself; it is a plotting
    layer. It takes the final recorded results and converts them into a visual
    representation that makes the formation evolution and failover obvious.

    The plotted paths are useful because they reveal:
    - how the participants are arranged while moving forward;
    - where the leader handover occurs;
    - whether the followers remain in their relative slots after promotion;
    - whether the old leader remains in the history as a failed aircraft.
    """
    world = World.empty((-8, 4, -3, 3, -2, 3))

    fig_3d = plt.figure("Leader Promotion On Straight Path")
    ax = fig_3d.add_subplot(projection="3d")
    world.draw(ax)
    for i in range(NUM_UAVS):
        path = results["positions"][:, i, :]
        label = "UAV 0 (failed leader)" if i == 0 else f"UAV {i}"
        ax.plot3D(path[:, 0], path[:, 1], path[:, 2], ".", label=label)

    # Determine the first frame in which the leader changes so we can mark where
    # the new leader took over the task. The logic checks the leader history for
    # the first index where it differs from the initial leader ID.
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
    """Entry point for the script.

    This is the standard entry function for running the experiment from the
    command line. It executes the simulation, prints summary information, and then
    opens the animation so the user can inspect the behavior.
    """
    results = run_simulation()
    print(f"Time steps: {len(results['time'])}")
    print("Opening leader-promotion animation...")
    display_results(results)


if __name__ == "__main__":
    main()
