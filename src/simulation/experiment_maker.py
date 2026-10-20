"""Build and run complete formation, path, and simulation experiments."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from election.election import ElectionType
from formation.formation_maker import FormationPlane, FormationShape, FormationType, make_formation
from pathing.path_maker import PathType, make_leader_path
from simulation.sim_runner import run_simulation
from utils.save_data import RunType


def make_experiment(
    number_of_drones: int = 5,
    initial_leader: int = 0,
    leader_position: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
    formation_type: FormationType = "leader_follower",
    shape: FormationShape = "arrow",
    spread: float = 1.0,
    dimension: int = 2,
    plane: FormationPlane = "xy",
    path_type: PathType = "line",
    velocity: float | Sequence[float] | np.ndarray = 1.0,
    simulation_time: float = 10.0,
    initial_heading: float = 0.0,
    path_seed: int | None = None,
    print_waypoints: bool = False,
    leader_failure_time: float | None = None,
    election_type: ElectionType = "random",
    print_election: bool = False,
    formation_offsets: Mapping[int, Sequence[float]] | None = None,
    route_aligned_formation: bool = True,
    sim_rate: int = 100,
    random_election_seed: int | None = None,
    display: bool | str = "no",
    run_type: RunType | None = None,
    save_data: bool = False,
    print_path: bool = False,
) -> dict[str, Any]:
    """Create a formation and path, then run one complete experiment.

    The arguments are grouped in the same order as the workflow used by the
    experiment tests: formation settings, path settings, and simulation
    settings.
    """
    if isinstance(number_of_drones, bool) or not isinstance(number_of_drones, int):
        raise TypeError("number_of_drones must be an integer.")
    if number_of_drones < 1:
        raise ValueError("number_of_drones must be at least one.")
    if isinstance(initial_leader, bool) or not isinstance(initial_leader, int):
        raise TypeError("initial_leader must be an integer.")
    if initial_leader < 0 or initial_leader >= number_of_drones:
        raise ValueError("initial_leader must identify one of the drones.")
    drone_ids = list(range(number_of_drones))
    leader_index = initial_leader

    generated_positions = make_formation(
        number_of_drones=number_of_drones,
        formation_type=formation_type,
        shape=shape,
        spread=spread,
        dimension=dimension,
        plane=plane,
        leader_index=leader_index,
        leader_position=leader_position,
    )
    initial_positions = {
        drone: generated_positions[index].tolist()
        for index, drone in enumerate(drone_ids)
    }

    leader_path = make_leader_path(
        path_type=path_type,
        first_waypoint=initial_positions[initial_leader],
        velocity=velocity,
        time=simulation_time,
        seed=path_seed,
        print_waypoints=print_waypoints,
        initial_heading=initial_heading,
    )

    return run_simulation(
        drones=drone_ids,
        initial_leader=initial_leader,
        initial_positions=initial_positions,
        leader_path=leader_path,
        leader_failure_time=leader_failure_time,
        election_type=election_type,
        print_election=print_election,
        formation_offsets=formation_offsets,
        route_aligned_formation=route_aligned_formation,
        sim_rate=sim_rate,
        t_final=simulation_time,
        random_election_seed=random_election_seed,
        display=display,
        run_type=run_type,
        save_data=save_data,
        print_path=print_path,
    )
