"""Check feasibility against a trajectory with multiple sharp turns."""

import unittest
import sys
from pathlib import Path

import numpy as np
from rotorpy.trajectories.minsnap import MinSnap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.run_analysis.check_feasibility import check_feasibility
from formation.formation_maker import make_formation
from simulation.sim_runner import run_simulation


class TestSharpTurnFeasibility(unittest.TestCase):
    """Run a normal five-drone simulation against a sharp-turn trajectory."""

    def test_sharp_turn_trajectory(self):
        drones = [0, 1, 2, 3, 4]
        initial_leader = 0
        leader_start = np.array([0.0, 0.0, 1.0])
        initial_positions = make_formation(
            number_of_drones=len(drones),
            formation_type="leader_follower",
            shape="arrow",
            spread=1.0,
            dimension=2,
            plane="xy",
            leader_index=initial_leader,
            leader_position=leader_start,
        )
        waypoints = np.array(
            [
                leader_start,
                [2.0, 0.0, 1.0],
                [2.0, 2.0, 1.0],
                [0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0],
                [-2.0, 0.0, 1.0],
                [-2.0, -2.0, 1.0],
            ]
        )
        leader_path = MinSnap(waypoints, v_avg=1, verbose=False)
        initial_positions = {
            drone: position.tolist()
            for drone, position in initial_positions.items()
        }

        run_simulation(
            drones=drones,
            initial_leader=initial_leader,
            initial_positions=initial_positions,
            leader_path=leader_path,
            leader_failure_time=None,
            election_type="closest",
            route_aligned_formation=True,
            sim_rate=100,
            t_final=10.0,
            display=True,
            save_data=True,
            save_data_filename="nonfail/sharp_turn_run.parquet",
            print_path=True,
            print_path_filename="sharp_turn_trajectory.csv",
        )

        result = check_feasibility(
            "sharp_turn_run.parquet",
            "sharp_turn_trajectory.csv",
        )
        print(f"sharp-turn feasibility result: {result}")
        self.assertEqual(result["run_file"], "sharp_turn_run.parquet")
        self.assertEqual(result["trajectory_file"], "sharp_turn_trajectory.csv")
        self.assertIn(result["result"], {"feasible", "infeasible"})


if __name__ == "__main__":
    unittest.main()
