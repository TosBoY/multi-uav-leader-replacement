"""Test the reusable runner with the straight-line leader-promotion scenario."""

import unittest

import numpy as np

from simulation.sim_runner import run_simulation
from pathing.path_maker import make_leader_path


class TestSimRunnerLeaderPromotionLine(unittest.TestCase):
    """Verify the same five-drone scenario through the shared runner."""

    def test_path_maker_to_sim_runner(self):
        """Promote a random follower after the initial leader fails."""
        drones = [0, 1, 2, 3, 4]
        leader_start = np.array([0, 0.0, 1.0])
        initial_positions = {
            0: leader_start.tolist(),
            1: (leader_start + np.array([-1, -1, 0.0])).tolist(),
            2: (leader_start + np.array([-2, -2, 0.0])).tolist(),
            3: (leader_start + np.array([-1, 1, 0.0])).tolist(),
            4: (leader_start + np.array([-2, 2, 0.0])).tolist(),
        }
        leader_speed = 1
        leader_heading = 0.0
        simulation_time = 10.0
        leader_path = make_leader_path(
            path_type="random",
            first_waypoint=initial_positions[0],
            velocity=leader_speed,
            time=simulation_time,
            initial_heading=leader_heading,
            print_waypoints=True,
            seed=42,
        )

        results = run_simulation(
            drones=drones,
            initial_leader=0,
            initial_positions=initial_positions,
            leader_path=leader_path,
            leader_failure_time= None,
            election_type="closest",
            route_aligned_formation=True,
            sim_rate=100,
            t_final=simulation_time,
            display=True,
            save_data=True,
        )

        self.assertEqual(results["promotion_time"], 5.0)
        self.assertEqual(results["leader_history"][0], 0)
        self.assertIn(results["leader_history"][-1], [1, 2, 3, 4])
        self.assertEqual(results["positions"].shape, (1001, 5, 3))
        self.assertEqual(results["rotations"].shape, (1001, 5, 3, 3))
        failure_step = int(5.0 * 100)
        self.assertTrue(np.all(results["failed_history"][failure_step:, 0]))
        self.assertTrue(np.isnan(results["positions"][failure_step:, 0]).all())


if __name__ == "__main__":
    unittest.main()
