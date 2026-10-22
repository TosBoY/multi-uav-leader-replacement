"""Test the simulation runner with a smooth generated waypoint path."""

import unittest

import numpy as np

from pathing.path_maker import make_leader_path
from simulation.sim_runner import run_simulation


class TestSimRunnerRandomWaypoints(unittest.TestCase):
    """Verify simulation along reproducible generated waypoints."""

    def test_waypoint_path_with_leader_promotion(self):
        """Follow generated waypoints and promote a replacement leader."""
        drones = [0, 1, 2, 3, 4]
        leader_start = np.array([0.0, 0.0, 1.0])
        leader_speed = 1
        leader_heading = 0.0
        simulation_time = 10.0
        leader_path = make_leader_path(
            path_type="random",
            first_waypoint=leader_start,
            velocity=leader_speed,
            time=simulation_time,
            initial_heading=leader_heading,
            # seed=42,
            print_waypoints=True,
        )

        initial_positions = {
            0: leader_start.tolist(),
            1: (leader_start + np.array([-0.8, -1.0, 0.0])).tolist(),
            2: (leader_start + np.array([-0.8, 1.0, 0.0])).tolist(),
            3: (leader_start + np.array([-1.6, -2.0, 0.0])).tolist(),
            4: (leader_start + np.array([-1.6, 2.0, 0.0])).tolist(),
        }
        failure_time = simulation_time / 2.0

        results = run_simulation(
            drones=drones,
            initial_leader=0,
            initial_positions=initial_positions,
            leader_path=leader_path,
            leader_failure_time=failure_time,
            election_type="closest",
            route_aligned_formation=True,
            sim_rate=100,
            t_final=simulation_time,
            display=True,
        )

        expected_promotion_time = np.ceil(failure_time * 100) / 100
        self.assertEqual(results["promotion_time"], expected_promotion_time)
        self.assertEqual(results["leader_history"][0], 0)
        self.assertIn(results["leader_history"][-1], [1, 2, 3, 4])
        self.assertEqual(results["positions"].shape[1:], (5, 3))
        self.assertTrue(np.isnan(results["positions"][:, 0, :]).any())


if __name__ == "__main__":
    unittest.main()
