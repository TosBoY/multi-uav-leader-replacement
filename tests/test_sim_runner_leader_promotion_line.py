"""Test the reusable runner with the straight-line leader-promotion scenario."""

import unittest

import numpy as np

from rotorpy.trajectories.speed_traj import ConstantSpeed
from simulation.sim_runner import run_simulation


class TestSimRunnerLeaderPromotionLine(unittest.TestCase):
    """Verify the same five-drone scenario through the shared runner."""

    def test_random_leader_promotion_on_straight_path(self):
        """Promote a random follower after the initial leader fails."""
        drones = [0, 1, 2, 3, 4]
        initial_positions = {
            0: [0, 0.0, 1.0],
            1: [-1, -1, 1.0],
            2: [-2, -2, 1.0],
            3: [-1, 1, 1.0],
            4: [-2, 2, 1.0],
        }
        leader_path = ConstantSpeed(
            init_pos=np.array(initial_positions[0]),    
            dist=20.0,
            speed=5,
            axis=0,
            yaw_traj="forward",
        )

        results = run_simulation(
            drones=drones,
            initial_leader=0,
            initial_positions=initial_positions,
            leader_path=leader_path,
            leader_failure_time=5.0,
            election_type="closest",
            sim_rate=100,
            t_final=10.0,
            display=True,
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
