"""Test the runner with and without a leader failure, saving data for both."""

import unittest

import numpy as np

from simulation.sim_runner import run_simulation
from pathing.path_maker import make_leader_path
from formation.formation_maker import make_formation


class TestSimRunnerSavedData(unittest.TestCase):
    """Run the same five-drone scenario with and without leader failure."""

    def setUp(self):
        """Build the shared scenario so both runs start identically."""
        self.drones = [0, 1, 2, 3, 4]
        self.initial_leader = 0
        self.simulation_time = 10.0
        self.sim_rate = 100
        self.expected_steps = int(self.simulation_time * self.sim_rate) + 1

        self.initial_positions = make_formation(
            number_of_drones=len(self.drones),
            formation_type="leader_follower",
            shape="arrow",
            spread=1.0,
            dimension=2,
            plane="xz",
            leader_index=self.initial_leader,
            leader_position=np.array([0.0, 0.0, 1.0]),
        )
        # Build a fresh path per test so no state is shared between runs.
        self.leader_path = make_leader_path(
            path_type="random",
            first_waypoint=self.initial_positions[self.initial_leader],
            velocity=1,
            time=self.simulation_time,
            initial_heading=0.0,
            print_waypoints=True,
            seed=42,
        )

    def _run(self, leader_failure_time):
        """Run one simulation and save its data."""
        return run_simulation(
            drones=self.drones,
            initial_leader=self.initial_leader,
            initial_positions=self.initial_positions,
            leader_path=self.leader_path,
            leader_failure_time=leader_failure_time,
            election_type="closest",
            route_aligned_formation=True,
            sim_rate=self.sim_rate,
            t_final=self.simulation_time,
            display=False,
            save_data=True,
        )

    def test_without_leader_failure(self):
        """The initial leader should stay in charge for the whole run."""
        results = self._run(leader_failure_time=None)

        self.assertIsNone(results["promotion_time"])
        self.assertTrue(np.all(results["leader_history"] == self.initial_leader))
        self.assertEqual(results["positions"].shape, (self.expected_steps, 5, 3))
        self.assertEqual(results["rotations"].shape, (self.expected_steps, 5, 3, 3))
        self.assertFalse(results["failed_history"].any())
        self.assertFalse(np.isnan(results["positions"]).any())

    def test_with_leader_failure(self):
        """A follower should be promoted after the initial leader fails."""
        failure_time = self.simulation_time / 2.0
        results = self._run(leader_failure_time=failure_time)

        self.assertEqual(results["promotion_time"], failure_time)
        self.assertEqual(results["leader_history"][0], self.initial_leader)
        self.assertIn(results["leader_history"][-1], [1, 2, 3, 4])
        self.assertEqual(results["positions"].shape, (self.expected_steps, 5, 3))
        self.assertEqual(results["rotations"].shape, (self.expected_steps, 5, 3, 3))

        failure_step = int(failure_time * self.sim_rate)
        self.assertTrue(np.all(results["failed_history"][failure_step:, 0]))
        self.assertTrue(np.isnan(results["positions"][failure_step:, 0]).all())


if __name__ == "__main__":
    unittest.main()