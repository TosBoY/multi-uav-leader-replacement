"""Test the complete experiment-maker workflow."""

import unittest

from simulation.experiment_maker import make_experiment


class TestExperimentMaker(unittest.TestCase):
    """Verify that make_experiment wires the project components together."""

    def test_runs_line_experiment_with_leader_failure(self):
        results = make_experiment(
            number_of_drones=5,
            initial_leader=0,
            leader_position=[0.0, 0.0, 1.0],
            formation_type="leader_follower",
            shape="line_x",
            spread=1.0,
            dimension=2,
            plane="xz",
            path_type="line",
            velocity=1.0,
            simulation_time=10,
            initial_heading=0.0,
            leader_failure_time=5.0,
            election_type="minimum_travel",
            sim_rate=100,
            display=False,
            run_type="fail",
            save_data=True,
            print_path=True,
        )

        self.assertEqual(results["promotion_time"], 5.0)
        self.assertEqual(results["leader_history"][0], 0)
        self.assertEqual(results["leader_history"][-1], 1)
        self.assertEqual(results["positions"].shape, (1001, 5, 3))
        self.assertEqual(results["rotations"].shape, (1001, 5, 3, 3))
        self.assertEqual(results["travel_distances"].shape, (1001, 5))


if __name__ == "__main__":
    unittest.main()
