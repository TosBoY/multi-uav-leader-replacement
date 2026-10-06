"""Tests for the leader path factory."""

import unittest

import numpy as np

from pathing.path_maker import make_leader_path


class TestPathMaker(unittest.TestCase):
    """Verify the supported leader path types."""

    def test_random_path_is_reproducible_and_starts_at_first_waypoint(self):
        """Create matching MinSnap paths from the same random seed."""
        first_waypoint = np.array([1.0, -2.0, 1.5])
        random_path = make_leader_path(
            path_type="random",
            first_waypoint=first_waypoint,
            velocity=1.0,
            time=4.0,
            seed=42,
            initial_heading=np.pi / 2,
        )
        repeated_path = make_leader_path(
            path_type="random",
            first_waypoint=first_waypoint,
            velocity=1.0,
            time=4.0,
            seed=42,
            initial_heading=np.pi / 2,
        )

        np.testing.assert_allclose(
            random_path.update(0.0)["x"],
            first_waypoint,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            random_path.t_keyframes,
            repeated_path.t_keyframes,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            random_path.update(1.0)["x"],
            repeated_path.update(1.0)["x"],
            atol=1e-12,
        )

    def test_line_path_follows_heading_and_speed(self):
        """Create a ConstantSpeed path along the positive y-axis."""
        line_path = make_leader_path(
            path_type="line",
            first_waypoint=[0.0, 0.0, 1.0],
            velocity=2.0,
            time=3.0,
            initial_heading=np.pi / 2,
        )

        np.testing.assert_allclose(
            line_path.update(0.0)["x"],
            [0.0, 0.0, 1.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            line_path.update(1.0)["x"],
            [0.0, 2.0, 1.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            line_path.update(3.0)["x"],
            [0.0, 6.0, 1.0],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
