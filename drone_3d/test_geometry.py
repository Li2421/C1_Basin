import unittest
import numpy as np
from .pilot import Config, constraints, geometry, initial_state, project, segment_box_distance, swept_pair_gap


class GeometryChecks(unittest.TestCase):
    def test_midstep_pair_collision(self):
        a = np.array([[-1., 0., 0.], [1., 0., 0.]])
        self.assertEqual(swept_pair_gap(a, a[::-1], np.full(3, .5)), -1.)

    def test_midstep_wall_collision(self):
        box = np.array([[-.1, -.1, -.1], [.1, .1, .1]])
        self.assertEqual(segment_box_distance(np.array([-1., 0., 0.]), np.array([1., 0., 0.]), box), 0.)
        self.assertAlmostEqual(segment_box_distance(np.array([-1., 1., 0.]), np.array([1., 1., 0.]), box), .9)

    def test_narrow_corridor_and_altitude_only_no_pass(self):
        c = Config()
        self.assertLess(((.6-2*c.radius)/c.pair_xy)**2 + ((.8-2*c.radius)/c.pair_z)**2, 1.)
        self.assertLess((1.1-2*c.radius)/c.pair_z, 1.)
        self.assertGreater((.60/c.pair_xy)**2 + ((.87-.4)/c.pair_z)**2, 1.)

    def test_single_agent_straight_path_clear(self):
        p = np.array(initial_state(17)["positions"])
        for box in geometry()[1]:
            self.assertGreater(segment_box_distance(p[0], p[1], box), Config().radius)

    def test_projected_active_pair_and_speed_ball(self):
        c = Config(); p = np.array([[-.2513, 0., .4], [.2513, 0., .4]])
        u = project(np.array([[3., 0., 0.], [-3., 0., 0.]]), p, c)
        a, b = constraints(p, c)
        self.assertGreaterEqual((a@u.ravel()-b).min(), -1e-8)
        self.assertLessEqual(np.linalg.norm(u, axis=1).max(), c.max_speed+1e-8)
        self.assertLess(np.linalg.norm(u), .001)


if __name__ == "__main__":
    unittest.main()
