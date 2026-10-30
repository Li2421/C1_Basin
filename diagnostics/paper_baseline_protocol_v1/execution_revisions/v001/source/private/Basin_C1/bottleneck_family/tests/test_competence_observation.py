import unittest

import numpy as np

from bottleneck_family.environment import BottleneckEnv
from bottleneck_family.observation import gate_waypoints_competence, policy_observation_competence
from bottleneck_family.scenario import Config


class CompetenceObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_agents=2, openings=(((0., .62),),), max_steps=2000)

    def test_aligned_entry_points_through_gate_in_both_directions(self):
        env = BottleneckEnv(self.config)
        offset = self.config.barrier_thickness / 2 + env.instance.geometry.margin + .12
        env.reset([[-offset, 0.], [offset, 0.]], [[4., 0.], [-4., 0.]])
        waypoints = gate_waypoints_competence(env)
        np.testing.assert_allclose(waypoints, [[offset, 0.], [-offset, 0.]])

    def test_reflection_preserves_eight_feature_semantics(self):
        env = BottleneckEnv(self.config)
        positions = np.array([[-2., 1.2], [3., -.7]])
        goals = np.array([[4., -.4], [-5., .8]])
        env.reset(positions, goals)
        env.velocities = np.array([[.15, -.1], [-.2, .05]])
        left = policy_observation_competence(env)
        reflected = BottleneckEnv(self.config)
        p = positions.copy(); p[:, 0] *= -1
        g = goals.copy(); g[:, 0] *= -1
        reflected.reset(p, g)
        reflected.velocities = env.velocities.copy()
        reflected.velocities[:, 0] *= -1
        right = policy_observation_competence(reflected)
        expected = left.copy()
        expected[:, (0, 2, 4, 6)] *= -1
        np.testing.assert_allclose(right, expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
