import unittest

import numpy as np

from single_integrator.cbf import CBFConfig, CBFSafetyFilter
from single_integrator.c1.risk.deadlock_geometry import (
    DeadlockGeometryConfig,
    extract_deadlock_geometry,
)
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import rollout
from single_integrator.filters import FilterResult


class HeadOn:
    def sample_actions(self, observations, seed):
        return np.array([[[.5, 0.], [-.5, 0.]]])


class C1DeadlockGeometryTests(unittest.TestCase):
    def setUp(self):
        self.env = GiveWayEnv(Config())
        self.cbf = CBFConfig()

    def test_task_force_is_bounded_nominal_not_safe_velocity(self):
        self.env.reset([[-.2, 0.], [.2, 0.]])
        nominal = np.array([[.5, 0.], [-.5, 0.]])
        safe = CBFSafetyFilter(self.cbf)(self.env.snapshot(), nominal).velocity
        geometry = extract_deadlock_geometry(self.env.snapshot(), nominal, safe, self.cbf)
        np.testing.assert_array_equal(geometry["task_force"], nominal)
        self.assertFalse(np.array_equal(geometry["task_force"], safe))

    def test_cbf_values_residual_and_per_agent_gradient_blocks(self):
        self.env.reset([[-.2, 0.], [.2, 0.]])
        nominal = np.array([[.5, 0.], [-.5, 0.]])
        safe = CBFSafetyFilter(self.cbf)(self.env.snapshot(), nominal).velocity
        geometry = extract_deadlock_geometry(self.env.snapshot(), nominal, safe, self.cbf)
        d_safe = 2 * self.env.config.agent_radius + self.env.config.agent_collision_margin + self.cbf.separation_buffer
        relative = self.env.positions[0] - self.env.positions[1]
        self.assertAlmostEqual(geometry["h"][0], relative @ relative - d_safe ** 2)
        np.testing.assert_allclose(geometry["safety_force_blocks"][0], [2 * relative, -2 * relative])
        self.assertEqual(geometry["safety_force_blocks"].shape, (17, 2, 2))
        self.assertEqual(geometry["h"].shape, (17,))
        self.assertEqual(geometry["cbf_residual"].shape, (17,))
        self.assertLessEqual(abs(geometry["cbf_residual"][0]), 1e-7)
        self.assertEqual(geometry["constraint_kind"][0], "pairwise")
        self.assertEqual(geometry["constraint_agent"][0], -1)

    def test_active_selection_and_empty_set(self):
        self.env.reset([[-.2, 0.], [.2, 0.]])
        nominal = np.array([[.5, 0.], [-.5, 0.]])
        safe = CBFSafetyFilter(self.cbf)(self.env.snapshot(), nominal).velocity
        active = extract_deadlock_geometry(
            self.env.snapshot(), nominal, safe, self.cbf,
            DeadlockGeometryConfig(rho=1., active_tol=1e-7),
        )
        np.testing.assert_array_equal(active["active_indices"], [0])
        self.assertEqual(active["active_generators"].shape, (1, 2, 2))
        self.assertEqual(active["active_constraint_agent"].tolist(), [-1])
        empty = extract_deadlock_geometry(
            self.env.snapshot(), nominal, safe, self.cbf,
            DeadlockGeometryConfig(rho=0., active_tol=0.),
        )
        self.assertTrue(empty["completely_deadlock_free"])
        self.assertEqual(empty["active_generators"].shape, (0, 2, 2))
        self.assertFalse(empty["cone_distance_defined"])
        self.assertIsNone(empty["cone_distance"])

    def test_opt_in_rollout_emits_geometry_without_risk_values(self):
        summary, arrays = rollout(
            HeadOn(), np.array([[-.2, 0.], [.2, 0.]]), Config(max_steps=1),
            0, 0, CBFSafetyFilter(self.cbf), cbf_config=self.cbf,
            c1_geometry_config=DeadlockGeometryConfig(rho=1., active_tol=1e-7),
        )
        self.assertEqual(summary["episode_steps"], 1)
        self.assertEqual(arrays["c1_task_force"].shape, (1, 2, 2))
        self.assertEqual(arrays["c1_h"].shape, (1, 17))
        self.assertEqual(arrays["c1_cbf_residual"].shape, (1, 17))
        self.assertEqual(arrays["c1_safety_force_blocks"].shape, (1, 17, 2, 2))
        self.assertEqual(arrays["c1_active_mask"].shape, (1, 17))
        self.assertTrue(arrays["c1_active_mask"][0, 0])
        self.assertFalse(arrays["c1_cone_distance_defined"].any())
        self.assertNotIn("r_t", arrays)
        self.assertNotIn("R_risk", arrays)


if __name__ == "__main__":
    unittest.main()
