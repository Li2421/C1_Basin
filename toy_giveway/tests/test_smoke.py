import unittest

import numpy as np

from single_integrator.cbf import CBFSafetyFilter as LegacySafetyFilter
from single_integrator.environment import Config as LegacyConfig
from single_integrator.environment import GiveWayEnv as LegacyEnv
from toy_giveway.environment import Config, GiveWayEnv
from toy_giveway.safety import CBFSafetyFilter
from toy_giveway.smoke import run_smoke


class ToyGiveWayNamespaceTests(unittest.TestCase):
    def test_new_namespace_is_exact_legacy_identity(self):
        self.assertIs(Config, LegacyConfig)
        self.assertIs(GiveWayEnv, LegacyEnv)
        self.assertIs(CBFSafetyFilter, LegacySafetyFilter)

    def test_short_projected_rollout(self):
        result = run_smoke(steps=16)
        self.assertEqual(result["executed_steps"], 16)
        self.assertEqual(result["observation_shape"], [2, 10])
        self.assertEqual(result["action_shape"], [2, 2])
        self.assertEqual(result["termination"], "running")
        self.assertFalse(result["wall_collision"])
        self.assertFalse(result["agent_collision"])
        self.assertLessEqual(result["max_speed"], Config().max_speed + 1e-12)
        self.assertLess(result["max_integration_residual"], 1e-15)

        # A direct legacy instance follows the identical first transition.
        new_env, old_env = GiveWayEnv(), LegacyEnv()
        action = np.array([[0.1, 0.0], [-0.1, 0.0]])
        new_env.step(action)
        old_env.step(action)
        np.testing.assert_array_equal(new_env.positions, old_env.positions)


if __name__ == "__main__":
    unittest.main()
