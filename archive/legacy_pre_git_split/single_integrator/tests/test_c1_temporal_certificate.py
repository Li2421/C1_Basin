import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.temporal_certificate import trajectory


class TemporalCertificateTests(unittest.TestCase):
    def setUp(self):
        self.goals = jnp.array([[1.09, 0.], [-1.09, 0.]])
        self.before = jnp.tile(jnp.array([[-.6, 0.], [.6, 0.]])[None], (850, 1, 1))
        self.u = jnp.zeros_like(self.before)
        self.alive = jnp.ones(850, bool)

    def test_stationary_unfinished_detected_with_finite_gradient(self):
        def score(x):
            return trajectory(x, x, self.u, self.goals, self.alive)['J_live']
        result = trajectory(self.before, self.before, self.u, self.goals, self.alive)
        self.assertGreaterEqual(float(result['deadlock_bound']), 1.)
        self.assertGreaterEqual(float(result['timeout_bound']), 1.)
        self.assertTrue(np.isfinite(jax.grad(score)(self.before)).all())

    def test_absorbed_goals_have_no_false_deadlock(self):
        at_goal = jnp.tile(self.goals[None], (850, 1, 1))
        result = trajectory(at_goal, at_goal, self.u, self.goals, jnp.zeros(850, bool))
        self.assertEqual(float(result['J_live']), 0.)

    def test_less_than_hold_cannot_trigger_deadlock(self):
        # Validity mask must include a full 2s history followed by 101 samples.
        result = trajectory(self.before, self.before, self.u, self.goals,
                            jnp.arange(850) < 139)
        self.assertEqual(float(result['deadlock_bound']), 0.)


if __name__ == '__main__':
    unittest.main()
