import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.completion_certificate import trajectory


class CompletionCertificateTests(unittest.TestCase):
    def test_slow_success_has_no_deadline_false_positive(self):
        goals = jnp.array([[1.09, 0.], [-1.09, 0.]])
        start = jnp.array([[-.6, 0.], [.6, 0.]])
        # Algebraic risk test only: these positions are not a collision-free
        # two-agent trajectory. Geometric feasibility is tested separately.
        fraction = jnp.minimum(jnp.arange(851)/600, 1.)
        x = start+(goals-start)*fraction[:, None, None]
        result = trajectory(x[:-1], x[1:], jnp.diff(x, axis=0)/.05,
                            goals, jnp.arange(850) < 600)
        self.assertEqual(float(result['timeout_bound']), 0.)
        self.assertGreater(float(result['unconditional_deadline_bound']), 0.)

    def test_failure_retains_bound_and_gradient(self):
        goals = jnp.array([[1.09, 0.], [-1.09, 0.]])
        x = jnp.tile(jnp.array([[-.6, 0.], [.6, 0.]])[None], (851, 1, 1))
        def score(z):
            return trajectory(z[:-1], z[1:], jnp.diff(z, axis=0)/.05,
                              goals, jnp.ones(850, bool))['J_live']
        self.assertGreaterEqual(float(score(x)), 1.)
        self.assertTrue(np.isfinite(jax.grad(score)(x)).all())


if __name__ == '__main__':
    unittest.main()
