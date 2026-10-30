import unittest

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.training.primal_dual import (
    PrimalDualState, deviation_cost, dual_update, primal_loss,
)


class PrimalDualTests(unittest.TestCase):
    def test_validation_selection_respects_constraint_then_deviation(self):
        from single_integrator.c1.training.selection import better_feasible_candidate
        incumbent=dict(J_def=.2,J_live=.4,epsilon=.5)
        self.assertTrue(better_feasible_candidate(incumbent))
        self.assertFalse(better_feasible_candidate(dict(J_def=0.,J_live=.6,epsilon=.5)))
        self.assertFalse(better_feasible_candidate(dict(J_def=.3,J_live=.1,epsilon=.5),incumbent))
        self.assertTrue(better_feasible_candidate(dict(J_def=.1,J_live=.45,epsilon=.5),incumbent))
        self.assertFalse(better_feasible_candidate(dict(J_def=float('nan'),J_live=.1,epsilon=.5),incumbent))

    def test_identity_and_weighted_deviation(self):
        safe = jnp.zeros((2, 3, 2))
        executed = safe.at[0, 0].set(jnp.array([1., 2.]))
        self.assertAlmostEqual(float(deviation_cost(executed, safe)), 5. / 6)
        self.assertAlmostEqual(float(deviation_cost(executed, safe, jnp.array([2., .5]))), 4. / 6)

    def test_loss_and_dual_projection(self):
        safe = jnp.zeros((2, 1, 1))
        executed = jnp.ones((2, 1, 1))
        loss, values = primal_loss(executed, safe, jnp.array([.3, .7]), 2., .4)
        self.assertAlmostEqual(float(values['J_def']), 1.)
        self.assertAlmostEqual(float(values['J_live']), .5)
        self.assertAlmostEqual(float(loss), 1.2)
        self.assertAlmostEqual(dual_update(PrimalDualState(.1), jnp.array([.3, .7]), .4, .5).dual, .15)
        self.assertEqual(dual_update(PrimalDualState(.1), jnp.array([0.]), .4, 1.).dual, 0.)

    def test_executed_control_retains_gradient(self):
        safe = jnp.zeros((1, 2, 2))
        grad = jax.grad(lambda u: primal_loss(u, safe, jnp.array([.5]), 0., .5)[0])(jnp.ones((1, 2, 2)))
        np.testing.assert_array_equal(grad, jnp.ones((1, 2, 2)))


if __name__ == '__main__':
    unittest.main()
