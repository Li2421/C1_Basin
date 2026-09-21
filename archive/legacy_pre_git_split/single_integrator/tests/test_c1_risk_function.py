import unittest

import jax
import jax.numpy as jnp

from single_integrator.c1.risk.risk_function import RiskConfig, instantaneous_risk, trajectory_risk


class ProvisionalRiskTests(unittest.TestCase):
    def test_empty_cone_is_zero_and_head_on_ray_is_high(self):
        cfg = RiskConfig(D0=.1, kappa=8.)
        generators = jnp.array([[[1., 0.], [0., 0.]]])
        empty, _ = instantaneous_risk(jnp.array([[.2, 0.], [0., 0.]]), generators, jnp.array([False]), cfg)
        head_on, _ = instantaneous_risk(jnp.array([[-.2, 0.], [0., 0.]]), generators, jnp.array([True]), cfg)
        self.assertEqual(float(empty), 0.)
        self.assertGreater(float(head_on), .9)

    def test_cone_risk_has_force_gradient_and_trajectory_aggregation(self):
        cfg = RiskConfig()
        generators = jnp.array([[[1., 0.], [0., 0.]]])
        grad = jax.grad(lambda force: instantaneous_risk(force.reshape(2, 2), generators, jnp.array([True]), cfg)[0])(jnp.array([-.2, .02, -.2, 0.]))
        self.assertGreater(float(jnp.linalg.norm(grad)), 0.)
        self.assertGreater(float(trajectory_risk(jnp.array([[0., 1.]]), cfg)[0]), .4)


if __name__ == '__main__':
    unittest.main()
