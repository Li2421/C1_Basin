import unittest

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.reachability_margin import (
    probability_upper, temporal_probability_upper)


class ReachabilityMarginTests(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)

    def test_event_and_safe_margin_bounds(self):
        for eta in (.01, .1, .4):
            for delta in (.1, .7):
                rho = jnp.linspace(-5., 5., 2001)
                r = np.asarray(probability_upper(rho, margin_width=delta,
                                                tail_budget=eta))
                self.assertTrue(np.all(r >= 0))
                self.assertTrue(np.all(r <= 1 + eta + 1e-12))
                self.assertTrue(np.all(r[np.asarray(rho) >= 0] >= 1 - 1e-12))
                self.assertTrue(np.all(r[np.asarray(rho) <= -delta] <= eta + 1e-12))
                self.assertAlmostEqual(float(probability_upper(0., margin_width=delta,
                                                               tail_budget=eta)), 1.)

    def test_distribution_gap_includes_smoothing_band(self):
        rng = np.random.default_rng(18)
        rho = rng.uniform(-2., 1., 2000)
        error, delta, eta = .08, .2, .03
        upper = rho + rng.uniform(0., error, len(rho))
        mean = float(jnp.mean(probability_upper(upper, margin_width=delta,
                                               tail_budget=eta)))
        p = np.mean(rho >= 0)
        near = np.mean((rho < 0) & (rho > -delta-error))
        self.assertGreaterEqual(mean + 1e-12, p)
        self.assertLessEqual(mean, p + near + eta + 1e-12)

    def test_temporal_event_and_directional_derivative(self):
        x = jnp.linspace(-.09, -.01, 24).reshape(8, 3)
        valid = jnp.ones(8, bool)
        def fn(a):
            return temporal_probability_upper(a, valid, hold_samples=3,
                temperature=.005, margin_width=.2, tail_budget=.1)['J_live']
        direction = jnp.cos(jnp.arange(24)).reshape(x.shape)
        g = jax.grad(fn)(x)
        eps = 1e-6
        fd = (fn(x+eps*direction)-fn(x-eps*direction))/(2*eps)
        self.assertAlmostEqual(float(fd), float(jnp.sum(g*direction)), places=7)
        self.assertGreaterEqual(float(fn(jnp.ones_like(x)*.05)), 1.)
        # These are predicate-space derivatives, not policy escape guarantees.
        self.assertTrue(np.isfinite(np.asarray(g)).all())

    def test_no_windows_and_invalid_parameters(self):
        for n in (2, 8):
            def fn(x):
                return temporal_probability_upper(x, jnp.zeros(n, bool),
                    hold_samples=3, temperature=.005, margin_width=.2,
                    tail_budget=.1)['J_live']
            x = jnp.ones((n, 3))
            self.assertEqual(float(fn(x)), 0.)
            np.testing.assert_array_equal(jax.grad(fn)(x), np.zeros_like(x))
        for eta in (0., 1., -1.):
            with self.assertRaises(ValueError):
                probability_upper(0., margin_width=.2, tail_budget=eta)


if __name__ == '__main__':
    unittest.main()
