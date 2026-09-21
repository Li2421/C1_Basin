import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.deadlock_union import terminal_stall_upper
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace


class DeadlockUnionTests(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)

    def test_bound_agrees_with_original_terminal_classifier(self):
        rng = np.random.default_rng(20260916)
        for dt in (.05, .07, .1):
            for speed in (.01, .04, .06):
                errors = 1+rng.uniform(-.004, .004, (100, 2))
                speeds = np.full_like(errors, speed)
                label, _ = classify_timeout_trace(dict(max_speed=speeds.max(axis=1), goal_errors=errors), 'other_timeout', dt)
                r = terminal_stall_upper(jnp.asarray(errors), jnp.asarray(speeds), jnp.ones(100, bool), True, dt=dt)
                self.assertEqual(float(r['stalled_hard_risk']) > 1, label=='stalled_deadlock')
                self.assertGreaterEqual(float(r['stalled_bound']), float(r['stalled_hard_risk'])-1e-12)
                if label=='stalled_deadlock':
                    self.assertGreaterEqual(float(r['stalled_bound']), 1.)

    def test_padding_does_not_change_terminal_detector(self):
        e = jnp.ones((60, 3)); s = jnp.full_like(e, .02)
        unpadded = terminal_stall_upper(e, s, jnp.ones(60, bool), True, dt=.05)
        padded = terminal_stall_upper(jnp.concatenate([e,jnp.zeros((20,3))]),
            jnp.concatenate([s,jnp.full((20,3),5.)]), jnp.arange(80)<60, True, dt=.05)
        self.assertAlmostEqual(float(unpadded['stalled_bound']), float(padded['stalled_bound']), places=12)

    def test_success_and_strict_deadlock_are_not_terminal_timeouts(self):
        e = jnp.ones((80, 2)); s = jnp.zeros_like(e)
        fn = lambda v: terminal_stall_upper(e, v, jnp.ones(80, bool), False, dt=.05)['stalled_bound']
        self.assertEqual(float(fn(s)), 0.)
        np.testing.assert_array_equal(jax.grad(fn)(s), 0.)

    def test_terminal_gradient_matches_finite_difference(self):
        e = jnp.ones((80, 2)); s = jnp.full_like(e, .02)
        fn = lambda v: terminal_stall_upper(e, v, jnp.ones(80, bool), True, dt=.05)['stalled_bound']
        d = jnp.sin(jnp.arange(160)).reshape(s.shape)
        h = 1e-6
        self.assertAlmostEqual(float((fn(s+h*d)-fn(s-h*d))/(2*h)),
                              float(jnp.sum(jax.grad(fn)(s)*d)), places=6)


if __name__=='__main__':
    unittest.main()
