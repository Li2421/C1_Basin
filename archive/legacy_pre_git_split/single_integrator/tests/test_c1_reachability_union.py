"""Event semantics of the experimental union adapter, independent of GPU."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.reachability_union import trajectory

jax.config.update('jax_enable_x64', True)


class UnionTest(unittest.TestCase):
    def score(self, n, timeout, speed=0., padding=0):
        before = jnp.broadcast_to(jnp.array([[-.3, 0.], [.3, 0.]]), (n+padding, 2, 2))
        goals = jnp.array([[1., 0.], [-1., 0.]])
        applied = jnp.ones((n+padding, 4))*speed
        return trajectory(before, before, applied, goals, jnp.arange(n+padding)<n,
            terminal_timeout=timeout, dt=.05, max_speed=.5, goal_tolerance=.08,
            hold_seconds=5., progress_window_seconds=2., progress_epsilon=.01,
            speed_epsilon_fraction=.05)

    def test_strict_event_is_bounded_above_indicator(self):
        value = float(self.score(150, False)['J_live'])
        self.assertGreaterEqual(value, 1.)
        self.assertLessEqual(value, 1.01)

    def test_terminal_stall_requires_timeout_and_full_window(self):
        self.assertEqual(float(self.score(40, False)['J_live']), 0.)
        self.assertEqual(float(self.score(39, True)['J_live']), 0.)
        self.assertGreaterEqual(float(self.score(40, True)['J_live']), 1.)

    def test_padding_does_not_change_terminal_score(self):
        np.testing.assert_allclose(self.score(40, True)['J_live'],
                                   self.score(40, True, padding=200)['J_live'], atol=1e-12)

    def test_increasing_speed_reduces_stall_risk(self):
        # At this point the speed predicate is the active escape direction.
        derivative = float(jax.grad(lambda v: self.score(40, True, speed=v)['J_live'])(.035))
        self.assertTrue(np.isfinite(derivative))
        self.assertLess(derivative, 0.)


if __name__ == '__main__':
    unittest.main()
