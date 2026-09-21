import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.risk_v3 import trajectory, progress_steps
from single_integrator.c1.risk.joint_frozen import trajectory as legacy


class V3RiskTests(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)
        self.goals = jnp.array([[1.09, 0.], [-1.09, 0.]])

    def path(self, n):
        t = jnp.arange(n+1)*.05
        positions = jnp.stack([jnp.stack([-.85+.004*t, jnp.zeros_like(t)], -1),
                               jnp.stack([.85-.004*t, jnp.zeros_like(t)], -1)], 1)
        return positions[:-1], positions[1:]

    def test_progress_matches_frozen_formula_in_old_window(self):
        before, after = self.path(500)
        success = jnp.arange(500) >= 250
        g = jnp.full((200,), .2)
        old = legacy(before, after, jnp.ones((500, 4))*.1, g, self.goals, success)
        progress, _ = progress_steps(before, after, self.goals, success)
        np.testing.assert_allclose(progress[100:300].mean(), old[0], atol=1e-14)

    def test_late_progress_is_invisible_to_old_Pg_but_not_v3(self):
        before, after = self.path(850)
        success = jnp.zeros(850, bool)
        def positions(shift):
            delta = jnp.zeros_like(after).at[600:, 0, 0].set(shift)
            return after+delta
        def old(shift):
            return legacy(before[:500], positions(shift)[:500], jnp.ones((500,4))*.1,
                          jnp.full((200,), .2), self.goals, success[:500])[:2].sum()
        def new(shift):
            return trajectory(before, positions(shift), jnp.full((750,), .2),
                              self.goals, success)['J_live']
        self.assertEqual(float(jax.grad(old)(0.)), 0.)
        self.assertLess(float(jax.grad(new)(0.)), -1e-4)
        self.assertLess(float(new(.001)), float(new(0.)))
        h = 1e-5
        np.testing.assert_allclose(jax.grad(new)(0.), (new(h)-new(-h))/(2*h), rtol=1e-5)

    def test_post_success_cost_is_absorbed_without_shorter_denominator(self):
        before, after = self.path(850)
        success = jnp.arange(850) >= 249
        score = lambda g: trajectory(before, after, g, self.goals, success)['J_live']
        g = jnp.full((750,), .2)
        changed = g.at[150:].set(100.)
        np.testing.assert_allclose(score(g), score(changed))
        derivative = jax.grad(score)(g)
        np.testing.assert_allclose(derivative[:150], 1/750)
        np.testing.assert_array_equal(derivative[150:], 0.)

    def test_exact_goal_has_finite_progress_gradient(self):
        before, after = self.path(850)
        after = after.at[600:].set(self.goals)
        fun = lambda x: trajectory(before, x, jnp.zeros(750), self.goals,
                                  jnp.arange(850) >= 600)['J_live']
        self.assertTrue(np.isfinite(np.asarray(jax.grad(fun)(after))).all())

    def test_missing_late_geometry_is_rejected(self):
        before, after = self.path(850)
        with self.assertRaises(ValueError):
            trajectory(before, after, jnp.zeros(200), self.goals, jnp.zeros(850, bool))


if __name__ == '__main__':
    unittest.main()
