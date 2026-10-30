import unittest
import jax.numpy as jnp
import optax
from single_integrator.c1.training.step_control import guarded_step as original
from single_integrator.c1.training.step_control_restart import guarded_step


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = optax.adam(.1)
        self.x = jnp.asarray(1.)
        self.state = self.optimizer.init(self.x)
        for _ in range(20):
            _, self.state = self.optimizer.update(jnp.asarray(-2.), self.state, self.x)
        self.objective = lambda x: (x*x, {})

    def test_stale_momentum_rejection_can_be_resolved(self):
        before = self.objective(self.x)
        old = original(self.x, self.state, jnp.asarray(2.), self.optimizer, self.objective, before)
        self.assertEqual(old[3]['status'], 'rejected')
        new = guarded_step(self.x, self.state, jnp.asarray(2.), self.optimizer, self.objective, before)
        self.assertEqual(new[3]['proposal'], 'restart')
        self.assertEqual(new[3]['status'], 'accepted')
        self.assertLess(float(new[2][0]), float(before[0]))
        self.assertEqual(int(new[1][0].count), 1)

    def test_failed_restart_keeps_original_moments(self):
        before = self.objective(self.x)
        result = guarded_step(self.x, self.state, jnp.asarray(2.), self.optimizer,
                              lambda x: (jnp.asarray(100.), {}), before)
        self.assertIs(result[0], self.x)
        self.assertIs(result[1], self.state)
        self.assertEqual(result[3]['status'], 'rejected')

    def test_wrong_gradient_is_not_blindly_reversed(self):
        initial = self.optimizer.init(self.x)
        result = guarded_step(self.x, initial, jnp.asarray(-2.), self.optimizer,
                              self.objective, self.objective(self.x))
        self.assertEqual(result[3]['proposal'], 'ordinary')
        self.assertEqual(result[3]['status'], 'rejected')


if __name__ == '__main__':
    unittest.main()
