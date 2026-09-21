import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from single_integrator.c1.training.step_control import guarded_step


class StepControlTests(unittest.TestCase):
    def test_overshoot_backtracks_and_commits_one_adam_observation(self):
        optimizer = optax.adam(4.)
        params = jnp.array(1.)
        state = optimizer.init(params)
        def objective(x):
            return x*x, {'J_live': x*x}
        gradient = jax.grad(lambda x: objective(x)[0])(params)
        new, new_state, after, info = guarded_step(params,state,gradient,optimizer,objective,objective(params))
        self.assertEqual(info['status'], 'accepted')
        self.assertLess(info['scale'], 1.)
        self.assertLessEqual(float(after[0]), 1.)
        self.assertEqual(int(new_state[0].count), 1)
        self.assertNotEqual(float(new), float(params))

    def test_rejection_rolls_back_parameters_and_moments(self):
        optimizer = optax.adam(.1)
        params = jnp.array(1.)
        state = optimizer.init(params)
        objective = lambda x: (x*x, {'J_live': x*x})
        new, new_state, after, info = guarded_step(params,state,jnp.array(-1.),optimizer,objective,objective(params),3)
        self.assertEqual(info['status'], 'rejected')
        self.assertIs(new, params)
        self.assertIs(new_state, state)
        self.assertEqual(len(info['trials']), 4)
        self.assertEqual(float(after[0]), 1.)

    def test_zero_step_preserves_adam_clock(self):
        optimizer = optax.adam(.1)
        params = jnp.array(0.)
        state = optimizer.init(params)
        def never_called(x):
            self.fail('zero step must not replay trajectories')
        _, new_state, _, info = guarded_step(params,state,params,optimizer,never_called,(params,{}))
        self.assertIs(new_state,state)
        self.assertEqual(info['status'], 'no_op')

    def test_nonfinite_candidate_objective_is_not_accepted(self):
        optimizer = optax.adam(.1)
        params = jnp.array(1.)
        state = optimizer.init(params)
        _, new_state, _, info = guarded_step(params,state,params,optimizer,
            lambda x: (jnp.array(np.nan),{}),(params,{}),1)
        self.assertIs(new_state,state)
        self.assertEqual(info['status'],'rejected')
        self.assertTrue(all(t['loss'] is None for t in info['trials']))
