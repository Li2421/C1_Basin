import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.r_cert import aggregate, atom_penalty


jax.config.update('jax_enable_x64', True)


class RCertAggregationTests(unittest.TestCase):
    def test_atom_origin_and_numerical_stability(self):
        values = atom_penalty(jnp.array([0., -1e6, 1e6]), jnp.ones(3))
        self.assertEqual(float(values[0]), 1.)
        self.assertTrue(np.isfinite(np.asarray(values)).all())
        self.assertGreater(float(values[1]), 1e6)
        self.assertEqual(float(values[2]), 0.)

    def test_normalized_softmin_bounds_for_variable_event_shapes(self):
        events = [jnp.array([[2., 3.], [-1., 4.], [.2, -.3]]),
                  jnp.array([[.1], [.4]])]
        scales = [jnp.array([[.5, 2.], [.7, 1.], [3., .4]]),
                  jnp.ones((2, 1))]
        kappa = .37
        result = aggregate(events, scales, kappa=kappa)
        for costs, risk in zip(result['S_ej'], result['R_e']):
            minimum = float(jnp.min(costs))
            self.assertGreaterEqual(float(risk), minimum-1e-12)
            self.assertLessEqual(float(risk),
                                 minimum+kappa*math.log(len(costs))+1e-12)

    def test_deadlock_implication_under_certificate_inclusion_assumption(self):
        # For the deadlocked event, every certificate row has at least one
        # nonpositive atom. This is exactly the contrapositive supplied by
        # C_ej subset complement(D_e), not an empirical certificate proof.
        deadlocked_event = jnp.array([
            [0., 8., 9.],
            [2., -1e-4, 7.],
            [-3., 4., 5.],
        ])
        other_event = jnp.full((2, 2), 10.)
        result = aggregate([deadlocked_event, other_event],
                           [jnp.ones_like(deadlocked_event), jnp.ones_like(other_event)],
                           kappa=.2)
        self.assertTrue(np.all(np.asarray(result['S_ej'][0]) >= 1.))
        self.assertGreaterEqual(float(result['R_e'][0]), 1.-1e-12)
        self.assertGreaterEqual(float(result['R_CERT']), 1.-1e-12)

    def test_inclusion_assumption_is_necessary(self):
        # A positive certificate on a trajectory labelled D violates the
        # required inclusion and can make the claimed bound false.
        invalid_certificate = jnp.full((1, 3), 100.)
        result = aggregate([invalid_certificate], [jnp.ones_like(invalid_certificate)],
                           kappa=.1)
        self.assertLess(float(result['R_CERT']), 1.)

    def test_gradients_are_not_detached_and_outer_max_is_exact(self):
        first = jnp.array([[-.2, .4], [.3, -.5]])
        second = jnp.array([[10., 10.]])
        fn = lambda x: aggregate([x, second], [jnp.ones_like(x), jnp.ones_like(second)],
                                 kappa=.3)['R_CERT']
        gradient = jax.grad(fn)(first)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0.)
        expected = jnp.max(aggregate([first, second],
                                     [jnp.ones_like(first), jnp.ones_like(second)],
                                     kappa=.3)['R_e'])
        self.assertEqual(float(fn(first)), float(expected))

    def test_jit_preserves_margin_gradient(self):
        scales = [np.ones((2, 2)), np.ones((1, 2))]
        other = jnp.array([[.5, .6]])
        compiled = jax.jit(lambda x: aggregate([x, other], scales, kappa=.2)['R_CERT'])
        margins = jnp.array([[.1, -.2], [.3, .4]])
        self.assertTrue(np.isfinite(float(compiled(margins))))
        self.assertGreater(float(jnp.linalg.norm(jax.grad(compiled)(margins))), 0.)

    def test_undefined_indices_and_invalid_constants_are_rejected(self):
        good = jnp.ones((1, 1))
        bad_calls = [
            lambda: aggregate([], [], kappa=.1),
            lambda: aggregate([jnp.ones((0, 1))], [jnp.ones((0, 1))], kappa=.1),
            lambda: aggregate([jnp.ones((1, 0))], [jnp.ones((1, 0))], kappa=.1),
            lambda: aggregate([good], [], kappa=.1),
            lambda: aggregate([good], [jnp.ones((1, 2))], kappa=.1),
            lambda: aggregate([good], [jnp.zeros((1, 1))], kappa=.1),
            lambda: aggregate([good], [good], kappa=0.),
        ]
        for call in bad_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()


if __name__ == '__main__':
    unittest.main()
