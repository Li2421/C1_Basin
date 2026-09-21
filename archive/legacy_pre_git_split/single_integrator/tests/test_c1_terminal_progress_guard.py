import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.terminal_progress_guard import combine


class GuardTests(unittest.TestCase):
    def test_order_and_upper_bound_preserved(self):
        for base in [0.,.2,.999,1.,1.001,2.,10.]:
            for margin in [-100.,-1.,0.,1.,100.]:
                risk,guard=combine(jnp.array(base),jnp.array(margin),True)
                self.assertGreaterEqual(float(risk),base-1e-6)
                self.assertLessEqual(float(guard),.5)
                if base<=1:self.assertLessEqual(float(risk),1.)
                else:self.assertGreater(float(risk),1.)

    def test_no_timeout_no_change(self):
        for base in [0.,.5,1.,2.]:
            risk,guard=combine(jnp.array(base),jnp.array(1.),False)
            self.assertEqual(float(risk),base);self.assertEqual(float(guard),0.)

    def test_actual_derivative_encourages_progress(self):
        f=lambda progress:combine(jnp.array(.2),1-progress/.025,True)[0]
        derivative=float(jax.grad(f)(jnp.array(0.)))
        h=1e-4
        finite=float((f(h)-f(-h))/(2*h))
        self.assertLess(derivative,0.)
        np.testing.assert_allclose(derivative,finite,rtol=2e-3,atol=1e-3)

    def test_deadlock_branch_has_progress_gradient(self):
        f=lambda margin:combine(jnp.array(1.1),margin,True)[0]
        self.assertGreater(float(jax.grad(f)(jnp.array(1.))),0.)
