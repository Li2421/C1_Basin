import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.ordered_guidance import ordered_score

jax.config.update('jax_enable_x64',True)


class OrderedTests(unittest.TestCase):
    def test_separation_independent_of_guidance(self):
        for s in (0.,.1,1.,2.,4.5):
            for rho in (-100.,-1.,-.01,0.,.01,1.):
                value=float(ordered_score(jnp.array(rho),jnp.array(s)))
                self.assertEqual(value>1,rho>0)
                self.assertGreaterEqual(value,0.)

    def test_both_partial_derivatives_match_actual_function(self):
        for rho in (-.1,.1,.5):
            x=jnp.array([rho,1.3])
            fn=lambda x:ordered_score(x[0],x[1])
            grad=np.asarray(jax.grad(fn)(x))
            fd=np.array([float((fn(x+jnp.eye(2)[i]*1e-6)-fn(x-jnp.eye(2)[i]*1e-6))/2e-6) for i in range(2)])
            np.testing.assert_allclose(grad,fd,rtol=1e-5)
            self.assertTrue(np.all(grad>0))

    def test_continuous_at_zero_margin(self):
        self.assertEqual(float(ordered_score(jnp.array(0.),jnp.array(2.))),1.)
        for r in (-1e-10,1e-10):
            self.assertAlmostEqual(float(ordered_score(jnp.array(r),jnp.array(2.))),1.,places=8)


if __name__=='__main__':unittest.main()
