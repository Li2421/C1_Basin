import unittest
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import quad
from scipy.stats import norm
from single_integrator.c1.risk.boundary_importance import event_envelope, importance_weights, proposal_centres

jax.config.update('jax_enable_x64', True)


class BoundaryImportanceTests(unittest.TestCase):
    def test_event_upper_bound_and_compact_gradient(self):
        x = jnp.linspace(-2., 2., 1001)
        r = np.asarray(event_envelope(x))
        self.assertTrue(np.all(r >= np.asarray(x >= 0)))
        self.assertTrue(np.all((0 <= r) & (r <= 1)))
        grad = jax.grad(event_envelope)
        for v in (-2., -.1, 0., 2.):
            self.assertEqual(float(grad(v)), 0.)
        self.assertGreater(float(grad(-.05)), 0.)

    def test_defensive_density_identity(self):
        x = np.linspace(-10, 10, 20001)
        centres = np.array([[-2.], [.8]])
        w = np.asarray(importance_weights(jnp.asarray(x[:, None]), jnp.asarray(centres)))
        p = norm.pdf(x)
        q = .5*p + .25*norm.pdf(x+2) + .25*norm.pdf(x-.8)
        np.testing.assert_allclose(q*w, p, rtol=1e-12, atol=1e-15)
        self.assertTrue(np.all((w > 0) & (w <= 2)))

    def test_gaussian_event_envelope_derivative(self):
        h, theta = .1, .2
        def risk(t):
            integral = quad(lambda z:(1+(t+z)/h)**2*(3-2*(1+(t+z)/h))*norm.pdf(z), -t-h, -t)[0]
            return norm.cdf(t) + integral
        derivative = quad(lambda z:6*(1+(theta+z)/h)*(1-(1+(theta+z)/h))/h*norm.pdf(z), -theta-h, -theta)[0]
        self.assertGreaterEqual(risk(theta), norm.cdf(theta))
        self.assertAlmostEqual(derivative, (risk(theta+1e-5)-risk(theta-1e-5))/2e-5, places=8)
        self.assertLess(risk(theta-.001*derivative), risk(theta))

    def test_pilot_degeneracy_and_norm_cap(self):
        mu = np.asarray(proposal_centres(jnp.array([1., -1.]), jnp.array([[0., 0.], [.001, 0.]])))
        np.testing.assert_array_equal(mu[0], [0., 0.])
        self.assertAlmostEqual(np.linalg.norm(mu[1]), 2.)


if __name__ == '__main__':
    unittest.main()
