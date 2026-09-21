import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.moment_probability import from_moments,from_samples

jax.config.update('jax_enable_x64',True)


class MomentProbabilityTests(unittest.TestCase):
    def test_bound_on_finite_distributions(self):
        rng=np.random.default_rng(2026091878)
        for _ in range(100):
            x=rng.normal(size=9)*rng.uniform(.1,3)+rng.uniform(-2,2)
            p=rng.dirichlet(np.ones(9));mu=p@x;var=p@((x-mu)**2)
            r=from_moments(mu,var)
            self.assertLessEqual(float(p@(x>0)),float(r['cantelli'])+1e-12)
            self.assertGreaterEqual(float(r['J_live']),float(r['cantelli']))

    def test_mean_gradient_never_flat_at_fixed_variance(self):
        grad=jax.grad(lambda m:from_moments(m,.3)['J_live'])
        for mu in (-2.,-.1,0.,.1,2.):self.assertGreater(float(grad(mu)),0.)

    def test_coupled_sample_gradient_matches_difference(self):
        f=lambda x:from_samples(x)['J_live']
        x=jnp.array([-.8,-.3,.1,.2]);d=jnp.array([.2,-.1,.4,.3])
        ad=float(jnp.vdot(jax.grad(f)(x),d));h=1e-6
        fd=float((f(x+h*d)-f(x-h*d))/(2*h))
        self.assertAlmostEqual(ad,fd,places=8)

    def test_plugin_is_not_population_certificate(self):
        # Actual equally weighted law {-1,+1}; two observed -1 samples
        # yield an empirical bound far below the actual 0.5 event probability.
        self.assertLess(float(from_samples(jnp.array([-1.,-1.]))['J_live']),.01)

    def test_batch_bound_and_expectation_without_population_moments(self):
        # Cantelli also applies to each empirical distribution: R_batch >=
        # mean empirical events. Therefore E_batch R_batch >= P(D).
        values=[]
        for a in (-1.,1.):
            for b in (-1.,1.):
                x=jnp.array([a,b]);r=float(from_samples(x)['J_live'])
                self.assertGreaterEqual(r,float(jnp.mean(x>0)))
                values.append(r)
        self.assertGreaterEqual(np.mean(values),.5)

if __name__=='__main__':unittest.main()
