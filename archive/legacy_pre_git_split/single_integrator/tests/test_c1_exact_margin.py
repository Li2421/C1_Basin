import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.exact_margin import trajectory

jax.config.update('jax_enable_x64',True)


class ExactMarginTests(unittest.TestCase):
    def score(self, speed, timeout=True):
        n=40
        x=jnp.array([[-.5,0.],[.5,0.]])
        u=jnp.array([[speed,0.],[-speed,0.]])
        before=x+jnp.arange(n)[:,None,None]*.05*u
        after=before+.05*u
        return trajectory(before,after,jnp.broadcast_to(u,(n,2,2)),
            jnp.array([[1.,0.],[-1.,0.]]),jnp.ones(n,bool),terminal_timeout=timeout,
            dt=.05,max_speed=.5,goal_tolerance=.08,hold_seconds=5.,
            progress_window_seconds=2.,progress_epsilon=.01,speed_epsilon_fraction=.05)['J_live']

    def test_terminal_event_separation(self):
        # Exact endpoint displacement is speed * 39 * dt in original detector.
        for speed in (.001,.005,.009):
            self.assertGreater(float(self.score(speed)),1.)
        for speed in (.012,.02,.08):
            self.assertLess(float(self.score(speed)),1.)

    def test_derivative_matches_finite_difference_away_from_ties(self):
        speed=.005
        analytic=float(jax.grad(self.score)(speed))
        fd=float((self.score(speed+1e-6)-self.score(speed-1e-6))/2e-6)
        np.testing.assert_allclose(analytic,fd,rtol=1e-6)
        self.assertLess(analytic,0.)

    def test_no_eligible_event_zero_and_finite_derivative(self):
        fn=lambda speed:self.score(speed,False)
        self.assertEqual(float(fn(.005)),0.)
        self.assertEqual(float(jax.grad(fn)(.005)),0.)


if __name__=='__main__':unittest.main()
