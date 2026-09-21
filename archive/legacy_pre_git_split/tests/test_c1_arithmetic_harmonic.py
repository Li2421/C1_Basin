import unittest
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.risk.arithmetic_harmonic import conjunction,disjunction,event_upper,temporal_event,trajectory

jax.config.update('jax_enable_x64',True)


class ArithmeticHarmonicTests(unittest.TestCase):
    def test_original_strict_and_stalled_enable_conditions(self):
        def score(n,timeout,speed=0.,pad=0):
            p=jnp.ones((n+pad,2,2));u=jnp.zeros((n+pad,4)).at[:,0].set(speed)
            return trajectory(p,p,u,jnp.zeros((2,2)),jnp.arange(n+pad)<n,
                terminal_timeout=timeout,dt=.05,max_speed=.5,goal_tolerance=.08,
                hold_seconds=5.,progress_window_seconds=2.,progress_epsilon=.01,
                speed_epsilon_fraction=.05)
        self.assertEqual(float(score(40,False)['J_live']),0.)
        self.assertGreater(float(score(40,True)['J_live']),1.)
        self.assertEqual(float(score(40,True,.05)['J_live']),1.)
        self.assertGreater(float(score(140,False)['J_live']),1.)
        np.testing.assert_allclose(float(score(40,True)['J_live']),
            float(score(40,True,pad=100)['J_live']),rtol=0.,atol=1e-12)

    def test_boolean_signs_and_event_bound(self):
        rng=np.random.default_rng(2026091860)
        x=rng.uniform(-1,1,(500,12))
        x[:100]=np.abs(x[:100]);x[100:200]=-np.abs(x[100:200])
        and_scores=np.asarray(conjunction(x));or_scores=np.asarray(disjunction(x))
        np.testing.assert_array_equal(and_scores>0,np.all(x>0,axis=1))
        np.testing.assert_array_equal(or_scores>0,np.any(x>0,axis=1))
        risks=np.asarray(event_upper(or_scores))
        self.assertTrue(np.all(risks[or_scores>0]>1))
        self.assertTrue(np.all(risks[or_scores<0]<1))
        self.assertEqual(float(event_upper(0.)),1.)

    def test_dense_positive_branch_derivative_matches_finite_difference(self):
        x=jnp.array([.2,.4,.7])
        grad=np.asarray(jax.grad(conjunction)(x))
        self.assertTrue(np.all(grad>0))
        for i in range(3):
            step=np.zeros(3);step[i]=1e-6
            fd=float((conjunction(x+step)-conjunction(x-step))/(2e-6))
            self.assertAlmostEqual(fd,grad[i],places=8)
        self.assertEqual(np.count_nonzero(np.asarray(jax.grad(jnp.min)(x))),1)

    def test_boundary_continuity_and_masking(self):
        self.assertEqual(float(conjunction(jnp.array([0.,.8]))),0.)
        self.assertLess(abs(float(conjunction(jnp.array([1e-8,.8])))),2.1e-8)
        self.assertLess(abs(float(conjunction(jnp.array([-1e-8,.8])))),1e-8)
        x=jnp.array([.2,.4,-100.]);mask=jnp.array([True,True,False])
        self.assertAlmostEqual(float(conjunction(x,mask)),float(conjunction(x[:2])))
        g=jax.grad(lambda a:conjunction(a,mask))(x)
        self.assertEqual(float(g[-1]),0.)
        self.assertTrue(np.isfinite(np.asarray(g)).all())

    def test_complete_windows_and_padding(self):
        margins=jnp.array([[-.3,.2],[.4,.2],[.5,.1],[.7,.3]])
        base=temporal_event(margins,jnp.ones(4,bool),hold_samples=2)
        self.assertGreater(float(base['J_live']),1.)
        padded=temporal_event(jnp.pad(margins,((0,3),(0,0)),constant_values=100.),
            jnp.array([True]*4+[False]*3),hold_samples=2)
        self.assertEqual(float(base['J_live']),float(padded['J_live']))
        empty=temporal_event(margins,jnp.zeros(4,bool),hold_samples=2)
        self.assertEqual(float(empty['J_live']),0.)
        g=jax.grad(lambda x:temporal_event(x,jnp.zeros(4,bool),hold_samples=2)['J_live'])(margins)
        self.assertTrue(np.isfinite(np.asarray(g)).all())
        self.assertTrue(np.all(np.asarray(g)==0))


if __name__=='__main__':unittest.main()
