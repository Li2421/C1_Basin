"""Optional C1 solver tests: run explicitly in the isolated C1 environment."""
import importlib.util
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.environment import Config, GiveWayEnv


@unittest.skipUnless(importlib.util.find_spec('cvxpylayers'), 'optional C1 SOCP environment required')
class SOCPTests(unittest.TestCase):
    def test_scan_adapter_matches_original_projection_and_adjoint(self):
        from single_integrator.c1.socp import ExactProjection
        previous=jax.config.x64_enabled;jax.config.update('jax_enable_x64',True)
        self.addCleanup(jax.config.update,'jax_enable_x64',previous)
        env=GiveWayEnv(Config());original=ExactProjection();scanned=jax.jit(original.for_scan())
        cases=[([[-2.,0.],[2.,0.]],[.1,.005,-.1,-.005]),
               ([[-.2,0.],[.2,0.]],[.5,0.,-.5,0.]),
               ([[-1.,.02],[1.,0.]],[0.,.2,0.,0.]),
               ([[-2.,0.],[2.,0.]],[.8,.01,-.7,-.01])]
        w=jnp.array([.3,-.4,.7,.2])
        for positions,target in cases:
            env.reset(positions);A,b,_=barrier_constraints(env.snapshot(),CBFConfig());x=jnp.array(target)
            expected=original(x,A,b,.5);actual=scanned(x,A,b,.5)
            np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=0.)
            original.validate(actual,A,b,.5)
            g0=jax.grad(lambda q:jnp.dot(original(q,A,b,.5),w))(x)
            g1=jax.jit(jax.grad(lambda q:jnp.dot(scanned(q,A,b,.5),w)))(x)
            np.testing.assert_allclose(g0,g1,atol=2e-7,rtol=1e-5)

    def test_reference_feasibility_and_implicit_vjp(self):
        from single_integrator.c1.socp import ExactProjection
        # Keep x64 local to this test, preserving baseline random tensor dtypes.
        with jax.experimental.enable_x64():
            env = GiveWayEnv(Config())
            solver = ExactProjection()
            cases = [([[-2.,0.],[2.,0.]], [.1,.005,-.1,-.005]),
                     ([[-.2,0.],[.2,0.]], [.5,0.,-.5,0.]),
                     ([[-1.,.02],[1.,0.]], [0.,.2,0.,0.]),
                     ([[-2.,0.],[2.,0.]], [.8,.01,-.7,-.01])]
            for positions, target in cases:
                env.reset(positions)
                A,b,_ = barrier_constraints(env.snapshot(), CBFConfig())
                x = jnp.asarray(target)
                f = lambda q: solver(q,A,b,.5)
                u = f(x)
                solver.validate(u,A,b,.5)
                ref,_ = project_velocity(x,A,b,.5,CBFConfig())
                np.testing.assert_allclose(u,ref.ravel(),atol=2e-6,rtol=0)
                w = jnp.asarray([.3,-.4,.7,.2])
                vjp = np.asarray(jax.grad(lambda q: jnp.dot(f(q),w))(x))
                eps = 1e-5
                fd = np.array([np.dot(np.asarray(f(x+eps*d)-f(x-eps*d)),w)/(2*eps)
                               for d in np.eye(4)])
                np.testing.assert_allclose(vjp,fd,atol=2e-5,rtol=2e-4)
                print('SOCP comparison',float(np.max(np.abs(u-ref.ravel()))),
                      'VJP error',float(np.max(np.abs(vjp-fd))),flush=True)

if __name__ == '__main__':
    unittest.main()
