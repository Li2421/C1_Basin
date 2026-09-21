import unittest
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from single_integrator.c1.risk.risk_function import cone_geometry, instantaneous_risk, RiskConfig

class ConeV0Tests(unittest.TestCase):
    def test_analytic_cases(self):
        with jax.experimental.enable_x64():
            cases = [([], [1., 1.], 0, np.inf),
                ([[1,0],[1,1],[0,1]], [-1.,0.], 3, 1.),
                ([[1,0],[1,1],[0,1]], [1.,1.], 3, -1.),
                ([[1,0],[1,1],[0,1]], [1.,0.], 3, 0.),
                ([[1,0]], [-1.,2.], 1, np.sqrt(5)),
                ([[1,0]], [2.,0.], 1, 0.),
                ([[1,0],[-1,0]], [2.,3.], 2, 3.),
                ([[1,0],[-1,0]], [-2.,0.], 2, 0.),
                ([[1,0],[0,1],[-1,0]], [1.,2.], 4, -2.),
                ([[1,0],[0,1],[-1,0]], [1.,-2.], 4, 2.),
                ([[1,0],[0,1],[-1,0]], [1.,0.], 4, 0.),
                ([[1,0],[-1,1],[-1,-1]], [1.,2.], 5, -np.inf)]
            for rays,q,kind,expected in cases:
                with self.subTest(rays=rays,q=q):
                    d,k = cone_geometry(jnp.array(q), jnp.array(rays,dtype=float).reshape(-1,2))
                    self.assertEqual(int(k),kind)
                    np.testing.assert_allclose(d,expected,atol=1e-10)
            r,_=instantaneous_risk(jnp.zeros((2,2)),jnp.zeros((3,2,2)),jnp.ones(3,bool),RiskConfig())
            self.assertEqual(float(r),0.)
            blocks=jnp.array([[[1.,0.],[0.,0.]],[[-1.,1.],[0.,0.]],[[-1.,-1.],[0.,0.]]])
            r,_=instantaneous_risk(jnp.ones((2,2)),blocks,jnp.ones(3,bool),RiskConfig())
            self.assertEqual(float(r),1.)

    def test_invariances(self):
        with jax.experimental.enable_x64():
            for rays in [[[1.,0.],[1.,1.],[0.,1.]], [[1.,0.],[-1.,1.],[-1.,-1.]], [[1.,0.],[-1.,0.],[0.,1.]]]:
                for q in [[-1.,.3],[.3,.7]]:
                    expected=cone_geometry(jnp.array(q),jnp.array(rays))
                    for perm in itertools.permutations(rays):
                        g=np.asarray(perm)*np.array([.1,7.,100.])[:,None]
                        g=np.vstack([g,np.zeros((3,2)),[5.,-1.]])
                        active=jnp.array([True]*6+[False])
                        d,k=cone_geometry(jnp.array(q),jnp.array(g),active)
                        np.testing.assert_allclose(d,expected[0],atol=1e-10)
                        self.assertEqual(int(k),int(expected[1]))

    def test_distance_gradients(self):
        with jax.experimental.enable_x64():
            for q in [[-1.,.4],[.4,.8]]:
                x=jnp.array(q+[1.,0.,0.,1.])
                f=lambda v: cone_geometry(v[:2],v[2:].reshape(2,2))[0]
                grad=np.asarray(jax.grad(f)(x))
                eps=1e-5
                fd=np.array([(f(x+eps*d)-f(x-eps*d))/(2*eps) for d in np.eye(6)])
                np.testing.assert_allclose(grad,fd,atol=2e-6,rtol=2e-5)

    def test_wraparound_and_agent_padding(self):
        from single_integrator.c1.risk.risk_function import risk_diagnostics
        with jax.experimental.enable_x64():
            angles=jnp.deg2rad(jnp.array([-170.,-160.,-150.]))
            rays=jnp.stack((jnp.cos(angles),jnp.sin(angles)),axis=1)
            d,k=cone_geometry(jnp.array([1.,0.]),rays)
            self.assertEqual(int(k),3);self.assertAlmostEqual(float(d),1.)
            blocks=jnp.array([[[1.,0.],[0.,0.]]])
            result=risk_diagnostics(jnp.array([[-.2,0.],[0.,0.]]),blocks,jnp.array([True]))
            self.assertEqual(float(result['agent_risk'][1]),0.)
            self.assertTrue(np.isposinf(result['margins'][1]))

class RolloutV0Tests(unittest.TestCase):
    def test_authoritative_geometry_state_jacobian(self):
        from single_integrator.c1.differentiable_rollout import barrier_constraints
        from single_integrator.cbf import CBFConfig, barrier_constraints as reference
        from single_integrator.environment import Config, GiveWayEnv
        with jax.experimental.enable_x64():
            plant=Config(corridor_half_length=1.3);env=GiveWayEnv(plant);cbf=CBFConfig()
            x=jnp.array([-.35,.017,.4,-.012])
            def tensor(v):
                A,b,_=barrier_constraints(v.reshape(2,2),jnp.array(env.walls),plant.to_dict(),cbf)
                return jnp.concatenate((A.ravel(),b))
            def oracle(v):
                A,b,_=reference(dict(positions=np.asarray(v).reshape(2,2),walls=env.walls,config=plant.to_dict()),cbf)
                return np.concatenate((A.ravel(),b))
            np.testing.assert_allclose(tensor(x),oracle(x),atol=1e-12)
            eps=1e-6
            fd=np.stack([(oracle(x+eps*d)-oracle(x-eps*d))/(2*eps) for d in np.eye(4)],axis=1)
            np.testing.assert_allclose(jax.jacrev(tensor)(x),fd,atol=1e-7,rtol=1e-5)

    def test_frozen_flow_multistep_risk_gradient_and_identity(self):
        from pathlib import Path
        from single_integrator.evaluate import load_policy
        from single_integrator.c1.train import rollout_terms, observation
        from single_integrator.c1.models import ResidualCorrection
        from single_integrator.c1.differentiable_rollout import ResidualFlowField
        from single_integrator.c1.socp import ExactProjection
        from single_integrator.cbf import CBFConfig
        from single_integrator.environment import Config, GiveWayEnv
        with jax.experimental.enable_x64():
            baseline,metadata=load_policy(Path('baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl'))
            plant=Config(**metadata['evaluation_environment']);env=GiveWayEnv(plant)
            model=ResidualCorrection(hidden_dims=(32,32))
            params=model.init(jax.random.PRNGKey(7),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
            field=ResidualFlowField(baseline,model);solver=ExactProjection();risk=RiskConfig(kappa=2.)
            initial=jnp.array([[[-.18,.008],[.18,-.005]]])
            key=jax.random.PRNGKey(1042)
            noise=jax.random.normal(key,(1,3,4),dtype=jnp.float32)
            obs=observation(initial,jnp.zeros_like(initial),jnp.array(env.goals))
            with jax.experimental.disable_x64():
                expected=baseline.sample_actions(obs.astype(jnp.float32),seed=key).reshape(1,4)
                single_noise=jax.random.normal(key,(1,4),dtype=jnp.float32)
            actual=field.baseline_sample(obs,single_noise)
            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)
            from single_integrator.c1.evaluate import C1Policy
            from single_integrator.cbf import CBFSafetyFilter
            from single_integrator.environment import bounded_nominal
            runtime=C1Policy(ResidualFlowField(baseline,model),params,plant)
            env.reset(np.asarray(initial[0]))
            candidate=np.asarray(runtime.sample_actions(obs,key))[0]
            actual_safe=CBFSafetyFilter()(env.snapshot(),candidate).velocity
            expected_safe=CBFSafetyFilter()(env.snapshot(),bounded_nominal(np.asarray(expected).reshape(2,2),plant.max_speed)).velocity
            np.testing.assert_allclose(actual_safe,expected_safe,atol=2e-6,rtol=0)
            state_gradient=jax.jacrev(lambda o:field.baseline_sample(o,single_noise))(obs)
            self.assertGreater(float(jnp.linalg.norm(state_gradient)),0.)
            u,safe,r=rollout_terms(params,field,solver,initial,noise,plant,CBFConfig(),risk)
            np.testing.assert_allclose(u,safe,atol=2e-6,rtol=0)
            self.assertGreater(float(r[0]),0.)
            def objective(phi):
                tree=dict(params);weights=dict(params['params'])
                output=dict(weights['zero_initialized_output'])
                output['bias']=jnp.array([phi[0],phi[1],-phi[0],.3*phi[1]])
                weights['zero_initialized_output']=output;tree['params']=weights
                u,safe,r=rollout_terms(tree,field,solver,initial,noise,plant,CBFConfig(),risk)
                return jnp.mean(jnp.sum((u-safe)**2,axis=-1))+.1*jnp.mean(r)
            x=jnp.array([.001,.003]);eps=1e-4
            grad=np.asarray(jax.grad(objective)(x))
            fd=np.array([(objective(x+eps*d)-objective(x-eps*d))/(2*eps) for d in np.eye(2)])
            print('FLOW MULTISTEP RISK gradient',grad,'FD',fd,'max error',float(np.max(np.abs(grad-fd))),flush=True)
            np.testing.assert_allclose(grad,fd,atol=3e-4,rtol=5e-3)

    def test_multistep_state_gradient(self):
        from single_integrator.c1.socp import ExactProjection
        from single_integrator.c1.differentiable_rollout import barrier_constraints
        from single_integrator.cbf import CBFConfig, barrier_constraints as reference
        from single_integrator.environment import Config, GiveWayEnv
        with jax.experimental.enable_x64():
            plant=Config(corridor_half_length=1.3)
            env=GiveWayEnv(plant);cbf=CBFConfig();solver=ExactProjection()
            p0=jnp.array([[-.18,.002],[.18,-.003]])
            A,b,_=barrier_constraints(p0,jnp.array(env.walls),plant.to_dict(),cbf)
            env.reset(np.asarray(p0));ar,br,_=reference(env.snapshot(),cbf)
            np.testing.assert_allclose(A,ar,atol=1e-12);np.testing.assert_allclose(b,br,atol=1e-12)
            def f(phi):
                p=p0
                for t in range(3):
                    A,b,_=barrier_constraints(p,jnp.array(env.walls),plant.to_dict(),cbf)
                    nominal=jnp.array([.2,.008,-.17,-.004])+.05*p.ravel()
                    safe=solver(nominal,A,b,.5)
                    u=solver(safe+jnp.array([phi[0],phi[1],-phi[0],.3*phi[1]]),A,b,.5)
                    p=p+plant.dt*u.reshape(2,2)
                return jnp.sum(p*jnp.array([[.3,.7],[-.2,.9]]))+.2*jnp.sum(u*u)
            x=jnp.array([.015,.01]);eps=1e-4
            g=np.asarray(jax.grad(f)(x));fd=np.array([(f(x+eps*d)-f(x-eps*d))/(2*eps) for d in np.eye(2)])
            error=float(np.max(np.abs(g-fd)))
            print('MULTISTEP gradient',g,'finite difference',fd,'max error',error,flush=True)
            np.testing.assert_allclose(g,fd,atol=3e-5,rtol=3e-3)

if __name__=='__main__':
    unittest.main()
