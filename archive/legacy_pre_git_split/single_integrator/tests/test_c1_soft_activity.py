import unittest
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from single_integrator.c1.risk.soft_activity import SoftRiskConfig,soft_risk_diagnostics
from single_integrator.c1.risk.risk_function import risk_diagnostics

class SoftActivityTests(unittest.TestCase):
    def fixture(self,h,s,cfg=None):
        return soft_risk_diagnostics(jnp.array([[-.1,.03],[.1,.03]]),
            jnp.array([[[1.,0.],[0.,0.]]]),jnp.array([h]),jnp.array([s]),cfg or SoftRiskConfig(kappa=2.))

    def test_residual_sweep_and_hard_semantics(self):
        with jax.experimental.enable_x64():
            cfg=SoftRiskConfig(kappa=2.)
            values=[]
            for s in [0.,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,.01,.02]:
                d=self.fixture(.04,s,cfg);values.append(float(d['risk']))
                self.assertEqual(bool(d['active_mask'][0]),s<=cfg.active_tol)
                old=risk_diagnostics(jnp.array([[-.1,.03],[.1,.03]]),jnp.array([[[1.,0.],[0.,0.]]]),d['active_mask'],cfg)
                np.testing.assert_allclose(d['hard_risk'],old['risk'])
                np.testing.assert_array_equal(d['hard_cone_types'],old['cone_types'])
                self.assertGreater(float(d['risk']),0.)
                np.testing.assert_allclose(d['risk'],self.fixture(.04,s,replace(cfg,active_tol=.1))['risk'])
            self.assertLess(abs(values[0]-values[3]),1e-7)
            self.assertTrue(np.all(np.diff(values)<=0))
            print('RESIDUAL SWEEP',values,flush=True)

    def test_h_boundary_continuity_and_locality(self):
        with jax.experimental.enable_x64():
            values=[float(self.fixture(h,.001)['risk']) for h in [.049,.04999999,.05,.05000001,.051]]
            self.assertLess(abs(values[1]-values[3]),1e-6)
            self.assertGreater(values[-1],0.)
            self.assertEqual(float(self.fixture(10.,0.)['risk']),0.)
            print('H SWEEP',values,flush=True)
            for key in ['tau_h','tau_s']:
                with self.assertRaises(ValueError):SoftRiskConfig(**{key:0.})

    def test_soft_gradients(self):
        with jax.experimental.enable_x64():
            cfg=SoftRiskConfig(kappa=2.)
            def f(v):
                return soft_risk_diagnostics(jnp.array([[v[2],v[3]],[.1,.03]]),
                    jnp.array([[[1.,0.],[0.,0.]]]),v[:1],v[1:2],cfg)['risk']
            x=jnp.array([.04,.004,-.1,.03]);eps=1e-6
            g=np.asarray(jax.grad(f)(x))
            fd=np.array([(f(x+eps*d)-f(x-eps*d))/(2*eps) for d in np.eye(4)])
            np.testing.assert_allclose(g,fd,atol=1e-6,rtol=1e-5)
            print('SOFT GRAD h,s,F error',np.max(np.abs(g-fd)),flush=True)

    def test_full_plane_and_empty_agent(self):
        with jax.experimental.enable_x64():
            empty=soft_risk_diagnostics(jnp.ones((2,2)),jnp.zeros((0,2,2)),jnp.zeros(0),jnp.zeros(0))
            self.assertEqual(float(empty['risk']),0.)
            blocks=jnp.array([[[1.,0.],[0.,0.]],[[-1.,1.],[0.,0.]],[[-1.,-1.],[0.,0.]]])
            d=soft_risk_diagnostics(jnp.ones((2,2)),blocks,jnp.full(3,.05),jnp.full(3,.01))
            np.testing.assert_allclose(d['agent_risk'],[.5*np.exp(-1),0.])
            self.assertEqual(int(d['cone_types'][0]),5)

    def test_multistep_rollout_gradient(self):
        from pathlib import Path
        from single_integrator.evaluate import load_policy
        from single_integrator.c1.models import ResidualCorrection
        from single_integrator.c1.differentiable_rollout import ResidualFlowField
        from single_integrator.c1.train import rollout_terms
        from single_integrator.c1.socp import ExactProjection
        from single_integrator.environment import Config
        from single_integrator.cbf import CBFConfig
        with jax.experimental.enable_x64():
            base,meta=load_policy(Path('baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl'))
            model=ResidualCorrection(hidden_dims=(32,32))
            params=model.init(jax.random.PRNGKey(7),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
            field=ResidualFlowField(base,model);solver=ExactProjection()
            field.baseline_sample=jax.jit(field.baseline_sample)
            field.correction=jax.jit(field.correction)
            p=jnp.array([[[-.18,.008],[.18,-.005]]])
            noise=jax.random.normal(jax.random.PRNGKey(1042),(1,3,4),dtype=jnp.float32)
            def f(phi):
                tree=dict(params);w=dict(tree['params']);o=dict(w['zero_initialized_output'])
                o['bias']=jnp.array([phi[0],phi[1],-phi[0],.3*phi[1]])
                w['zero_initialized_output']=o;tree['params']=w
                u,safe,r=rollout_terms(tree,field,solver,p,noise,Config(**meta['evaluation_environment']),CBFConfig(),SoftRiskConfig(kappa=2.))
                return jnp.mean(jnp.sum((u-safe)**2,axis=-1))+.1*jnp.mean(r)
            x=jnp.array([.001,.003]);eps=1e-4
            g=np.asarray(jax.grad(f)(x));fd=np.array([(f(x+eps*d)-f(x-eps*d))/(2*eps) for d in np.eye(2)])
            print('SOFT MULTISTEP gradient',g,'FD',fd,'error',np.max(np.abs(g-fd)),flush=True)
            np.testing.assert_allclose(g,fd,atol=3e-4,rtol=5e-3)

if __name__=='__main__':unittest.main()
