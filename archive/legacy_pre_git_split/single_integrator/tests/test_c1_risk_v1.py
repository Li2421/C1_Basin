import unittest
import numpy as np
import jax
import jax.numpy as jnp

from single_integrator.c1.risk.risk_v1 import RiskV1Config, task_potential, trajectory_risk_v1


class RiskV1Tests(unittest.TestCase):
    def cfg(self):
        return RiskV1Config(window_seconds=.2, delta_prog=.03, tau_prog=.005,
                            task_mask_temperature=.02, goal_tolerance=.08, kappa=2.)

    def trace(self, progress, cone=.1, retreat=False):
        # Two agents begin one metre from opposite goals. Each time step moves
        # both symmetrically toward them; optional early retreat tests a full window.
        T=20; x=np.zeros((T+1,2,2)); goals=np.array([[1.,0.],[-1.,0.]])
        for t in range(T+1):
            amount=progress*t
            if retreat and t < 4: amount=-.02*t
            elif retreat: amount=-.08+progress*(t-4)
            x[t]=[[amount,0.],[-amount,0.]]
        return jnp.full(T,cone),jnp.asarray(x),jnp.asarray(goals)

    def report(self, cone, pos, goals):
        return trajectory_risk_v1(cone,pos,goals,self.cfg(),4)

    def test_semantic_ordering(self):
        success=self.report(*self.trace(.04,cone=.05))
        yielding=self.report(*self.trace(.035,cone=.05,retreat=True))
        slow=self.report(*self.trace(.01,cone=.05))
        stalled=self.report(*self.trace(0.,cone=.05))
        slack_stalled=self.report(*self.trace(0.,cone=0.))
        weak_cone=self.report(*self.trace(.04,cone=.9))
        for value in (success,yielding,slow,stalled,slack_stalled,weak_cone):
            self.assertTrue(np.all(np.isfinite(value['trajectory_risk'])))
        self.assertLess(float(success['trajectory_risk'][0]),float(stalled['trajectory_risk'][0]))
        self.assertLess(float(yielding['trajectory_risk'][0]),float(stalled['trajectory_risk'][0]))
        self.assertGreater(float(slack_stalled['trajectory_risk'][0]),.9)
        self.assertGreater(float(weak_cone['trajectory_risk'][0]),float(success['trajectory_risk'][0]))
        self.assertTrue(np.allclose(np.asarray(stalled['cone_risk_t']),.05))
        self.assertTrue(np.all(np.isnan(np.asarray(stalled['stall_risk_t'])[0,:4])))

    def test_potential_goal_and_window(self):
        cfg=self.cfg();goal=jnp.array([[1.,0.],[-1.,0.]])
        p=jnp.stack((jnp.array([[-1.,0.],[1.,0.]]),goal))
        v=task_potential(p,goal,p[0],cfg)
        self.assertAlmostEqual(float(v[0]),1.,places=5)
        self.assertLess(float(v[1]),1e-4)
        self.assertEqual(cfg.window_steps(.05),4)
        with self.assertRaises(ValueError):RiskV1Config(window_seconds=.23).window_steps(.05)

    def test_progress_and_cone_gradients(self):
        cfg=self.cfg(); cone,pos,goals=self.trace(.01,cone=.2)
        def f(z):
            # Risk uses pre-control states; the terminal state is outside its
            # time support. Perturb a state that actually enters progress.
            candidate=pos.at[-2,0,0].set(z[0])
            c=cone.at[8].set(z[1])
            return trajectory_risk_v1(c,candidate,goals,cfg,4)['trajectory_risk'][0]
        x=jnp.array([.18,.2]);eps=1e-4
        grad=np.asarray(jax.grad(f)(x));fd=np.array([(f(x+eps*d)-f(x-eps*d))/(2*eps) for d in np.eye(2)])
        self.assertGreater(abs(grad[0]),1e-5)
        self.assertGreater(abs(grad[1]),1e-5)
        np.testing.assert_allclose(grad,fd,atol=1e-3,rtol=2e-2)

    def test_one_unfinished_agent_cannot_be_hidden_by_the_other(self):
        positions=jnp.broadcast_to(jnp.array([[.12,0.],[0.,0.]]),(5,2,2))
        args=(jnp.zeros(4),positions,jnp.zeros((2,2)))
        current=trajectory_risk_v1(*args,RiskV1Config(window_seconds=.05),1)
        legacy=trajectory_risk_v1(*args,RiskV1Config(window_seconds=.05,unfinished_mode='joint_sum'),1)
        self.assertGreater(float(current['unfinished_mask'][0,1]),.85)
        self.assertLess(float(legacy['unfinished_mask'][0,1]),.15)

    def test_zero_risk_has_finite_gradient(self):
        cfg=RiskV1Config(window_seconds=.05,delta_prog=.02,tau_prog=1e-5)
        positions=jnp.array([[[0.,0.],[0.,0.]],[[.4,0.],[-.4,0.]],
                             [[.8,0.],[-.8,0.]],[[1.,0.],[-1.,0.]]])
        goals=jnp.array([[1.,0.],[-1.,0.]])
        f=lambda c:trajectory_risk_v1(c,positions,goals,cfg,1)['trajectory_risk'].sum()
        risk=f(jnp.zeros(3));gradient=jax.grad(f)(jnp.zeros(3))
        self.assertEqual(float(risk),0.)
        self.assertTrue(np.isfinite(gradient).all())

    def test_real_flow_multistep_bptt_gradient(self):
        from pathlib import Path
        from single_integrator.evaluate import load_policy
        from single_integrator.c1.models import ResidualCorrection
        from single_integrator.c1.differentiable_rollout import ResidualFlowField
        from single_integrator.c1.train import rollout_terms
        from single_integrator.c1.socp import ExactProjection
        from single_integrator.environment import Config
        from single_integrator.cbf import CBFConfig
        with jax.experimental.enable_x64():
            base, meta = load_policy(Path('baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl'))
            model=ResidualCorrection(hidden_dims=(32,32))
            params=model.init(jax.random.PRNGKey(7),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
            field=ResidualFlowField(base,model);field.baseline_sample=jax.jit(field.baseline_sample);field.correction=jax.jit(field.correction)
            initial=jnp.array([[[-.18,.008],[.18,-.005]]]);noise=jax.random.normal(jax.random.PRNGKey(9),(1,4,4),dtype=jnp.float32)
            risk=RiskV1Config(window_seconds=.1,delta_prog=.001,tau_prog=.005,kappa=2.)
            def f(x):
                tree=dict(params); weights=dict(tree['params']); out=dict(weights['zero_initialized_output'])
                out['bias']=jnp.array([x[0],x[1],-x[0],.3*x[1]]);weights['zero_initialized_output']=out;tree['params']=weights
                u,safe,r=rollout_terms(tree,field,ExactProjection(),initial,noise,Config(**meta['evaluation_environment']),CBFConfig(),risk)
                return jnp.mean(jnp.sum((u-safe)**2,axis=-1))+.1*jnp.mean(r)
            x=jnp.array([.001,.003]);eps=1e-4;g=np.asarray(jax.grad(f)(x))
            fd=np.array([(f(x+eps*d)-f(x-eps*d))/(2*eps) for d in np.eye(2)])
            print('V1 multistep gradient',g,'FD',fd,'error',np.max(np.abs(g-fd)),flush=True)
            np.testing.assert_allclose(g,fd,atol=5e-4,rtol=1e-2)

if __name__=='__main__':unittest.main()
