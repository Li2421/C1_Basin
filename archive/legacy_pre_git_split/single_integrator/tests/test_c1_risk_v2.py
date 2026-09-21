import unittest
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.risk.risk_v1 import RiskV1Config,trajectory_risk_v1
from single_integrator.c1.risk.risk_v2 import RiskV2Config,trajectory_risk_v2
from single_integrator.c1.risk.risk_function import trajectory_risk
from single_integrator.c1.training.primal_dual import deviation_cost
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.cbf import CBFConfig


class RiskV2Tests(unittest.TestCase):
    def setUp(self):
        self.goals=jnp.array([[1.,0.],[-1.,0.]])
        self.risk=RiskV2Config(window_seconds=.1,delta_prog=.02,tau_prog=.005)

    def score(self,p,c=None,length=None,code=5):
        T=len(p)-1;c=jnp.zeros(T) if c is None else c
        length=T if length is None else length
        return trajectory_risk_v2(c[None],jnp.asarray(p)[None],self.goals,self.risk,
                                  (jnp.arange(T)<length)[None],jnp.array([code]),.05)

    def test_final_action_has_progress_and_terminal_gradients(self):
        p=jnp.zeros((5,2,2)).at[:,0,0].set(jnp.arange(5)*.01).at[:,1,0].set(-jnp.arange(5)*.01)
        def f(x):return self.score(p.at[-1,0,0].set(x))['trajectory_risk'][0]
        x=.045;eps=1e-3;g=float(jax.grad(f)(x));fd=float((f(x+eps)-f(x-eps))/(2*eps))
        self.assertGreater(abs(g),1e-5)
        np.testing.assert_allclose(g,fd,atol=2e-4,rtol=2e-2)

    def test_post_terminal_padding_is_irrelevant(self):
        p=jnp.broadcast_to(self.goals,(10,2,2))
        one=self.score(p,length=1,code=1)
        changed=self.score(p.at[2:].set(100.),c=jnp.r_[0.,jnp.ones(8)],length=1,code=1)
        self.assertEqual(float(one['trajectory_risk'][0]),0.)
        self.assertEqual(float(changed['trajectory_risk'][0]),0.)
        self.assertEqual(int(changed['episode_lengths'][0]),1)

    def test_terminal_failure_not_diluted_by_easy_prefix(self):
        # Synthetic goal-distance history: steady progress, then a full-window
        # stall. This tests the risk formula, not collision-free motion planning.
        amount=jnp.r_[jnp.linspace(0.,.8,921),jnp.full(80,.8)]
        p=jnp.zeros((1001,2,2)).at[:,0,0].set(amount).at[:,1,0].set(-amount)
        self.risk=replace(self.risk,window_seconds=4.)
        risk=self.score(p)
        legacy=trajectory_risk_v1(jnp.zeros(1000),p,self.goals,RiskV1Config(),80)
        self.assertLess(float(legacy['trajectory_risk'][0]),.6)
        self.assertGreater(float(risk['trajectory_risk'][0]),float(legacy['trajectory_risk'][0]))
        self.assertGreater(float(risk['terminal_risk'][0]),.6)

    def test_warmup_geometry_not_discarded_and_short_success_measured(self):
        p=jnp.broadcast_to(self.goals,(3,2,2))
        r=self.score(p,c=jnp.array([.9,0.]),length=1,code=1)
        self.assertAlmostEqual(float(r['trajectory_risk'][0]),.225,places=6)
        self.assertEqual(int(r['progress_valid_mask'].sum()),0)

    def test_one_agent_stuck_terminal_distance_not_joint_cancellation(self):
        p=jnp.broadcast_to(self.goals,(6,2,2)).at[:,0,0].set(0.)
        r=self.score(p)
        self.assertGreater(float(r['terminal_goal_risk'][0]),.9)
        self.assertGreater(float(r['trajectory_risk'][0]),.9)

    def test_saturated_geometry_does_not_erase_goal_gradient(self):
        p=jnp.zeros((5,2,2))
        f=lambda x:self.score(p.at[-1,0,0].set(x),c=jnp.ones(4))['trajectory_risk'][0]
        gradient=float(jax.grad(f)(0.))
        self.assertLess(gradient,-1e-5)
        self.assertTrue(np.isfinite(gradient))

    def test_one_agent_progress_cannot_hide_other_unfinished_at_deadline(self):
        self.risk=replace(self.risk,window_seconds=4.)
        p=jnp.zeros((201,2,2)).at[:,0,0].set(.7).at[:,1,0].set(jnp.linspace(-.5,-1.,201))
        r=self.score(p)
        # With default weights, >=0.1m excess at deadline must rank above
        # every success, whose terminal cost is zero and exposure <= 0.25.
        self.assertGreater(float(r['trajectory_risk'][0]),.25)
        self.assertGreater(float(r['terminal_goal_risk'][0]),.5)

    def test_near_tolerance_timeout_does_not_outrank_success(self):
        success=jnp.broadcast_to(self.goals,(2,2,2))
        failure=success.at[:,0,0].set(1.-.080001)
        good=self.score(success,c=jnp.ones(1),code=1)['trajectory_risk'][0]
        bad=self.score(failure,c=jnp.zeros(1),code=5)['trajectory_risk'][0]
        self.assertGreater(float(bad),float(good))

    def test_tiny_positive_power_mean_and_episode_deviation_normalization(self):
        r=jnp.full(4,1e-15)
        self.assertAlmostEqual(float(trajectory_risk(r))/1e-15,1.,places=5)
        u=jnp.ones((2,5,4));safe=jnp.zeros_like(u)
        mask=jnp.array([[True,False,False,False,False],[True]*5])
        self.assertEqual(float(deviation_cost(u,safe,step_mask=mask)),4.)

    def test_authoritative_first_event_masks_and_bptt(self):
        from single_integrator.c1.train import rollout_terms
        class Field:
            def control(self,params,obs,noise,A,b,speed,projection):
                u=noise.astype(jnp.float64)*params
                return dict(nominal=u,safe=jnp.zeros_like(u),correction=u,candidate=u,applied=u)
        with jax.experimental.enable_x64():
            plant=Config(corridor_half_length=1.3,max_steps=6,progress_window_seconds=.075,deadlock_hold_seconds=.1)
            env=GiveWayEnv(plant)
            initial=jnp.array([env.goals,[[-.7,0.],[.7,0.]]])
            noise=jnp.zeros((2,6,4))
            u,safe,r,details=rollout_terms(1.,Field(),None,initial,noise,plant,CBFConfig(),self.risk,return_details=True)
            self.assertEqual(details['terminal_codes'].tolist(),[1,4])
            self.assertEqual(details['episode_lengths'].tolist(),[1,4])
            for i in range(2):
                env.reset(np.asarray(initial[i]));done=False
                for t in range(6):
                    _,_,done,_=env.step(np.zeros((2,2)))
                    if done:break
                self.assertEqual(int(details['episode_lengths'][i]),t+1)
            # Check derivatives survive controls/state while the discrete
            # terminal event itself has no derivative.
            noise=jnp.full((1,6,4),.001)
            f=lambda x:rollout_terms(x,Field(),None,initial[1:],noise,plant,CBFConfig(),self.risk)[2].sum()
            grad=float(jax.grad(f)(1.));eps=1e-3
            fd=float((f(1.+eps)-f(1.-eps))/(2*eps))
            self.assertTrue(np.isfinite(grad));np.testing.assert_allclose(grad,fd,atol=2e-5,rtol=2e-2)

    def test_collision_and_timeout_masks_match_environment(self):
        from single_integrator.c1.train import rollout_terms
        class Field:
            def control(self,params,obs,noise,A,b,speed,projection):
                u=noise.astype(jnp.float64)
                return dict(nominal=u,safe=u,correction=jnp.zeros_like(u),candidate=u,applied=u)
        with jax.experimental.enable_x64():
            plant=Config(corridor_half_length=1.3,max_steps=4)
            cases=[([[-.7,.025],[.7,0.]],[[0.,.4],[0.,0.]],2,1),
                   ([[-.17,0.],[.17,0.]],[[.4,0.],[-.4,0.]],3,1),
                   ([[-.7,0.],[.7,0.]],[[.02,0.],[0.,0.]],5,4)]
            for initial,action,code,length in cases:
                noise=jnp.broadcast_to(jnp.asarray(action).reshape(1,1,4),(1,4,4))
                _,_,r,d=rollout_terms(None,Field(),None,jnp.array([initial]),noise,plant,CBFConfig(),self.risk,return_details=True)
                self.assertEqual(int(d['terminal_codes'][0]),code)
                self.assertEqual(int(d['episode_lengths'][0]),length)
                if code in (2,3):self.assertEqual(float(r[0]),1.)
                env=GiveWayEnv(plant);env.reset(initial)
                for t in range(4):
                    _,_,done,_=env.step(action)
                    if done:break
                self.assertEqual(t+1,length)

    def test_censored_training_prefix_is_rejected(self):
        from single_integrator.c1.train import rollout_terms
        with self.assertRaisesRegex(ValueError,'full environment'):
            rollout_terms(None,None,None,jnp.array([[[-.7,0.],[.7,0.]]]),
                          jnp.zeros((1,4,4)),Config(),CBFConfig(),self.risk)

    def test_scanned_real_flow_socp_gradient(self):
        # CPU host callback dtype canonicalization uses the global x64 flag,
        # as set by the real trainer, not a main-thread-only context manager.
        previous=jax.config.x64_enabled
        jax.config.update('jax_enable_x64',True)
        self.addCleanup(jax.config.update,'jax_enable_x64',previous)
        from pathlib import Path
        from single_integrator.evaluate import load_policy
        from single_integrator.c1.models import ResidualCorrection
        from single_integrator.c1.differentiable_rollout import ResidualFlowField
        from single_integrator.c1.socp import ExactProjection
        from single_integrator.c1.train import rollout_terms
        with jax.experimental.enable_x64():
            base,meta=load_policy(Path('baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl'))
            plant=replace(Config(**meta['evaluation_environment']),max_steps=4)
            model=ResidualCorrection(hidden_dims=(32,32))
            params=model.init(jax.random.PRNGKey(7),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
            field=ResidualFlowField(base,model)
            field.baseline_sample=jax.jit(field.baseline_sample);field.correction=jax.jit(field.correction)
            solver=ExactProjection()
            initial=jnp.array([[[-.18,.008],[.18,-.005]]])
            noise=jax.random.normal(jax.random.PRNGKey(9),(1,4,4),dtype=jnp.float32)
            def f(x):
                tree=dict(params);weights=dict(tree['params']);out=dict(weights['zero_initialized_output'])
                out['bias']=jnp.array([x[0],x[1],-x[0],.3*x[1]])
                weights['zero_initialized_output']=out;tree['params']=weights
                u,safe,r,mask=rollout_terms(tree,field,solver,initial,noise,plant,CBFConfig(),self.risk,return_mask=True)
                return deviation_cost(u,safe,step_mask=mask)+.1*r.mean()
            x=jnp.array([.001,.003]);eps=1e-4
            g=np.asarray(jax.grad(f)(x))
            fd=np.array([(f(x+eps*d)-f(x-eps*d))/(2*eps) for d in np.eye(2)])
            self.assertTrue(np.isfinite(g).all());self.assertGreater(float(np.linalg.norm(g)),1e-6)
            np.testing.assert_allclose(g,fd,atol=5e-4,rtol=1e-2)
            print('v2 scanned real Flow/SOCP gradient',g,'FD',fd,flush=True)


if __name__=='__main__':unittest.main()
