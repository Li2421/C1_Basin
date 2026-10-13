"""Independent scenario adapters; frozen source/ and original repository are read-only."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from poc.field_controller import ROOT, Runtime, FixedSafetySet
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import HardProjectionConfig
from single_integrator.cbf import CBFConfig, barrier_constraints, CBFSolverError
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry

ASSETS = ROOT / 'validation_v2/assets'

class ToySafetySet:
    def __init__(self, snapshot):
        self.config = CBFConfig()
        self.A, self.lower, self.geometry = barrier_constraints(snapshot, self.config)
        self.max_speed = snapshot['config']['max_speed']
        self.calls = 0
        if max(self.config.gamma, self.config.gamma_wall)*snapshot['config']['dt'] > 1:
            raise ValueError('held-action safety condition violated')
    def certificate(self, a):
        a = np.asarray(a)
        residual = self.A @ a.ravel() - self.lower
        excess = np.linalg.norm(a, axis=1).max() - self.max_speed
        return {'min_linear_residual': float(residual.min()), 'max_speed_excess': float(excess),
                'max_violation': float(max(0.,-residual.min(),excess))}
    def assert_safe(self, a):
        d = self.certificate(a)
        if not np.isfinite(a).all() or d['min_linear_residual'] < -self.config.feasibility_tol or d['max_speed_excess'] > self.config.speed_tol:
            raise CBFSolverError('unsafe_action', d)
        return d
    def project(self, target):
        self.calls += 1
        a, status, retried, detail = project_velocity_with_retry(target,self.A,self.lower,self.max_speed,self.config)
        self.assert_safe(a)
        return a, {'status':status, 'retried':retried, 'detail':detail}
    def project_field_target(self,target):
        a=np.asarray(target,dtype=np.float64).copy()
        a*=np.minimum(1.,self.max_speed/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-30))
        if (self.A@a.ravel()-self.lower).min() >= 0:
            self.calls+=1
            d=self.assert_safe(a)
            return a,{**d,'status':'exact_speed_ball_projection_linear_feasible'}
        return self.project(target)

class OtherRuntime(Runtime):
    def __init__(self,scenario):
        self.scenario=scenario
        if scenario=='toy_give_way':
            from single_integrator.evaluate import load_policy
            from single_integrator.environment import Config, GiveWayEnv
            self.checkpoint=ASSETS/'toy_ckpt.pkl'
            self.agent, provenance=load_policy(self.checkpoint)
            self.config=Config(**provenance['evaluation_environment'])
            self.env_cls=GiveWayEnv
            self.cbf=CBFConfig()
        elif scenario=='double_bottleneck':
            from double_bottleneck.environment import Config, DoubleBottleneckEnv
            from double_bottleneck.flowbc_4a_agent import load_checkpoint
            self.checkpoint=ASSETS/'db_ckpt.pkl'
            with self.checkpoint.open('rb') as f: payload=pickle.load(f)
            manifest=json.loads((ASSETS/'db_manifest.json').read_text())
            self.config=Config(**manifest['environment']['config'])
            fingerprint=manifest['environment_fingerprint']
            if fingerprint != payload['config']['environment_fingerprint']:
                raise ValueError('DB environment/checkpoint mismatch')
            self.agent,_=load_checkpoint(self.checkpoint,fingerprint)
            self.env_cls=DoubleBottleneckEnv
            self.cbf=HardProjectionConfig()
        else: raise ValueError(scenario)
        self.basis=get_basis_family('orthoflow3')
        c=self.agent.config
        c=c.copy(add_or_replace={'joint_obs_dim':20 if scenario=='toy_give_way' else 72,
                                 'joint_act_dim':4 if scenario=='toy_give_way' else 8})
        self.agent=self.agent.replace(config=c)
        if not c['normalize']: raise ValueError('normalized checkpoint required')
        self.steps=int(c['flow_steps'])
        self.scale=np.asarray(c['act_scale'],dtype=np.float64).reshape(-1,2)
        self.mean=np.asarray(c['act_mean'],dtype=np.float64).reshape(-1,2)
        if (self.scale<=0).any(): raise ValueError('noninvertible normalization')
        position_only=bool(c.get('position_only',False))
        mask=jnp.asarray([1,1,0,0,1,1,1,1,0,0]*2) if scenario=='toy_give_way' else None
        def flatten(obs):
            x=obs.reshape(1,c['joint_obs_dim'])
            x=(x-jnp.asarray(c['obs_mean']))/jnp.asarray(c['obs_scale'])
            return x*mask if position_only else x
        @jax.jit
        def trace(obs,key):
            flat=flatten(obs)
            z=jax.random.normal(key,(1,c['joint_act_dim']))
            path=[z[0]]
            for k in range(self.steps):
                t=jnp.full((1,1),k/self.steps)
                z=z+self.agent.network.select('actor_bc_flow')(flat,z,t)/self.steps
                path.append(z[0])
            return jnp.stack(path)
        @jax.jit
        def velocity(obs,z,s):
            return self.agent.network.select('actor_bc_flow')(flatten(obs),z.reshape(1,-1),s.reshape(1,1))[0]
        @jax.jit
        def native(obs,key):
            raw=self.agent.sample_actions(obs[None],key)[0]
            norm=jnp.linalg.norm(raw,axis=1,keepdims=True)
            return raw*jnp.minimum(1.,self.config.max_speed/jnp.maximum(norm,1e-30))
        self.trace_fn,self.velocity_fn,self.native_fn=trace,velocity,native
        self.manifest={'scenario_config':self.config.to_dict()}
    def make_env(self,state):
        env=self.env_cls(self.config)
        p=state['physical']
        if self.scenario=='toy_give_way': env.reset(np.asarray(p['positions']))
        else:
            env.reset(np.asarray(p['positions']),regime=state['metadata']['regime'])
            env.velocities=np.asarray(p['velocities'],dtype=np.float64).copy()
        np.testing.assert_allclose(env.goals,p['goals'],rtol=0,atol=1e-6)
        if p.get('timestep',0)!=0: raise ValueError('non-t0 state needs monitor restoration')
        return env
    def observation(self,env): return env.observation()
    def to_world(self,a,env): return np.asarray(a)
    def to_local(self,a,env): return np.asarray(a)
    def safety(self,env):
        return ToySafetySet(env.snapshot()) if self.scenario=='toy_give_way' else FixedSafetySet(env.snapshot(),self.cbf)
    def nominal(self,env,key,traced=False):
        obs=jnp.asarray(self.observation(env))
        raw=np.asarray(self.native_fn(obs,key),dtype=np.float64)
        flow=raw.copy()
        flow*=np.minimum(1.,self.config.max_speed/np.maximum(np.linalg.norm(flow,axis=1,keepdims=True),1e-30))
        path=None
        if traced:
            zpath=np.asarray(self.trace_fn(obs,key)).reshape(self.steps+1,-1,2)
            path=np.asarray([self.mean+self.scale*z for z in zpath])
            decoded=np.clip(path[-1],-1.,1.)
            decoded*=np.minimum(1.,self.config.max_speed/np.maximum(np.linalg.norm(decoded,axis=1,keepdims=True),1e-30))
            np.testing.assert_allclose(decoded,flow,rtol=1e-6,atol=1e-7)
        return obs,flow,path

class FourRingRuntime(Runtime):
    def nominal(self,env,key,traced=False):
        obs,flow,_=super().nominal(env,key,traced=False)
        path=None
        if traced:
            zpath=np.asarray(self.trace_fn(obs,key)).reshape(self.steps+1,-1,2)
            path=np.stack([self.to_world(self.mean+self.scale*z,env) for z in zpath])
            decoded=np.clip(zpath[-1]*self.scale+self.mean,-1.,1.)
            decoded*=np.minimum(1.,self.config.max_speed/np.maximum(np.linalg.norm(decoded,axis=1,keepdims=True),1e-30))
            decoded=self.to_world(decoded,env)
            np.testing.assert_allclose(decoded,flow,rtol=1e-6,atol=1e-7)
        return obs,flow,path
