"""Cheap analytic base fields inside the existing FF integration operator.

Only the base vector field is changed. Safety, eta lifting, physical simulator,
the ten pseudo-time steps, native noise scaling and termination are inherited.
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path
import numpy as np

FIELD = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')


def native_runtime(scene):
    sys.path.insert(0, str(FIELD))
    from validation_stage2.run_rollouts import make_runtime
    return make_runtime(scene)


def bounded(a, speed):
    return a*np.minimum(1., speed/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-30))


class AnalyticField:
    def __init__(self, base, parameters):
        self.base = base
        self.parameters = np.asarray(parameters, float)
        assert self.parameters.shape == (6,)
        self.steps = base.steps
        self.agent = base.agent
        self.scale, self.mean = base.scale, base.mean
        self.config, self.scenario, self.basis = base.config, base.scenario, base.basis

    def __getattr__(self, name):
        return getattr(self.base, name)

    def target(self, env):
        speed, brake, turn, repel, damping, relaxation = self.parameters
        p, g, v = env.positions, env.goals, env.velocities
        delta = g-p
        dist = np.linalg.norm(delta,axis=-1,keepdims=True)
        direction = delta/np.maximum(dist,1e-12)
        desired = speed*self.config.max_speed*np.tanh(dist/brake)*direction-damping*v
        # Agent-relative avoidance; no scene or controller identity is used.
        dp=p[:,None,:]-p[None,:,:]
        dd=np.linalg.norm(dp,axis=-1)
        proximity=np.exp(-np.square(dd/.65))*(1-np.eye(len(p)))
        radial=dp/np.maximum(dd[...,None],1e-12)
        lateral=np.stack((-radial[...,1],radial[...,0]),-1)
        desired += self.config.max_speed*np.sum(proximity[...,None]*(repel*radial+turn*lateral),axis=1)
        # Circular obstacle avoidance uses the actual map, not a scene label.
        if hasattr(self.config,'obstacle_radius'):
            center=np.asarray(getattr(self.config,'obstacle_center',(0.,0.)),float)
            d=p-center;rr=np.linalg.norm(d,axis=-1,keepdims=True)
            radial=d/np.maximum(rr,1e-12)
            influence=np.exp(-np.maximum(rr-self.config.obstacle_radius-self.config.agent_radius,0.)/.45)
            desired += self.config.max_speed*influence*(repel*radial+turn*np.stack((-radial[:,1],radial[:,0]),-1))
        return bounded(desired,self.config.max_speed)

    def velocity_fn(self, obs, z, time):
        h=self.to_world(self.mean+self.scale*np.asarray(z),self._env)
        w=self.parameters[5]*(self._target-h)
        return (self.to_local(w,self._env)/self.scale).ravel()

    def nominal(self, env, key, traced=False):
        import jax
        self._env=env
        self._target=self.target(env)
        obs=self.observation(env)
        z0=np.asarray(jax.random.normal(key,(1,self.agent.config['joint_act_dim']))).reshape(-1,2)
        h=self.to_world(self.mean+self.scale*z0,env)
        path=[h.copy()]
        for _ in range(self.steps):
            h=h+self.parameters[5]*(self._target-h)/self.steps
            path.append(h.copy())
        local=np.clip(self.to_local(h,env),-1.,1.)
        flow=self.to_world(bounded(local,self.config.max_speed),env)
        return obs,flow,np.asarray(path) if traced else None

    def prepare(self, env, key, eta, traced=False):
        from poc.field_controller import Runtime
        return Runtime.prepare(self,env,key,eta,traced)

    def integrate(self, env, obs, key, delta, safety, **kwargs):
        from poc.field_controller import Runtime
        return Runtime.integrate(self,env,obs,key,delta,safety,**kwargs)

    def action(self, env, key, eta, chain='field'):
        from poc.field_controller import Runtime
        assert chain=='field'
        return Runtime.action(self,env,key,eta,chain)


def controller_context(runtime, state):
    """Policy-function queries only: current and goal-neighborhood motion.

    No environment step, eta query, rollout outcome, scene label, or checkpoint
    identity enters this descriptor. Agent association is retained.
    """
    import jax
    env=runtime.make_env(state)
    p0,g0,v0=env.positions.copy(),env.goals.copy(),env.velocities.copy()
    e=g0-p0;e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
    lat=np.stack((-e[:,1],e[:,0]),-1)
    probes=[(p0,v0),(p0,.5*runtime.config.max_speed*e),
            (p0,-.5*runtime.config.max_speed*e),
            (g0-.25*e,np.zeros_like(v0)),
            (g0-.25*e,.5*runtime.config.max_speed*e),
            (g0-.25*e,-.5*runtime.config.max_speed*e),
            (g0-.25*e,.5*runtime.config.max_speed*lat)]
    values=[]
    for positions,velocities in probes:
        env.positions[:]=positions;env.velocities[:]=velocities
        draws=[]
        for k in range(4):
            key=jax.random.fold_in(jax.random.PRNGKey(2026100907),k)
            _,u,_=runtime.nominal(env,key)
            draws.append(np.stack(((u*e).sum(-1),(u*lat).sum(-1)),-1)/runtime.config.max_speed)
        values.append(np.mean(draws,axis=0))
    env.positions[:]=p0;env.velocities[:]=v0
    return np.concatenate(values,axis=-1).astype(np.float32)
