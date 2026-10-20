"""Locked direct-gradient MPC. Plant and projection stay in the old runtime."""
from common import *
from legacy_controllers import Nominal,bound
import jax,jax.numpy as jnp,optax
from functools import lru_cache
from types import SimpleNamespace
jax.config.update('jax_enable_x64',False)
@jax.custom_jvp
def norm2(x):return jnp.sqrt(jnp.sum(x*x,axis=-1))
@norm2.defjvp
def norm2_jvp(primals,tangents):
    x,=primals;dx,=tangents;r=jnp.sqrt(jnp.sum(x*x,axis=-1));derivative=jnp.sum(x*dx,axis=-1)/jnp.where(r>0,r,1.)
    return r,jnp.where(r>0,derivative,0.)
def shooting_cost_jnp(positions,actions,goals,cfg,weight,previous):
    goal=jnp.square(positions-goals).sum(-1).mean((1,2))
    radius=norm2(positions)
    inner=jnp.maximum(cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin-radius,0)**2
    outer=jnp.maximum(radius-(cfg.outer_radius-cfg.agent_radius-cfg.collision_margin),0)**2
    ii,jj=jnp.triu_indices(4,1);dd=norm2(positions[:,:,ii]-positions[:,:,jj])
    pairs=jnp.maximum(2*cfg.agent_radius+cfg.agent_collision_margin-dd,0)**2
    penalty=inner.mean((1,2))+outer.mean((1,2))+pairs.mean((1,2))
    changes=jnp.diff(jnp.concatenate((jnp.broadcast_to(previous,(len(actions),1,4,2)),actions),axis=1),axis=1)
    smooth=jnp.square(changes).sum(-1).mean((1,2))
    return goal+weight*penalty+.05*smooth

def interpolation(horizon,dtype=jnp.float32):
    x=np.linspace(0,2,horizon);w=np.zeros((horizon,3),float)
    for i,v in enumerate(x):
        lo=min(int(v),1);w[i,lo]=1-(v-lo);w[i,lo+1]=v-lo
    return jnp.asarray(w,dtype)
def linear_actions(knots,horizon):return jnp.einsum('hk,kad->had',interpolation(horizon,knots.dtype),knots)
def clip_actions(a,speed):return a*jnp.minimum(1.,speed/jnp.maximum(norm2(a)[...,None],1e-30))
@lru_cache(None)
def optimizer(horizon,iterations,rate,fields):
    cfg=SimpleNamespace(**dict(fields));opt=optax.adam(rate)
    @jax.jit
    def optimize(knots,initial,goals,previous):
        def loss(k):
            actions=linear_actions(k,horizon);positions=initial[None]+cfg.dt*jnp.cumsum(actions,axis=0)
            return shooting_cost_jnp(positions[None],actions[None],goals,cfg,100,previous)[0]
        def update(carry,_):
            k,st=carry;grad=jax.grad(loss)(k);delta,st=opt.update(grad,st,k);k=clip_actions(optax.apply_updates(k,delta),cfg.max_speed);return (k,st),None
        (result,_),_=jax.lax.scan(update,(knots,opt.init(knots)),None,length=iterations)
        return result,linear_actions(result,horizon),loss(result)
    return optimize
class GradientMPC(Nominal):
    def __init__(self,parameters,manifest):
        super().__init__('gradient_mpc',parameters,manifest);self.episode_env=None;self.warm=None;self.last_step=None;self.last_action=None;self.optimizations=0
    def nominal(self,env,key,traced=False):
        assert not traced
        if env is not self.episode_env:
            self.episode_env=env;self.warm=None;self.last_step=None;self.last_action=None;self.optimizations=0
        if self.last_step==env.step_count:return None,self.last_action.copy(),None
        h,k,rate=self.parameters;cfg=env.config
        if self.warm is None:self.warm=np.broadcast_to(bound(env.goals-env.positions,cfg.max_speed),(3,4,2)).astype(np.float32).copy()
        fields=tuple((n,float(getattr(cfg,n))) for n in ('dt','obstacle_radius','agent_radius','collision_margin','outer_radius','agent_collision_margin','max_speed'))
        knots,actions,cost=optimizer(int(h),int(k),float(rate),fields)(jnp.asarray(self.warm),jnp.asarray(env.positions,jnp.float32),jnp.asarray(env.goals,jnp.float32),jnp.asarray(env.velocities,jnp.float32))
        actions=np.asarray(actions);assert np.isfinite(actions).all() and np.isfinite(cost)
        shifted=np.concatenate((actions[1:],actions[-1:]),axis=0);times=np.linspace(0,h-1,3);self.warm=np.asarray([[np.interp(times,np.arange(h),shifted[:,a,d]) for d in range(2)] for a in range(4)],np.float32).transpose(2,0,1)
        self.last_action=bound(actions[0].astype(np.float64).copy(),cfg.max_speed);self.last_step=env.step_count;self.optimizations+=1
        return None,self.last_action.copy(),None
