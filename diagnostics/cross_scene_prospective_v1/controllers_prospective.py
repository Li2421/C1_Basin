"""Eight nominal optimizers; only wall/agent-count inputs are generalized.

Algorithms and hyperparameter tuples follow the frozen 48-row zoo manifest.
Sources: cross-type-six controllers.py, legacy_controllers.py,
mixed_controllers.py, historical_controllers.py, gradient_mpc.py.
"""
from experiment import *
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import lru_cache
jax.config.update('jax_enable_x64',False)

def bound(a,speed):return a*np.minimum(1.,speed/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-30))
def rng_for(key):return np.random.default_rng(int.from_bytes(hashlib.sha256(np.asarray(key,np.uint32).tobytes()).digest()[:8],'little'))

def wall_penalties(p,env,xp=np):
    c=env.config;scene=c.scene;margin=c.agent_radius+c.wall_collision_margin+c.wall_radius
    zero=xp.zeros(p.shape[:-1])
    if scene.startswith(('xsc_','xsp_')):
        walls=xp.asarray(env.walls)
        if not len(walls):return zero,zero
        a=walls[:,0];ab=walls[:,1]-walls[:,0];d=p[...,None,:]-a
        t=xp.clip(xp.sum(d*ab,axis=-1)/xp.sum(ab*ab,axis=-1),0,1)
        delta=d-t[...,None]*ab
        distance=xp.sqrt(xp.maximum(xp.sum(delta*delta,axis=-1),1e-30))
        return xp.maximum(margin-xp.min(distance,axis=-1),0)**2,zero
    if scene.startswith('lit_swap8_'):
        from swap_geometry_user_fix_20261023 import static_walls
        walls=static_walls(scene)
        if not len(walls):return zero,zero
        a=xp.asarray(walls[:,0]);ab=xp.asarray(walls[:,1]-walls[:,0]);d=p[...,None,:]-a
        t=xp.clip(xp.sum(d*ab,axis=-1)/xp.sum(ab*ab,axis=-1),0,1)
        delta=d-t[...,None]*ab
        distance=xp.sqrt(xp.maximum(xp.sum(delta*delta,axis=-1),1e-30))
        return xp.maximum(margin-xp.min(distance,axis=-1),0)**2,zero
    if scene=='lit_narrow_passage':return zero,zero
    if scene=='lit_crowded_random':
        clear=xp.minimum(xp.minimum(p[...,0],p[...,1]),xp.minimum(2-p[...,0],2-p[...,1]))-margin
        return zero,xp.maximum(-clear,0)**2
    if scene=='lit_circle_five':
        # Every 128-gon normal has unit norm to floating-point precision.
        # With this conservative interior guard, every old dot product is
        # strictly below 11-margin, so its positive-part penalty is exactly +0.
        # JAX and all other dtypes retain the original expression.
        if xp is np and p.dtype==np.float64 and np.all(np.sum(p*p,axis=-1)<=(11-margin-1e-10)**2):
            return zero,zero
        normals=xp.asarray(np.stack([np.cos(np.arange(128)*2*np.pi/128),np.sin(np.arange(128)*2*np.pi/128)],-1))
        return zero,xp.maximum(xp.max(p@normals.T,axis=-1)-(11-margin),0)**2
    values=[]
    for r in ((.9,.3,1.1,1.1),(.9,-1.,1.1,-.2)):
        center=xp.asarray([(r[0]+r[2])/2,(r[1]+r[3])/2]);half=xp.asarray([(r[2]-r[0])/2,(r[3]-r[1])/2]);q=xp.abs(p-center)-half
        outside=xp.sqrt(xp.maximum(xp.sum(xp.maximum(q,0)**2,axis=-1),1e-30));inside=xp.minimum(xp.max(q,axis=-1),0)
        values.append(outside+inside-margin)
    return xp.maximum(-xp.minimum(values[0],values[1]),0)**2,zero

def shooting_cost(positions,actions,goals,env,weight,previous):
    c=env.config;goal=np.square(positions-goals).sum(-1).mean((1,2));inner,outer=wall_penalties(positions,env)
    ii,jj=np.triu_indices(len(goals),1);dd=np.linalg.norm(positions[:,:,ii]-positions[:,:,jj],axis=-1)
    pairs=np.maximum(2*c.agent_radius+c.agent_collision_margin-dd,0)**2
    penalty=inner.mean((1,2))+outer.mean((1,2))+(pairs.mean((1,2)) if len(ii) else 0.)
    changes=np.diff(np.concatenate((np.broadcast_to(previous,(len(actions),1,len(goals),2)),actions),axis=1),axis=1)
    return goal+weight*penalty+.05*np.square(changes).sum(-1).mean((1,2))

def scoped_cost(positions,actions,goal,env,weight,previous,forecast):
    c=env.config;distance=np.square(positions-goal).sum(-1).mean(1);inner,outer=wall_penalties(positions,env)
    dd=np.linalg.norm(positions[:,:,None]-forecast[None],axis=-1);pairs=np.maximum(2*c.agent_radius+c.agent_collision_margin-dd,0)**2
    penalty=inner.mean(1)+outer.mean(1)+(pairs.mean((1,2)) if forecast.shape[1] else 0.)
    change=np.diff(np.concatenate((np.broadcast_to(previous,(len(actions),1,2)),actions),axis=1),axis=1)
    return distance+weight*penalty+.05*np.square(change).sum(-1).mean(1)

def cross_entropy(env,key,parameters,distributed=False):
    horizon,iterations,samples=parameters;c=env.config;n=len(env.positions)
    if distributed:
        answer=[]
        for i in range(n):
            rng=rng_for(key if n==1 else jax.random.fold_in(key,i));mean=np.broadcast_to(bound((env.goals-env.positions)[i],c.max_speed),(3,2)).copy();std=np.full_like(mean,.8*c.max_speed)
            neighbors=[j for j in range(n) if j!=i];forecast=env.positions[neighbors][None]+c.dt*np.arange(1,horizon+1)[:,None,None]*env.velocities[neighbors][None]
            for _ in range(iterations):
                knots=bound(mean[None]+rng.normal(size=(samples,3,2))*std[None],c.max_speed);actions=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)]
                positions=env.positions[i]+c.dt*np.cumsum(actions,axis=1);cost=scoped_cost(positions,actions,env.goals[i],env,100,env.velocities[i],forecast)
                elite=np.argsort(cost,kind='stable')[:int(np.ceil(.1*samples))];mean=knots[elite].mean(0);std=np.maximum(knots[elite].std(0),.001)
            answer.append(actions[int(np.argmin(cost)),0].copy())
        return np.asarray(answer)
    rng=rng_for(key);mean=np.broadcast_to(bound(env.goals-env.positions,c.max_speed),(3,n,2)).copy();std=np.full_like(mean,.8*c.max_speed)
    for _ in range(iterations):
        knots=bound(mean[None]+rng.normal(size=(samples,3,n,2))*std[None],c.max_speed);actions=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)]
        positions=env.positions[None,None]+c.dt*np.cumsum(actions,axis=1);cost=shooting_cost(positions,actions,env.goals,env,100,env.velocities)
        elite=np.argsort(cost,kind='stable')[:int(np.ceil(.1*samples))];mean=knots[elite].mean(0);std=np.maximum(knots[elite].std(0),.001)
    return actions[int(np.argmin(cost)),0].copy()

def shooting(env,key,parameters,distributed=False):
    horizon,samples,weight=parameters;c=env.config;n=len(env.positions)
    if distributed:
        answer=[]
        for i in range(n):
            # Unlike distributed CEM, this source folds i even for N=1.
            rng=rng_for(jax.random.fold_in(key,i));mean=bound((env.goals-env.positions)[i],c.max_speed)
            knots=bound(mean[None,None]+rng.normal(0,.8*c.max_speed,(samples,3,2)),c.max_speed);acts=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)];acts[0]=mean;acts[1]=0
            positions=env.positions[i]+c.dt*np.cumsum(acts,axis=1);neighbors=[j for j in range(n) if j!=i]
            forecast=env.positions[neighbors][None]+c.dt*np.arange(1,horizon+1)[:,None,None]*env.velocities[neighbors][None]
            cost=scoped_cost(positions,acts,env.goals[i],env,weight,env.velocities[i],forecast);answer.append(acts[int(cost.argmin()),0])
        return np.asarray(answer)
    rng=rng_for(key);mean=bound(env.goals-env.positions,c.max_speed);knots=bound(mean[None,None]+rng.normal(0,.8*c.max_speed,(samples,3,n,2)),c.max_speed)
    acts=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)];acts[0]=mean;acts[1]=0
    positions=env.positions[None,None]+c.dt*np.cumsum(acts,axis=1);cost=shooting_cost(positions,acts,env.goals,env,weight,env.velocities)
    return acts[int(np.argmin(cost)),0].copy()

@jax.custom_jvp
def norm2(x):return jnp.sqrt(jnp.sum(x*x,axis=-1))
@norm2.defjvp
def norm2_jvp(primals,tangents):
    x,=primals;dx,=tangents;r=jnp.sqrt(jnp.sum(x*x,axis=-1));return r,jnp.where(r>0,jnp.sum(x*dx,axis=-1)/jnp.where(r>0,r,1.),0.)

@lru_cache(None)
def optimizer(horizon,iterations,rate,scene,dt,radius,margin,speed,wall_key=()):
    from types import SimpleNamespace
    env=SimpleNamespace(config=SimpleNamespace(scene=scene,dt=dt,agent_radius=radius,agent_collision_margin=0.,wall_collision_margin=margin,wall_radius=0.),walls=np.asarray(wall_key,float).reshape(-1,2,2))
    x=np.linspace(0,2,horizon);w=np.zeros((horizon,3))
    for i,value in enumerate(x):lo=min(int(value),1);w[i,lo]=1-(value-lo);w[i,lo+1]=value-lo
    w=jnp.asarray(w,jnp.float32);opt=optax.adam(rate)
    @jax.jit
    def run(knots,initial,goals,previous):
        def actions(k):return jnp.einsum('hk,kad->had',w,k)
        def loss(k):
            a=actions(k);p=initial[None]+dt*jnp.cumsum(a,axis=0);inner,outer=wall_penalties(p,env,jnp)
            ii,jj=jnp.triu_indices(len(initial),1);dd=norm2(p[:,ii]-p[:,jj]);pairs=jnp.maximum(2*radius-dd,0)**2
            penalty=inner.mean()+outer.mean()+(pairs.mean() if len(ii) else 0.)
            changes=jnp.diff(jnp.concatenate((previous[None],a),axis=0),axis=0)
            return jnp.square(p-goals).sum(-1).mean()+100*penalty+.05*jnp.square(changes).sum(-1).mean()
        def step(carry,_):
            k,st=carry;grad=jax.grad(loss)(k);delta,st=opt.update(grad,st,k);k=optax.apply_updates(k,delta);k=k*jnp.minimum(1.,speed/jnp.maximum(norm2(k)[...,None],1e-30));return (k,st),None
        (result,_),_=jax.lax.scan(step,(knots,opt.init(knots)),None,length=iterations)
        return result,actions(result),loss(result)
    return run

class Nominal:
    def __init__(self,row):self.row=row;self.kind=row['type_code'];self.parameters=json.loads(row['parameters']);self.mean=None;self.warm=None
    def action(self,env,key):
        c=env.config;kind=self.kind
        if kind=='goal_reference':return bound(env.goals-env.positions,c.max_speed)
        if kind=='cross_entropy':return cross_entropy(env,key,self.parameters)
        if kind=='distributed_cross_entropy':return cross_entropy(env,key,self.parameters,True)
        if kind=='centralized_sampling':return shooting(env,key,self.parameters)
        if kind=='distributed_shooting':return shooting(env,key,self.parameters,True)
        if kind=='centralized_mppi':
            h,samples,temperature=self.parameters
            if self.mean is None:self.mean=np.broadcast_to(bound(env.goals-env.positions,c.max_speed),(h,len(env.positions),2)).copy()
            rng=rng_for(key);a=bound(self.mean[None]+rng.normal(size=(samples,*self.mean.shape))*(.8*c.max_speed),c.max_speed)
            p=env.positions[None,None]+c.dt*np.cumsum(a,axis=1);cost=shooting_cost(p,a,env.goals,env,100,env.velocities)
            weight=np.exp(-(cost-cost.min())/temperature);weight/=weight.sum();updated=np.sum(weight[:,None,None,None]*a,axis=0)
            self.mean=np.concatenate((updated[1:],updated[-1:]),axis=0);return updated[0].copy()
        if kind=='gradient_mpc':
            h,k,rate=self.parameters
            if self.warm is None:self.warm=np.broadcast_to(bound(env.goals-env.positions,c.max_speed),(3,len(env.positions),2)).astype(np.float32).copy()
            knots,a,cost=optimizer(int(h),int(k),float(rate),c.scene,c.dt,c.agent_radius,c.wall_collision_margin+c.wall_radius,c.max_speed,tuple(tuple(v) for v in np.asarray(env.walls).reshape(-1,4)))(jnp.asarray(self.warm),jnp.asarray(env.positions,jnp.float32),jnp.asarray(env.goals,jnp.float32),jnp.asarray(env.velocities,jnp.float32))
            a=np.asarray(a);assert np.isfinite(a).all() and np.isfinite(cost)
            shifted=np.concatenate((a[1:],a[-1:]),axis=0);times=np.linspace(0,h-1,3)
            self.warm=np.asarray([[np.interp(times,np.arange(h),shifted[:,i,d]) for d in range(2)] for i in range(len(env.positions))],np.float32).transpose(2,0,1)
            return bound(a[0].astype(np.float64).copy(),c.max_speed)
        if kind in ('prioritized_astar','conflict_based_search'):
            if not hasattr(self,'planner'):
                from planning import Planner
                self.planner=Planner(self.row)
            return self.planner.action(env)
        raise ValueError(kind)
