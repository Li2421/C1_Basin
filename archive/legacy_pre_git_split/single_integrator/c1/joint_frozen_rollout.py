"""Validated5s decision/25s prediction protocol, with full state BPTT."""
import jax
import jax.numpy as jnp
from single_integrator.environment import GiveWayEnv
from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.c1.train import observation
from single_integrator.c1.termination import event_step
from single_integrator.c1.risk.joint_frozen import controller_projection,geometry,trajectory

def rollout(params,field,initial,noise,plant,cbf,interventions=None):
    if plant.terminate_on_deadlock:raise ValueError('Frozen protocol continues after deadlock')
    env=GiveWayEnv(plant);goals=jnp.asarray(env.goals);walls=jnp.asarray(env.walls)
    initial=jnp.asarray(initial);positions=initial[None];history=jnp.zeros((1,41,2)).at[:,0].set(jnp.linalg.norm(positions-goals,axis=-1))
    carry=(positions,jnp.zeros_like(positions),jnp.ones(1,bool),jnp.full((1,),-1,jnp.int32),history)
    if interventions is None:interventions=jnp.zeros((500,4))
    def step(carry,inputs):
        positions,velocity,alive,since,history=carry;t,xi,delta=inputs
        obs=observation(positions,velocity,goals)
        A,b,_=jax.vmap(lambda x:barrier_constraints(x,walls,plant.to_dict(),cbf))(positions)
        ctrl=field.prepare(params,obs,xi[None],A,b,.5,lambda v,a,b,s:controller_projection(v,a,b))
        v=ctrl['safe']+jnp.where(t>=100,ctrl['correction'],0.)+delta[None]
        u=controller_projection(v,A,b);u=jnp.where(alive[:,None],u,0.)
        after=positions+.05*u.reshape(1,2,2)
        errors=jnp.linalg.norm(after-goals,axis=-1);history=history.at[:,(t+1)%41].set(errors)
        start=jnp.maximum(0,t+1-40);past=history[:,start%41]
        codes,since=event_step(positions,u.reshape(1,2,2),past,past,since,alive,t,plant)
        success=(codes==1)|~alive  # Frozen success is absorbing; collision separately rejected below.
        def check(codes):
            import numpy as np
            if np.any((np.asarray(codes)==2)|(np.asarray(codes)==3)):raise RuntimeError('Collision: trajectory is unscorable')
        jax.debug.callback(check,codes)
        nextalive=alive&(codes==0)
        output=(positions[0],after[0],v[0],u[0],ctrl['safe'][0],A[0],b[0],success[0],alive[0])
        return (after,jnp.where(nextalive[:,None,None],u.reshape(1,2,2),0.),nextalive,since,history),output
    _,data=jax.lax.scan(jax.checkpoint(step),carry,(jnp.arange(500,dtype=jnp.int32),noise,interventions))
    before,after,v,u,safe,A,b,success,alive=data
    vg=jnp.where(alive[100:300,None],v[100:300],jnp.ones((200,4))*.1)
    g=geometry(vg,A[100:300],b[100:300]);terms=trajectory(before,after,v,g,goals,success)
    # Same-state baseline deviation, only the20s trainable prediction interval.
    mask=alive[100:];deviation=jnp.sum(jnp.sum((u[100:]-safe[100:])**2,axis=-1)*mask)/jnp.maximum(mask.sum(),1)
    return terms,deviation,dict(before=before,after=after,candidate=v,applied=u,g=g,success=success)
