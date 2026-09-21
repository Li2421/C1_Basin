"""Fixed-horizon model continuation, with immutable actual first-event metrics.

Only forecast semantics change. The physical evaluator still stops at its
original first event; post-event model actions are never executed there.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.c1.train import observation
from single_integrator.c1.termination import event_step
from single_integrator.c1.risk.joint_frozen import controller_projection
from single_integrator.c1.risk.event_history import trajectory
from single_integrator.environment import GiveWayEnv


def rollout(params,field,initial,noise,plant,cbf,action_offsets=None):
    if noise.shape!=(plant.max_steps,4):raise ValueError('full-horizon noise required')
    if plant.dt!=.05 or plant.max_speed!=.5 or not all((plant.terminate_on_success,plant.terminate_on_deadlock,plant.terminate_on_collision)):
        raise ValueError('original first-event experiment required')
    if action_offsets is None:action_offsets=jnp.zeros_like(noise)
    if action_offsets.shape!=noise.shape:raise ValueError('offset shape mismatch')
    initial=jnp.asarray(initial,jnp.float64);env=GiveWayEnv(plant)
    jax.debug.callback(lambda x:GiveWayEnv(plant).reset(x),jax.lax.stop_gradient(initial))
    goals,walls=jnp.asarray(env.goals),jnp.asarray(env.walls)
    window=plant.progress_window_seconds/plant.dt;size=int(math.ceil(window))+1
    positions=initial[None]
    history=jnp.zeros((1,size,2)).at[:,0].set(jnp.linalg.norm(positions-goals,axis=-1))
    carry=(positions,jnp.zeros_like(positions),jnp.ones(1,bool),jnp.full((1,),-1,jnp.int32),history)
    def step(carry,inputs):
        positions,velocity,actual_alive,since,history=carry;t,xi=inputs
        obs=observation(positions,velocity,goals)
        A,b,_=jax.vmap(lambda x:barrier_constraints(x,walls,plant.to_dict(),cbf))(positions)
        prepared=field.prepare(params,obs,xi[None],A,b,plant.max_speed,
            lambda v,a,b,s:controller_projection(v,a,b))
        candidate=prepared['safe']+prepared['correction']+action_offsets[t][None]
        applied=controller_projection(candidate,A,b)
        after=positions+plant.dt*applied.reshape(1,2,2)
        history=history.at[:,(t+1)%size].set(jnp.linalg.norm(after-goals,axis=-1))
        start=jnp.maximum(0.,t+1-window);lo=jnp.floor(start).astype(int);hi=jnp.ceil(start).astype(int)
        raw_codes,since=event_step(positions,applied.reshape(1,2,2),history[:,lo%size],history[:,hi%size],
            since,jnp.ones(1,bool),t,plant)
        def check(c):
            if np.any((c==2)|(c==3)):raise RuntimeError('unsafe model continuation: reject forecast')
        jax.debug.callback(check,raw_codes)
        codes=jnp.where(actual_alive,raw_codes,0)
        next_alive=actual_alive & (codes==0)
        output=(positions[0],after[0],candidate[0],applied[0],prepared['safe'][0],
                A[0],b[0],actual_alive[0],codes[0])
        return (after,applied.reshape(1,2,2),next_alive,since,history),output
    _,data=jax.lax.scan(jax.checkpoint(step),carry,(jnp.arange(plant.max_steps),noise))
    before,after,candidate,applied,safe,A,b,alive,codes=data
    risk=trajectory(before,after,applied,goals,dt=plant.dt,max_speed=plant.max_speed,
        goal_tolerance=plant.goal_tolerance,hold_seconds=plant.deadlock_hold_seconds,
        progress_window_seconds=plant.progress_window_seconds,progress_epsilon=plant.progress_epsilon,
        speed_epsilon_fraction=plant.speed_epsilon_fraction)
    deviation=jnp.sum(jnp.where(alive,jnp.sum((applied-safe)**2,axis=-1),0.))/jnp.maximum(jnp.sum(alive),1)
    return dict(J_live=risk['J_live'],J_def=deviation,deadlock=jnp.any(codes==4),
        success=jnp.any(codes==1),timeout=jnp.any(codes==5),steps=jnp.sum(alive),
        robustness=risk['robustness']),dict(before=before,after=after,candidate=candidate,
            applied=applied,safe=safe,A=A,b=b,alive=alive,codes=codes)
