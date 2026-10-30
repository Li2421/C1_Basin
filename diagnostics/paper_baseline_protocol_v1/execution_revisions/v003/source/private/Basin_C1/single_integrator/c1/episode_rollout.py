"""Memory-bounded, full-horizon C1 BPTT using scan and rematerialization."""
import math
import jax
import jax.numpy as jnp

from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.c1.risk.soft_activity import soft_risk_diagnostics
from single_integrator.c1.risk.risk_v2 import trajectory_risk_v2
from single_integrator.c1.termination import event_step
from single_integrator.environment import GiveWayEnv


def rollout_episode_terms(params,field,projection,initial,noise,plant,cbf,risk,observation,
                          return_details=False,return_mask=False):
    if noise.shape[1]<plant.max_steps:
        raise ValueError('v2 rollout must cover the full environment time limit')
    if not all((plant.terminate_on_success,plant.terminate_on_collision,plant.terminate_on_deadlock)):
        raise ValueError('v2 requires the original first-event termination flags')
    if hasattr(projection,'for_scan'):
        projection=projection.for_scan()
    initial=jnp.asarray(initial,jnp.float64)
    def validate_initial(batch):
        check=GiveWayEnv(plant)
        for positions in batch:check.reset(positions)
    jax.debug.callback(validate_initial,jax.lax.stop_gradient(initial))
    env=GiveWayEnv(plant);goals,walls=jnp.asarray(env.goals),jnp.asarray(env.walls)
    B=len(initial);buffer_length=int(math.ceil(plant.progress_window_seconds/plant.dt))+1
    history=jnp.zeros((B,buffer_length,2),jnp.float64).at[:,0].set(jnp.linalg.norm(initial-goals,axis=-1))
    carry=(initial,jnp.zeros_like(initial),jnp.ones(B,bool),jnp.zeros(B,jnp.int32),
           jnp.full((B,),-1,jnp.int32),history)

    def step_fn(carry,inputs):
        positions,velocity,alive,codes,since,history=carry
        step,xi=inputs
        control_positions=jnp.where(alive[:,None,None],positions,initial)
        obs=observation(control_positions,velocity,goals)
        A,b,h=jax.vmap(lambda p:barrier_constraints(p,walls,plant.to_dict(),cbf))(control_positions)
        # One shared compiled branch, with no solver/model calls after every
        # lane has terminated. Masks still handle individually finished lanes.
        zeros={k:jnp.zeros((B,4),jnp.float64) for k in ('nominal','safe','correction','candidate','applied')}
        control=jax.lax.cond(jnp.any(alive),
            lambda _: {k:jnp.asarray(v,jnp.float64) for k,v in field.control(params,obs,xi,A,b,plant.max_speed,projection).items()},
            lambda _:zeros,operand=None)
        control={k:jnp.where(alive[:,None],v,0.) for k,v in control.items()}
        u=control['applied'].reshape(B,2,2)
        after=positions+plant.dt*u
        errors=jnp.linalg.norm(after-goals,axis=-1)
        history=history.at[:,(step+1)%buffer_length].set(errors)
        start=jnp.maximum(0.,step+1-plant.progress_window_seconds/plant.dt)
        lo=jnp.floor(start).astype(jnp.int32)%buffer_length
        hi=jnp.ceil(start).astype(jnp.int32)%buffer_length
        event,since=event_step(positions,u,history[:,lo],history[:,hi],since,alive,step,plant)
        new_codes=jnp.where(alive,event,codes)
        next_alive=alive & (event==0)
        slack=jnp.einsum('bij,bj->bi',A,control['applied'])-b
        diagnostic=jax.vmap(lambda f,g,h,s:soft_risk_diagnostics(f,g,h,s,risk))(
            control['candidate'].reshape(B,2,2),A.reshape(B,17,2,2),h,slack)
        output=dict(**control,**diagnostic,positions=positions,positions_after=after,
                    episode_mask=alive,A=A,b=b,h=h)
        carry=(after,jnp.where(next_alive[:,None,None],u,0.),next_alive,new_codes,since,history)
        return carry,output

    # Recompute step internals during reverse mode, rather than keeping every
    # Flow/MLP/SOCP intermediate from all 850 steps resident in memory.
    final,steps=jax.lax.scan(jax.checkpoint(step_fn),carry,
                             (jnp.arange(noise.shape[1],dtype=jnp.int32),jnp.swapaxes(noise,0,1)))
    details=jax.tree_util.tree_map(lambda a:jnp.swapaxes(a,0,1),steps)
    all_positions=jnp.concatenate((initial[:,None],details['positions_after']),axis=1)
    scored=trajectory_risk_v2(details['risk'],all_positions,goals,risk,
                              details['episode_mask'],final[3],plant.dt)
    result=(details['applied'],details['safe'],scored['trajectory_risk'])
    if return_details:return (*result,dict(details,**scored))
    if return_mask:return (*result,details['episode_mask'])
    return result
