"""JAX Ring plant for training only; no safety projection or evaluation override."""
from typing import NamedTuple
import jax
import jax.numpy as jnp

PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


class State(NamedTuple):
    positions: object
    velocities: object
    goals: object
    step_count: object
    arrived: object


def segment_distance(first,second):
    direction=second-first
    fraction=jnp.clip(-jnp.sum(first*direction,axis=-1)/jnp.maximum(jnp.sum(direction*direction,axis=-1),1e-30),0.,1.)
    return jnp.linalg.norm(first+fraction[...,None]*direction,axis=-1)


class TrainingEnv:
    def __init__(self,config):
        self.config=dict(config)

    def reset(self,positions,velocities,goals):
        return State(positions,velocities,goals,jnp.asarray(0,jnp.int32),
                     jnp.linalg.norm(goals-positions,axis=-1)<=self.config['goal_tolerance'])

    def step(self,state,action,active=None):
        c=self.config;p=state.positions;g=state.goals
        active=jnp.ones(4,bool) if active is None else active
        before_error=jnp.linalg.norm(g-p,axis=-1)
        # The same explicit goal-stop is applied in equivalence reference tests.
        u=jnp.where((active & (before_error>c['goal_tolerance']))[:,None],action,0.)
        q=p+c['dt']*u
        obstacle=segment_distance(p,q)-c['obstacle_radius']-c['agent_radius']
        outer=jnp.minimum(c['outer_radius']-c['agent_radius']-jnp.linalg.norm(p,axis=-1),
                          c['outer_radius']-c['agent_radius']-jnp.linalg.norm(q,axis=-1))
        ii=jnp.asarray([a for a,b in PAIRS]);jj=jnp.asarray([b for a,b in PAIRS])
        pair=segment_distance(p[ii]-p[jj],q[ii]-q[jj])-2*c['agent_radius']
        wall_hit=((obstacle<=c['collision_margin']) | (outer<=c['collision_margin'])) & active
        pair_hit=(pair<=c['agent_collision_margin']) & active[ii] & active[jj]
        agent_hit=jnp.zeros(4,bool).at[ii].max(pair_hit).at[jj].max(pair_hit)
        collided=wall_hit | agent_hit
        collision=collided.any()
        error=jnp.linalg.norm(g-q,axis=-1)
        arrived=(error<=c['goal_tolerance']) & active
        success=jnp.all(arrived | ~active)
        steps=state.step_count+1
        terminal=jnp.where(collision & c['terminate_on_collision'],1,
                 jnp.where(success & c['terminate_on_success'],2,jnp.where(steps>=c['max_steps'],3,0)))
        reward=2.5*(before_error-error)+15*(arrived & ~state.arrived)+10*success-.01
        reward=reward-jnp.where(collision,jnp.where(collided,15.,5.),0.)
        reward=jnp.where(active,reward,0.)
        next_state=State(q,u,g,steps,state.arrived|arrived)
        info=dict(termination=terminal,swept_obstacle_distance=obstacle,swept_outer_distance=outer,
                  swept_agent_distances=pair,collided=collided,success=success)
        return next_state,reward,terminal!=0,info


def observation(state,config,frame):
    p,v,g=state.positions,state.velocities,state.goals
    radius=jnp.linalg.norm(p,axis=-1)
    radial=p/jnp.maximum(radius[:,None],1e-12)
    tangent=jnp.stack((-radial[:,1],radial[:,0]),axis=-1)
    rows=[]
    for i in range(4):
        def chart(x):
            return jnp.stack((jnp.sum(x*radial[i],axis=-1),jnp.sum(x*tangent[i],axis=-1)),axis=-1) if frame=='local' else x
        own=jnp.stack((radius[i],jnp.zeros_like(radius[i]))) if frame=='local' else p[i]
        relative=[]
        for j in range(4):
            if i!=j:
                relative.extend((chart(p[j]-p[i]),chart(v[j]-v[i])))
        centre=jnp.stack((-radius[i],jnp.zeros_like(radius[i]))) if frame=='local' else -p[i]
        geom=jnp.concatenate((centre,jnp.asarray([config['obstacle_radius'],config['outer_radius']],dtype=p.dtype),
                              (radius[i]-config['obstacle_radius']-config['agent_radius'])[None]))
        rows.append(jnp.concatenate((own,chart(v[i]),chart(g[i]-p[i]),*relative,geom)))
    # The original observation chart explicitly stores float32 even in x64 tests.
    return jnp.stack(rows).astype(jnp.float32)


def to_world(action,positions,frame):
    if frame=='global':
        return action
    radial=positions/jnp.maximum(jnp.linalg.norm(positions,axis=-1,keepdims=True),1e-12)
    tangent=jnp.stack((-radial[:,1],radial[:,0]),axis=-1)
    return action[:,:1]*radial+action[:,1:]*tangent
