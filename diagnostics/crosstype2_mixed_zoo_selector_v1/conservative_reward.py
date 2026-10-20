"""Reward-only subclass: the copied, equivalence-tested plant is untouched."""
from training_env import TrainingEnv
import jax.numpy as jnp


def reward_delta(info,active):
    collided=info['collided'];collision=jnp.any(collided)
    extra_collision=jnp.where(collision,jnp.where(collided,-45.,-15.),0.)
    pair=info['swept_agent_distances']
    ii=jnp.asarray([0,0,0,1,1,2]);jj=jnp.asarray([1,2,3,2,3,3])
    pair=jnp.where(active[ii]&active[jj],pair,jnp.inf)
    nearest=jnp.full((4,),jnp.inf).at[ii].min(pair).at[jj].min(pair)
    gap=jnp.minimum(nearest,jnp.minimum(info['swept_obstacle_distance'],info['swept_outer_distance']))
    proximity=jnp.where(gap<.2,-.05*(.2-gap)/.2,0.)
    return jnp.where(active,extra_collision+proximity,0.)


class ConservativeEnv(TrainingEnv):
    def step(self,state,action,active=None):
        active=jnp.ones(4,bool) if active is None else active
        next_state,reward,done,info=super().step(state,action,active)
        return next_state,reward+reward_delta(info,active),done,info
