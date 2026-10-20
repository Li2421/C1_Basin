"""Parameter-shared independent PPO: actor and critic each see one robot row."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class Actor(nn.Module):
    @nn.compact
    def __call__(self,obs):
        x=obs
        for _ in range(3):
            x=nn.tanh(nn.Dense(256)(x))
        mean=nn.Dense(2)(x)
        logs=self.param('log_std',nn.initializers.zeros,(2,))
        return mean,jnp.broadcast_to(logs,mean.shape)


class Critic(nn.Module):
    @nn.compact
    def __call__(self,obs):
        x=obs
        for _ in range(3):
            x=nn.tanh(nn.Dense(256)(x))
        return nn.Dense(1)(x)[...,0]


def logprob(action,mean,logs):
    return jnp.sum(-.5*jnp.square((action-mean)*jnp.exp(-logs))-logs-.5*jnp.log(2*jnp.pi),axis=-1)


def base_normalize(obs,normal):
    return (obs-jnp.asarray(normal['obs_mean']).reshape(4,23))/jnp.asarray(normal['obs_scale']).reshape(4,23)


def online_normalize(obs,stats):
    return (obs-stats['mean'])/jnp.sqrt(stats['variance']+1e-8)


def update_stats(stats,observations,mask):
    x=observations.reshape(-1,23);w=mask.reshape(-1).astype(x.dtype)
    n=w.sum();m=jnp.sum(x*w[:,None],axis=0)/jnp.maximum(n,1)
    v=jnp.sum(jnp.square(x-m)*w[:,None],axis=0)/jnp.maximum(n,1)
    old=stats['count'];total=old+n;delta=m-stats['mean']
    mean=stats['mean']+delta*n/total
    variance=(stats['variance']*old+v*n+jnp.square(delta)*old*n/total)/total
    return dict(count=total,mean=mean,variance=variance)


def decode_action(z,normal):
    raw=jnp.clip(jnp.asarray(normal['act_mean']).reshape(4,2)+jnp.asarray(normal['act_scale']).reshape(4,2)*z,-1.,1.)
    return raw*jnp.minimum(1.,normal['max_speed']/jnp.maximum(jnp.linalg.norm(raw,axis=-1,keepdims=True),1e-30))


def generalized_advantage(rewards,values,dones,bootstrap):
    def step(carry,row):
        advantage,next_value=carry
        reward,value,done=row
        alive=1.-done[...,None]
        delta=reward+.99*next_value*alive-value
        advantage=delta+.99*.95*alive*advantage
        return (advantage,value),advantage
    _,adv=jax.lax.scan(step,(jnp.zeros_like(bootstrap),bootstrap),(rewards,values,dones),reverse=True)
    return adv,adv+values
