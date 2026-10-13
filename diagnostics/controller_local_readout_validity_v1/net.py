import jax.numpy as jnp
import flax.linen as nn
from diagnostics.orthoflow3_unified_representation_v1.representation import pool

class Encoder(nn.Module):
    @nn.compact
    def __call__(self,x):
        am=x['agent_mask'];n=am.shape[-1]
        pm=am[:,:,None]*am[:,None,:]*(1-jnp.eye(n)[None])
        pair=nn.silu(nn.Dense(32,name='pair1')(x['pairs']));pair=nn.silu(nn.Dense(32,name='pair2')(pair))
        message=pool(pair,pm,2)
        om=am[:,:,None]*x['obstacle_mask'][:,None,:]
        ob=nn.silu(nn.Dense(32,name='obstacle1')(x['obstacles']));ob=nn.silu(nn.Dense(32,name='obstacle2')(ob))
        geometry=pool(ob,om,2)
        own=nn.silu(nn.Dense(32,name='agent1')(x['agents']))
        agent=nn.silu(nn.Dense(64,name='agent2')(jnp.concatenate((own,message,geometry),axis=-1)))
        scene=pool(agent,am,1)
        count=jnp.stack((jnp.log1p(am.sum(-1)),jnp.log1p(x['obstacle_mask'].sum(-1))),axis=-1)
        scene=nn.silu(nn.Dense(128,name='scene')(jnp.concatenate((scene,x['globals'],count),axis=-1)))
        return scene,agent

class Critic(nn.Module):
    kind:str
    outputs:int=8
    @nn.compact
    def __call__(self,x,c,e):
        assert self.kind=='state_field_compact' and self.outputs==8
        xx=dict(x);xx['agents']=jnp.concatenate((xx['agents'],c),-1)
        scene,agent=Encoder(name='physical_encoder')(xx)
        eta=nn.silu(nn.Dense(32,name='eta')(e))
        n=agent.shape[1]
        parts=(jnp.broadcast_to(scene[:,None,:],(len(scene),n,128)),agent,
               jnp.broadcast_to(eta[:,None,:],(len(eta),n,32)))
        # 100 chosen algebraically: this decoder has290fewerparameters than
        # the old128/64/global8head, while retainingallencoderweights/shapes.
        h=nn.silu(nn.Dense(100,name='shared1')(jnp.concatenate(parts,-1)))
        h=nn.silu(nn.Dense(64,name='shared2')(h))
        z=nn.Dense(2,name='out')(h)
        return z.reshape(len(z),-1)
