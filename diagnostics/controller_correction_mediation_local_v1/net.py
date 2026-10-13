import jax.numpy as jnp
import flax.linen as nn
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep

class Critic(nn.Module):
    kind:str
    outputs:int=1
    @nn.compact
    def __call__(self,x,c,e):
        x=dict(x)
        raw_eta=e*jnp.asarray([0.375, 0.5, 0.375])+jnp.asarray([0.875, 0.0, 0.375])
        delta=jnp.einsum('bajk,bk->baj',x['correction_basis'],raw_eta)
        x['agents']=x['agents'].at[...,10:12].set(delta/2.963346634825058)
        if self.kind in ('state_contact_bilinear','state_field_compact_bilinear'):
            physical=rep.Encoder(name='physical_encoder')(x)
            eta=nn.silu(nn.Dense(32,name='eta')(e))
            h=nn.silu(nn.Dense(128,name='shared1')(jnp.concatenate((physical,eta),-1)))
            h=nn.silu(nn.Dense(64,name='shared2')(h))
            controller=jnp.clip(rep.pool(c,x['agent_mask'],1),-3.,3.)
            phi=jnp.concatenate((controller,controller**2),-1)
            coefficients=nn.Dense(phi.shape[-1]+1,name='controller_coefficients')(h)
            return coefficients[...,0]+jnp.sum(coefficients[...,1:]*phi,-1)/jnp.sqrt(float(phi.shape[-1]))
        parts=[]
        if self.kind!='eta_only':
            if self.kind in ('state_only','state_context','state_field','state_field_compact','state_contact'):
                xx=dict(x)
                cc=c if self.kind in ('state_field','state_field_compact','state_contact') else c[...,:14] if self.kind=='state_context' else jnp.zeros_like(c[...,:14])
                xx['agents']=jnp.concatenate((xx['agents'],cc),-1)
                parts.append(rep.Encoder(name='physical_encoder')(xx))
            else:
                cc=c if self.kind=='field_only' else c[...,:14]
                h=nn.silu(nn.Dense(64,name='context1')(cc))
                h=nn.silu(nn.Dense(64,name='context2')(h))
                parts.append(rep.pool(h,x['agent_mask'],1))
        parts.append(nn.silu(nn.Dense(32,name='eta')(e)))
        h=nn.silu(nn.Dense(128,name='shared1')(jnp.concatenate(parts,-1)))
        h=nn.silu(nn.Dense(64,name='shared2')(h))
        z=nn.Dense(self.outputs,name='out')(h)
        return z[...,0] if self.outputs==1 else z

