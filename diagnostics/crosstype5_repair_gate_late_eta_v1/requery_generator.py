"""Read-only current-state conditioning and exactly frozen generator mean."""
from common import *
from legacy_conditioning import ReadOnlyConditioning
from frozen_networks import Generator,merge_initialized,eta_mean,scalar_environment_fields
from flax import serialization
import jax,jax.numpy as jnp
CHECKPOINT=D/'orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack'
NORMALIZATION=D/'orthoflow3_generator_critic_v1/normalization.json'
class FrozenMean:
    def __init__(self):
        self.norm=read(NORMALIZATION);self.n=self.norm['scenarios']['ring_exchange'];self.cdim=len(self.norm['environment_keys'])+3;self.net=Generator();self.parameters=serialization.from_bytes(merge_initialized(self.net,{k:len(v['h_mean']) for k,v in self.norm['scenarios'].items()},self.cdim),CHECKPOINT.read_bytes());self.reference=ReadOnlyConditioning();jax.config.update('jax_enable_x64',False)
    def query(self,state_uid,physical):
        h,descriptor=self.reference.conditioning(dict(uid=state_uid,physical=physical));c=np.zeros(self.cdim,np.float32);values=scalar_environment_fields(descriptor)
        for j,k in enumerate(self.norm['environment_keys']):c[j]=values.get(k,0.)
        c[-1]=1.;hh=(np.asarray(h,np.float32)-np.asarray(self.n['h_mean'],np.float32))/np.asarray(self.n['h_std'],np.float32);cc=(c-np.asarray(self.n['c_mean'],np.float32))/np.asarray(self.n['c_std'],np.float32);raw=self.net.apply(self.parameters,jnp.asarray(hh[None]),jnp.asarray(cc[None]),method=self.net.ring);eta=np.asarray(eta_mean(raw))[0].astype(np.float64)
        return dict(eta_b=eta,h=np.asarray(h,np.float32),c=c,conditioning_uid=state_uid,checkpoint_sha256=sha(CHECKPOINT),normalization_sha256=sha(NORMALIZATION))
