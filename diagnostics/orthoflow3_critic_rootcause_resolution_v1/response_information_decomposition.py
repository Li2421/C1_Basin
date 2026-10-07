"""Frozen source-VAL diagnostic: what state information enters through C?

Replace response groups by their FIT controller-by-eta means. Unlike a wrong
controller swap this retains controller/eta identity and removes within-class
state variation. This is an intervention on model inputs, not a retraining
ablation or an independently certified physical rollout.
"""
import os,json
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from pathlib import Path
from scipy.special import expit
from . import support_train as tr
from .state_support import read,write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite

def main():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    frozen=read(tr.OUT/'models_frozen.json');out=[]
    m=tr.model('entity_response_mean');predict=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
    for entry in frozen['models']:
        if entry['kind']!='entity_response_mean':continue
        size=entry['size'];base_size=size.replace('_pair_equal','');d,x,fit,val,norm=tr.load(base_size)
        saved=read(Path(entry['path'])/'normalization.json');assert norm==saved
        params=serialization.msgpack_restore((Path(entry['path'])/'best.msgpack').read_bytes());ctx=d['input_context'];reference=np.empty_like(ctx)
        for c in range(2):
          for e in (10,15):
            f=fit[d['eta_index'][fit]==e];reference[c,d['eta_index']==e]=ctx[c,f].mean(0)
        both=np.ones(88,bool);global_only=np.arange(88)<24;response_only=~global_only
        tangent=np.zeros(88,bool)
        for agent in range(4):tangent[24+agent*16+np.array([1,3,5,9,11,13])]=True
        conditions={'correct':np.zeros(88,bool),'global24_at_controller_eta_mean':global_only,
            'signed_agent_response_at_controller_eta_mean':response_only,'all_behavior_at_controller_eta_mean':both,
            'tangential_response_at_controller_eta_mean':tangent}
        base=None;basechoice=None
        for condition,mask in conditions.items():
            cx=ctx[:,val].copy();cx[:,:,mask]=reference[:,val][:,:,mask]
            z=np.array([np.asarray(predict(params,gather(x,d['state_index'][val]),jnp.asarray(d['eta'][val],jnp.float32),jnp.asarray(cx[c],jnp.float32))) for c in range(2)])
            met,_=tr.metrics(z,d,val)
            if base is None:base=z;basechoice=z.reshape(2,32,2).argmax(-1)
            out.append(dict(size=size,seed=entry['seed'],condition=condition,
                changed_top1=int((z.reshape(2,32,2).argmax(-1)!=basechoice).sum()),
                mean_probability_change=float(abs(expit(z)-expit(base)).mean()),**met))
    csvwrite(tr.OUT/'response_information_decomposition.csv',out)
    write(tr.OUT/'response_information_decomposition_protocol.json',dict(
        models_frozen=True,only_source_VAL_used=True,confirmation_not_read=True,
        replacement='FIT controller-by-exact-eta feature means, preserving controller/eta class while removing state-specific response',
        interpretation='Input-use diagnosis; an off-manifold replacement can affect outputs and cannot by itself prove deployment benefit or information sufficiency.',
        tangent_channels='Signed perpendicular-to-initial-goal Flow/safe/executed action means, nominal and eta-conditioned arms'))
    print(json.dumps([r for r in out if r['size']=='expanded206_pair_equal'],indent=2))

if __name__=='__main__':main()
