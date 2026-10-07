"""Frozen-checkpoint same-controller, cross-state context replacement.

This post-hoc diagnostic does not select or train any checkpoint. It asks whether
the network uses *state-specific* response C beyond controller-level identity.
"""
from __future__ import annotations
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .pipeline import OUT,read,write
from .train_models import model_for,SEEDS
from .summarize import evaluate,_csv

KINDS=('context_only','full')

def main():
    raw=dict(np.load(OUT/'dataset.npz'));x=dict(np.load(OUT/'entities.npz'))
    plan=read(OUT/'train_split.json')
    indices=np.flatnonzero(raw['split']=='validation')
    states=plan['independent_validation_state_indices']
    assert len(states)>=12 and len(indices)==len(states)*2
    # Same eta, same controller, different independent state; deterministic
    # cyclic replacement never supplies the correct state's response.
    successor={s:states[(j+1)%len(states)] for j,s in enumerate(states)}
    pair_lookup={(int(raw['state_index'][i]),int(raw['eta_index'][i])):i for i in indices}
    swapped=np.asarray([pair_lookup[(successor[int(raw['state_index'][i])],int(raw['eta_index'][i]))]
                        for i in indices],int)
    assert not np.any(raw['state_index'][swapped]==raw['state_index'][indices])
    results=[];per_state=[]
    for kind in KINDS:
      for seed in SEEDS:
        base=OUT/'models'/kind/f'seed{seed}'
        metadata=read(base/'normalization.json')
        model=model_for(kind)
        template=model.init(jax.random.PRNGKey(seed),gather(x,raw['state_index'][:1]),
                            jnp.zeros((1,3)),jnp.zeros((1,24)),jnp.zeros((1,3)))
        params=serialization.from_bytes(template,(base/'checkpoint.msgpack').read_bytes())
        eta=(raw['eta']-metadata['eta_center'])/metadata['eta_scale']
        context=(raw['context']-metadata['context_center'])/metadata['context_scale']
        predictor=jax.jit(lambda xx,ee,cc:model.apply(params,xx,ee,cc,jnp.zeros((len(ee),3))))
        logits=np.zeros((2,len(indices)),np.float32)
        for c in range(2):
            for start in range(0,len(indices),128):
                ii=indices[start:start+128];jj=swapped[start:start+128]
                logits[c,start:start+len(ii)]=np.asarray(predictor(gather(x,raw['state_index'][ii]),
                    jnp.asarray(eta[ii]),jnp.asarray(context[c,jj])))
        rr,metrics=evaluate(logits,raw,indices)
        results.append({'kind':kind,'seed':seed,'condition':'same_controller_wrong_state_context',**metrics})
        per_state.extend({'kind':kind,'seed':seed,**r} for r in rr)
    _csv(OUT/'same_controller_context_shuffle_metrics.csv',results)
    _csv(OUT/'same_controller_context_shuffle_per_state.csv',per_state)
    write(OUT/'same_controller_context_shuffle_protocol.json',{
        'posthoc_diagnostic':True,'new_rollout':0,'new_training':0,'checkpoint_modified':False,
        'replacement':'same eta index and same controller, cyclic next independent source family',
        'state_h_kept_correct':True,'no_controller_identity_change':True})
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
