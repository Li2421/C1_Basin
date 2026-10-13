"""Freeze every confirmation prediction and falsification before rollout labels."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from .experiment import HERE,read,freeze,sha
from .train import Critic,inputs,KINDS,SEEDS


class Arrays(dict):
    @property
    def files(self):return list(self)


def run():
    p=read(HERE/'protocol.json');frozen=read(HERE/'models_frozen.json');allocation=read(HERE/'confirmation_allocation.json')
    keys=[];ent=[];nominal=[];field=[]
    states=[s for s in p['states'] if s['study_split']=='final' and s['study_index']<allocation['states_per_scene']]
    for s in sorted(states,key=lambda v:v['state_uid']):
        for ctl in ['analytic6','analytic7']:
            data=np.load(HERE/'features'/s['state_uid']/(ctl+'.npz'))
            ent.append({k[7:]:data[k] for k in data.files if k.startswith('entity_')})
            nominal.append(np.pad(data['nominal'],((0,4-len(data['nominal'])),(0,0))))
            field.append(np.pad(data['field'],((0,4-len(data['field'])),(0,0))))
            keys.append(dict(state_uid=s['state_uid'],scene=s['scenario'],controller=ctl))
    d=Arrays({f'entity_{k}':v for k,v in rep.batch(ent).items()})
    d.update(nominal=np.asarray(nominal),field=np.asarray(field),eta=np.asarray(p['eta'],np.float32))
    for k in keys[0]:d[k]=np.asarray([v[k] for v in keys])
    swap=np.arange(len(keys)).reshape(-1,2)[:,::-1].ravel()
    assert np.all(d['state_uid'][swap]==d['state_uid']) and np.all(d['controller'][swap]!=d['controller'])
    result=dict(state_uid=d['state_uid'],scene=d['scene'],controller=d['controller'],controller_swap=swap)
    for kind in KINDS:
        predictions=[];mismatch=[]
        for seed in SEEDS:
            row=next(r for r in frozen['records'] if r['kind']==kind and r['seed']==seed)
            from pathlib import Path
            folder=Path(row['path']);assert row['checkpoint_sha256']==sha(folder/'checkpoint.msgpack')
            x,c,e=inputs(d,read(folder/'normalization.json'));net=Critic(kind)
            initial=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
            params=serialization.from_bytes(initial,(folder/'checkpoint.msgpack').read_bytes())
            @jax.jit
            def infer(ii,jj,cc):return net.apply(params,{k:v[ii] for k,v in x.items()},c[cc],e[jj])
            def get(indices):
                ii=np.repeat(np.arange(len(keys)),16);jj=np.tile(np.arange(16),len(keys));ci=np.repeat(indices,16);zs=[]
                for start in range(0,len(ii),256):
                    zs.append(np.asarray(infer(jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]),jnp.asarray(ci[start:start+256]))))
                return np.asarray(jax.nn.sigmoid(jnp.asarray(np.concatenate(zs).reshape(len(keys),16))))
            predictions.append(get(np.arange(len(keys))));mismatch.append(get(swap))
        result[kind]=np.asarray(predictions);result[kind+'_controller_mismatch']=np.asarray(mismatch)
        if kind in ('eta_only','state_only'):np.testing.assert_array_equal(result[kind],result[kind+'_controller_mismatch'])
    rng=np.random.default_rng(2026100919);perms=[]
    for _ in range(100):
        mapping=np.arange(len(keys))
        for scene in sorted(set(d['scene'])):
            first=np.flatnonzero((d['scene']==scene)&(d['controller']=='analytic6'))
            while True:
                perm=rng.permutation(len(first))
                if np.all(perm!=np.arange(len(first))):break
            mapping[first]=first[perm];mapping[first+1]=first[perm]+1
        assert np.all(d['controller'][mapping]==d['controller']) and np.all(d['scene'][mapping]==d['scene'])
        assert np.all(d['state_uid'][mapping]!=d['state_uid'])
        perms.append(mapping)
    result['state_permutations']=np.asarray(perms)
    dest=HERE/'confirmation_predictions.npz';assert not dest.exists()
    np.savez_compressed(dest,**result)
    freeze(HERE/'confirmation_predictions_frozen.json',dict(predictions_sha256=sha(dest),
         models_sha256=sha(HERE/'models_frozen.json'),allocation_sha256=sha(HERE/'confirmation_allocation.json'),
         no_confirmation_labels_opened=True,rows=len(keys),physical_states=len(states),
         controller_mismatch='Other held controller evaluated at same physical state',
         state_mismatch='100 fixed derangements within scene, same controller, swap physical features AND state-dependent function queries',
         tie_break='lowest candidate index'))
    print('All confirmation predictions and permutations frozen',flush=True)


if __name__=='__main__':run()
