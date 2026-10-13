"""Physical-state-only and eta-only critic fits on identical Q4 labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.physical_state_critic_formal_v1.train_ring_ff import StateCritic,EtaCritic
from diagnostics.physical_state_critic_formal_v1.assemble_trainval import PANELS

HERE=Path(__file__).resolve().parent
SEEDS=(17,23,41)
STEPS=tuple(range(200,12001,200))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def select_key(row,tol):
    bucket=int(np.floor(row['selection_regret']/tol+1e-9))
    pair=row['pairwise_accuracy'] if row['pairwise_accuracy'] is not None else -.5
    return (bucket,-pair,row['validation_nll'],row['step'])


def metrics(logits,success,failure,numerical,scenes):
    assert logits.shape==success.shape==failure.shape==numerical.shape
    assert logits.shape[1]==16
    lower=success/16
    upper=(success+numerical)/16
    selected=np.argmax(logits,axis=1)
    scene_rows={}
    for scene in sorted(set(scenes)):
        ix=np.flatnonzero(scenes==scene)
        regret=np.max(lower[ix],axis=1)-lower[ix,selected[ix]]
        a,b=np.triu_indices(16,1)
        direction=np.where(lower[ix][:,a]-upper[ix][:,b]>=.25,1,
                           np.where(lower[ix][:,b]-upper[ix][:,a]>=.25,-1,0))
        pred=np.sign(logits[ix][:,a]-logits[ix][:,b])
        valid=direction!=0
        pair=float(np.mean((pred[valid]==direction[valid]).astype(float)+
                           .5*(pred[valid]==0))) if valid.any() else None
        scene_rows[scene]=dict(states=len(ix),selection_regret=float(regret.mean()),
                               pairwise_accuracy=pair,decisive_pair_cases=int(valid.sum()),
                               validation_B15=int(np.sum(success[ix,selected[ix]]>=15)))
    regret=float(np.mean([r['selection_regret'] for r in scene_rows.values()]))
    valid_pair=[r['pairwise_accuracy'] for r in scene_rows.values() if r['pairwise_accuracy'] is not None]
    pair=float(np.mean(valid_pair)) if valid_pair else None
    nll=float((success*np.logaddexp(0.,-logits)+failure*np.logaddexp(0.,logits)).sum()/
              (success+failure).sum())
    return dict(selection_regret=regret,pairwise_accuracy=pair,validation_nll=nll,
                validation_B15=sum(r['validation_B15'] for r in scene_rows.values()),
                by_scene=scene_rows,unknown=int(numerical.sum()))


def fit(panel,kind,seed):
    assert panel in PANELS and kind in ('state','eta_only') and seed in SEEDS
    folder=HERE/'fits'/panel/kind/f'seed{seed}'
    trainval=HERE/'fits'/panel/'trainval.npz'
    manifest=read(HERE/'fits'/panel/'trainval_manifest.json')
    plan=read(HERE/'unified_train_plan.json')
    assert manifest['trainval_sha256']==sha(trainval)
    assert manifest['plan_sha256']==sha(HERE/'unified_train_plan.json')
    assert not manifest['test_labels_opened']
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json')
        assert done['trainval_sha256']==sha(trainval)
        assert done['checkpoint_sha256']==sha(folder/'checkpoint.msgpack')
        return done
    folder.mkdir(parents=True,exist_ok=True)
    data=np.load(trainval)
    train=np.flatnonzero(data['split']=='train')
    val=np.flatnonzero(data['split']=='validation')
    assert len(train)==manifest['train_states'] and len(val)==manifest['validation_states']
    e=jnp.asarray(data['eta'],dtype=jnp.float32)
    s=jnp.asarray(data['success'],dtype=jnp.float32)
    f=jnp.asarray(data['failure'],dtype=jnp.float32)
    mask=np.asarray(data['success']+data['failure']>0)
    keys=[k for k in data.files if k.startswith('entity_')]
    ring=panel=='within_ring_TT'
    x=(jnp.asarray(data['physical23']) if ring else
       {key[7:]:jnp.asarray(data[key]) for key in keys})
    if kind=='eta_only':net=EtaCritic()
    elif ring:net=StateCritic()
    else:net=rep.Critic()

    def gather(indices):
        return x[indices] if ring else {k:v[indices] for k,v in x.items()}

    if kind=='eta_only':parameters=net.init(jax.random.PRNGKey(seed),e[:1,:1].reshape(1,3))
    else:parameters=net.init(jax.random.PRNGKey(seed),gather(jnp.array([0])),e[:1,:1].reshape(1,3))
    optimizer=optax.chain(optax.clip_by_global_norm(5.),
                          optax.adamw(.0008,weight_decay=.0001))
    opt_state=optimizer.init(parameters)
    mean_exposure=float(np.mean(np.asarray(s+f)[train]))
    assert mean_exposure>0

    @jax.jit
    def block(weights,state,draws):
        def step(carry,batch):
            p,o=carry;ii,jj=batch[:,0],batch[:,1]
            def loss(pp):
                if kind=='eta_only':z=net.apply(pp,e[ii,jj])
                else:z=net.apply(pp,gather(ii),e[ii,jj])
                return jnp.mean(s[ii,jj]*jax.nn.softplus(-z)+f[ii,jj]*jax.nn.softplus(z))/mean_exposure
            value,grad=jax.value_and_grad(loss)(p)
            updates,o=optimizer.update(grad,o,p)
            return (optax.apply_updates(p,updates),o),value
        return jax.lax.scan(step,(weights,state),draws)

    @jax.jit
    def predict(weights,ii,jj):
        return net.apply(weights,e[ii,jj]) if kind=='eta_only' else net.apply(weights,gather(ii),e[ii,jj])

    train_scenes=np.asarray(data['scene'])[train]
    pools={sc:np.asarray([(i,j) for i in train[np.asarray(data['scene'])[train]==sc]
                                  for j in range(16) if mask[i,j]],np.int32)
           for sc in sorted(set(train_scenes))}
    assert all(len(v)>=4000 for v in pools.values())
    val_i=np.repeat(val,16)
    val_j=np.tile(np.arange(16),len(val))
    rng=np.random.default_rng(seed)
    history=[];best=None;best_bytes=None
    started=time.monotonic();prior_step=0
    for step in STEPS:
        chunks=[]
        for sc,pool in pools.items():
            amount=128//len(pools)+(1 if list(pools).index(sc)<128%len(pools) else 0)
            chunks.append(pool[rng.integers(len(pool),size=(step-prior_step,amount))])
        draws=np.concatenate(chunks,axis=1)
        assert draws.shape==(step-prior_step,128,2)
        (parameters,opt_state),_=block(parameters,opt_state,jnp.asarray(draws))
        predictions=[]
        for start in range(0,len(val_i),256):
            ii=jnp.asarray(val_i[start:start+256],jnp.int32)
            jj=jnp.asarray(val_j[start:start+256],jnp.int32)
            predictions.append(np.asarray(predict(parameters,ii,jj)))
        logits=np.concatenate(predictions).reshape(len(val),16)
        score=metrics(logits,np.asarray(s)[val],np.asarray(f)[val],
                      np.asarray(data['numerical'])[val],np.asarray(data['scene'])[val])
        row=dict(step=step,**score)
        history.append(row)
        if best is None or select_key(row,plan['checkpoint_selection']['near_tie_tolerance'])<select_key(best,plan['checkpoint_selection']['near_tie_tolerance']):
            best=row;best_bytes=serialization.to_bytes(parameters)
        prior_step=step
        if step%1000==0:
            print(json.dumps(dict(panel=panel,kind=kind,seed=seed,step=step,
                                  regret=score['selection_regret'],best_step=best['step'],
                                  seconds=time.monotonic()-started)),flush=True)
    (folder/'checkpoint.msgpack').write_bytes(best_bytes)
    dump(folder/'history.json',history)
    done=dict(panel=panel,kind=kind,seed=seed,selected_step=best['step'],
              selection_regret=best['selection_regret'],pairwise_accuracy=best['pairwise_accuracy'],
              validation_nll=best['validation_nll'],validation_B15=best['validation_B15'],
              by_scene=best['by_scene'],trainval_sha256=sha(trainval),
              plan_sha256=sha(HERE/'unified_train_plan.json'),
              checkpoint_sha256=sha(folder/'checkpoint.msgpack'),
              elapsed_seconds=time.monotonic()-started,backend=jax.default_backend(),
              test_labels_used=False)
    dump(folder/'complete.json',done)
    return done


def freeze(panel):
    plan=read(HERE/'unified_train_plan.json')
    output={}
    for kind in ('state','eta_only'):
        rows=[read(HERE/'fits'/panel/kind/f'seed{seed}'/'complete.json') for seed in SEEDS]
        selected=min(rows,key=lambda r:(int(np.floor(r['selection_regret']/plan['checkpoint_selection']['near_tie_tolerance']+1e-9)),
                                        -(r['pairwise_accuracy'] if r['pairwise_accuracy'] is not None else -.5),
                                        r['validation_nll'],r['selected_step'],r['seed']))
        output[kind]=dict(runs=rows,selected=selected)
    output.update(panel=panel,test_labels_used=False,plan_sha256=sha(HERE/'unified_train_plan.json'),
                  trainval_sha256=sha(HERE/'fits'/panel/'trainval.npz'))
    target=HERE/'fits'/panel/'models_frozen.json'
    if target.exists():assert read(target)==output
    else:dump(target,output)
    print(json.dumps({k:(output[k]['selected']['seed'],output[k]['selected']['selected_step'])
                      for k in ('state','eta_only')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('fit','freeze'))
    p.add_argument('--panel',choices=tuple(PANELS),required=True)
    p.add_argument('--kind',choices=('state','eta_only'))
    p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.panel)
    else:
        assert a.kind and a.seed in SEEDS
        print(json.dumps(fit(a.panel,a.kind,a.seed)),flush=True)
