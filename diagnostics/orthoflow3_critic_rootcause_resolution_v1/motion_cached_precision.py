"""Reuse compatible additional TRAIN trials; matched source-only fitting."""
import argparse,hashlib,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import motion_factorial_train as base
from .motion_eta_control import model as eta_model
from shared_rollout_db.src.rollout_db import connect,canonical

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'motion_cached_precision'
RULE=ROOT/'motion_cached_precision_protocol.json'
SOURCE=ROOT/'controller_function_support'
read,write,sha=base.read,base.write,base.sha


def prepare():
    assert not (OUT/'dataset.npz').exists(), 'Preserve materialized dataset'
    OUT.mkdir(exist_ok=True)
    original=base.OUT/'dataset.npz';d=dict(np.load(original))
    pairs=read(SOURCE/'pairs.json');profiles=read(SOURCE/'protocol.json')['profiles']
    train=np.flatnonzero(d['split']=='train');keys=[];extra=0;changed=0
    with connect(True) as db:
        for ci,c in enumerate(profiles):
            assert sha(c['path'])==c['sha256']
            for pi in train:
                pair=pairs[pi]
                rr={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',
                    (pair['state_uid'],pair['eta_uid'],c['controller_uid']))}
                rows=[]
                for j in range(16):
                    r=rr.get(canonical({'future_index':j}))
                    if r is None:continue
                    assert not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE'
                    rows.append(r)
                first=[rr[canonical({'future_index':j})] for j in range(4)]
                fs=sum(r['success'] for r in first if not r['numerical_failure'])
                fn=sum(not r['numerical_failure'] for r in first)
                assert fs==d['success'][ci,pi] and fn-fs==d['failure'][ci,pi]
                valid=[r for r in rows if not r['numerical_failure']]
                s=sum(r['success'] for r in valid);f=len(valid)-s
                extra+=len(valid)-fn;changed+=len(valid)>fn
                for k,v in (('success',s),('failure',f),('numerical',len(rows)-len(valid)),
                            ('standard_success',s),('standard_failure',f)):
                    d[k][ci,pi]=v
                keys.append(dict(controller_uid=c['controller_uid'],state_uid=pair['state_uid'],eta_uid=pair['eta_uid'],
                    rollout_uids=[r['rollout_uid'] for r in rows],observed_success=s,observed_failure=f,numerical=len(rows)-len(valid)))
    old=dict(np.load(original));va=np.flatnonzero(d['split']=='validation')
    counts=('success','failure','numerical','standard_success','standard_failure')
    for key in d:
        if key not in counts:assert np.array_equal(d[key],old[key]),key
        else:
            assert np.array_equal(d[key][:,va],old[key][:,va])
            assert np.array_equal(d[key][12:],old[key][12:])
    np.savez_compressed(OUT/'dataset.npz',**d)
    write(OUT/'dataset_DB_keys.json',keys)
    write(OUT/'dataset_manifest.json',dict(original_sha256=sha(original),dataset_sha256=sha(OUT/'dataset.npz'),
        rule_sha256=sha(RULE),code_sha256=sha(__file__),additional_compatible_trials=extra,
        changed_TRAIN_pairs=changed,target_labels_used=False,new_rollouts=0,all_inputs_and_VAL_exactly_unchanged=True))
    print(dict(extra_cached_trials=extra,changed_TRAIN_pairs=changed,new_rollouts=0),flush=True)


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold,seed=index//3,base.SEEDS[index%3]
    manifest=read(OUT/'dataset_manifest.json')
    assert manifest['code_sha256']==sha(__file__) and manifest['rule_sha256']==sha(RULE)
    held=base.folds()[fold];native=[i for i in range(12) if i not in held]
    trainc=native+([12,13,14,15] if 0 in native and 1 in native else [])
    d,x,tr,va,norm=base.load(native);updated=dict(np.load(OUT/'dataset.npz'))
    assert sha(OUT/'dataset.npz')==manifest['dataset_sha256']
    for key in ('success','failure','numerical','standard_success','standard_failure'):d[key]=updated[key]
    si,e,c=d['state_index'],d['normalized_eta'],d['input_context']
    for arm,steps in (('trunk_only',400),('eta_only',75)):
        dest=OUT/f'cv/fold{fold}/{arm}/seed{seed}'
        assert not (dest/'complete.json').exists(), 'Already trained'
        m=base.model('rest_motion') if arm=='trunk_only' else eta_model()
        p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
        init_hash=hashlib.sha256(serialization.to_bytes(p)).hexdigest()
        masks={k:k[-2] in {'trunk1','trunk2','out'} for k in traverse_util.flatten_dict(p)}
        mask=traverse_util.unflatten_dict(masks)
        opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
        def make_step(freeze):
            @jax.jit
            def update(pp,oo,xx,ee,cc,ss,ff):
                def loss(params):
                    z=m.apply(params,xx,ee,cc);q=ss/jnp.maximum(ss+ff,1)
                    return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
                v,g=jax.value_and_grad(loss)(pp)
                if freeze:g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
                u,oo=opt.update(g,oo,pp)
                if freeze:u=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),u,mask)
                return optax.apply_updates(pp,u),oo,v
            return update
        warm,fixed=make_step(False),make_step(True)
        rng=np.random.default_rng(seed);order=hashlib.sha256();frozen=None
        for it in range(1,steps+1):
            draw=rng.choice(tr,32);order.update(draw.tobytes())
            ii,ci=np.tile(draw,len(trainc)),np.repeat(trainc,len(draw))
            step=fixed if arm=='trunk_only' and it>25 else warm
            p,o,loss=step(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ci,ii]),
                jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
            if it==25:frozen={k:np.array(v) for k,v in traverse_util.flatten_dict(p).items() if not masks[k]}
        if arm=='trunk_only':assert all(np.array_equal(v,traverse_util.flatten_dict(p)[k]) for k,v in frozen.items())
        predict=jax.jit(lambda xx,ee,cc:m.apply(p,xx,ee,cc))
        predictions={}
        shifted=np.roll(va.reshape(-1,2),-1,axis=0).ravel()
        for condition in ('correct','joint_state_context_shuffle'):
            jj=shifted if condition!='correct' else va
            predictions[condition]=np.array([np.asarray(predict(gather(x,si[jj]),jnp.asarray(e[va]),jnp.asarray(c[k,jj]))) for k in range(16)])
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'checkpoint.msgpack').write_bytes(serialization.to_bytes(p))
        np.savez_compressed(dest/'predictions.npz',**predictions)
        write(dest/'normalization.json',norm)
        write(dest/'complete.json',dict(fold=fold,seed=seed,arm=arm,steps=steps,held=held,fit=trainc,
            initial_parameters_sha256=init_hash,batch_order_sha256=order.hexdigest(),dataset_sha256=sha(OUT/'dataset.npz'),
            correct=base.metrics(predictions['correct'],d,va,held),
            shuffled=base.metrics(predictions['joint_state_context_shuffle'],d,va,held),
            source_only=True,target_labels_used=False,new_rollouts=0))
        print(dict(fold=fold,seed=seed,arm=arm,complete=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','train'))
    parser.add_argument('--index',type=int,default=0);a=parser.parse_args()
    prepare() if a.action=='prepare' else train(a.index)
