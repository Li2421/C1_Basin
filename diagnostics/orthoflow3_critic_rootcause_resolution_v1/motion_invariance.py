"""Matched TRAIN-only behavioral-equivalence constraint; frozen raw inputs."""
import argparse
import csv
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from scipy.special import expit
from . import motion_factorial_train as base
from .motion_freeze_train import OUT as OLD

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'motion_invariance_source'
RULE=ROOT/'motion_invariance_protocol.json'
read,write,sha=base.read,base.write,base.sha


def verify_relations(d,tr):
    with (base.OUT/'rest_invariance_audit/matched_seed_records.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='train']
    fields=('success','deadlock','timeout','collision','numerical_failure','episode_length','j_def','min_wall_distance','min_agent_distance')
    assert len(rows)==1024 and all(r[k+'_equal']=='True' for r in rows for k in fields)
    for k in ('success','failure','numerical'):
        assert np.array_equal(d[k][12:14,tr],d[k][14:16,tr])
    c=d['input_context']
    assert np.array_equal(c[12:14,tr,:88],c[14:16,tr,:88])
    assert np.array_equal(c[12:14,tr,104:],c[14:16,tr,104:])
    assert np.any(c[12:14,tr,88:104]!=c[14:16,tr,88:104])
    return len(rows)


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold,seed=index//3,base.SEEDS[index%3]
    held=base.folds()[fold];fit=[i for i in range(12) if i not in held]
    have=0 in fit and 1 in fit
    trainc=fit+([12,13,14,15] if have else [])
    d,x,tr,va,norm=base.load(fit)
    count=verify_relations(d,tr)
    dest=OUT/f'fold{fold}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists()
    reference=OLD/f'cv/fold{fold}/trunk_only/seed{seed}'
    if not have:
        write(dest/'complete.json',dict(fold=fold,seed=seed,checkpoint_reuse=str(reference),
            code_sha256=sha(__file__),protocol_sha256=sha(RULE),new_rollouts=0,target_labels_used=False,
            reason='No permitted program relation in this source fold; exact baseline reuse'))
        print(dict(fold=fold,seed=seed,reused=True),flush=True);return
    si,e,c=d['state_index'],d['normalized_eta'],d['input_context']
    model=base.model('rest_motion')
    p=model.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
    masks={k:k[-2] in ('trunk1','trunk2','out') for k in traverse_util.flatten_dict(p)}
    mask=traverse_util.unflatten_dict(masks)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    def make_step(constrain):
        @jax.jit
        def step(pp,oo,xx,ee,cc,ss,ff):
            def loss(params):
                z=model.apply(params,xx,ee,cc);q=ss/jnp.maximum(ss+ff,1)
                value=jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
                if constrain:
                    block=z.reshape(len(trainc),32)
                    value+=jnp.mean((block[-4:-2]-block[-2:])**2)
                return value
            value,g=jax.value_and_grad(loss)(pp)
            if constrain:g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
            update,oo=opt.update(g,oo,pp)
            if constrain:update=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),update,mask)
            return optax.apply_updates(pp,update),oo,value
        return step
    warm,fixed=make_step(False),make_step(True)
    predict=jax.jit(lambda pp,xx,ee,cc:model.apply(pp,xx,ee,cc))
    def evaluate(pp,condition='correct'):
        shift=np.roll(va.reshape(-1,2),-1,axis=0).ravel();scores=[]
        for ci in range(16):
            ss=si[shift] if condition in ('joint_state_context_shuffle','state_shuffle') else si[va]
            cc=c[ci,shift] if condition=='joint_state_context_shuffle' else c[ci,va]
            if condition=='wrong_controller':cc=c[(ci+1)%12,va]
            scores.append(np.asarray(predict(pp,gather(x,ss),jnp.asarray(e[va]),jnp.asarray(cc))))
        return np.asarray(scores)
    rng=np.random.default_rng(seed);history=[];predictions={};frozen=None
    for it in range(1,1501):
        draw=rng.choice(tr,32);ii,ci=np.tile(draw,len(trainc)),np.repeat(trainc,32)
        step=warm if it<=25 else fixed
        p,o,loss=step(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ci,ii]),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it==25:
            old=serialization.msgpack_restore((reference/'step25.msgpack').read_bytes())
            assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(old)))
            frozen={k:np.array(v) for k,v in traverse_util.flatten_dict(p).items() if not masks[k]}
        if it not in base.STEPS:continue
        z=evaluate(p);predictions[f'step{it}']=z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        prob=expit(z)
        history.append(dict(step=it,held_native_VAL=base.metrics(z,d,va,held),
            causal_motion_VAL=base.causal(z,d,va),
            invariance_probability_gap=float(abs(prob[12:14]-prob[14:16]).mean()),
            invariance_ranking_flips=int((z[12:14].reshape(2,16,2).argmax(-1)!=z[14:16].reshape(2,16,2).argmax(-1)).sum())))
    assert all(np.array_equal(v,traverse_util.flatten_dict(p)[k]) for k,v in frozen.items())
    for it in base.STEPS:
        pp=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in ('wrong_controller','joint_state_context_shuffle','state_shuffle'):
            predictions[f'step{it}__{condition}']=evaluate(pp,condition)
    np.savez_compressed(dest/'predictions.npz',**predictions)
    write(dest/'history.json',history);write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(fold=fold,seed=seed,checkpoint_reuse=None,TRAIN_invariance_seed_pairs=count,
        warm25_exact_replay=True,encoders_frozen=True,dataset_sha256=sha(base.OUT/'dataset.npz'),
        code_sha256=sha(__file__),protocol_sha256=sha(RULE),target_labels_used=False,new_rollouts=0))
    print(dict(fold=fold,seed=seed,completed=True),flush=True)


def summarize():
    rows=[];chosen=[];effects=[]
    d=dict(np.load(base.OUT/'dataset.npz'));va=np.flatnonzero(d['split']=='validation')
    for it in base.STEPS:
        rr=[]
        for fold in range(3):
            for seed in base.SEEDS:
                folder=OUT/f'fold{fold}/seed{seed}';done=read(folder/'complete.json')
                assert done['code_sha256']==sha(__file__) and done['protocol_sha256']==sha(RULE)
                actual=Path(done['checkpoint_reuse']) if done['checkpoint_reuse'] else folder
                z=np.load(actual/'predictions.npz')[f'step{it}']
                row=dict(step=it,fold=fold,seed=seed,**base.metrics(z,d,va,base.folds()[fold]))
                rows.append(row);rr.append(row)
        chosen.append(dict(step=it,NLL=float(np.mean([r['NLL'] for r in rr])),B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==seed) for seed in base.SEEDS]))
    best=min(chosen,key=lambda r:r['NLL'])
    for fold in range(3):
        for seed in base.SEEDS:
            folder=OUT/f'fold{fold}/seed{seed}';done=read(folder/'complete.json')
            actual=Path(done['checkpoint_reuse']) if done['checkpoint_reuse'] else folder
            p=np.load(actual/'predictions.npz');old=np.load(OLD/f'cv/fold{fold}/trunk_only/seed{seed}/predictions.npz')
            for arm,zz in (('constrained',p[f'step{best["step"]}']),('baseline400',old['step400'])):
                pp=expit(zz)
                effects.append(dict(fold=fold,seed=seed,arm=arm,constituents_seen=fold!=2,
                    probability_gap=float(abs(pp[12:14]-pp[14:16]).mean()),
                    ranking_flips=int((zz[12:14].reshape(2,16,2).argmax(-1)!=zz[14:16].reshape(2,16,2).argmax(-1)).sum()),
                    **base.causal(zz,d,va)))
    base.csvwrite(OUT/'source_trajectories.csv',rows);base.csvwrite(OUT/'invariance_and_causal_controls.csv',effects)
    write(OUT/'selection_frozen.json',dict(selected=best,criterion='pooledsourceheldcontrollerVALNLL',
        baseline_NLL=.5397345258129967,baseline_B15=[122,123,124],
        gate_passed=best['NLL']<.5397345258129967 and np.mean(best['B15_by_seed'])>=123,
        new_rollouts=0,target_labels_used=False,protocol_sha256=sha(RULE),code_sha256=sha(__file__)))
    print(dict(selected=best),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('train','summarize'))
    parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    train(args.index) if args.action=='train' else summarize()
