"""Minimal representation-unfreezing control, no new inputs or labels."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from scipy.special import expit
from . import motion_factorial_train as base
from .motion_freeze_train import OUT as OLD
from .motion_invariance import verify_relations

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'motion_context_refit_source'
RULE=ROOT/'motion_context_refit_protocol.json'
ARMS=('context_nll','context_invariant')
read,write,sha=base.read,base.write,base.sha


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold,seed=index//3,base.SEEDS[index%3]
    held=base.folds()[fold];fit=[i for i in range(12) if i not in held]
    have=0 in fit and 1 in fit;trainc=fit+([12,13,14,15] if have else [])
    d,x,tr,va,norm=base.load(fit);verify_relations(d,tr)
    si,e,c=d['state_index'],d['normalized_eta'],d['input_context']
    for arm in ARMS:
        dest=OUT/f'fold{fold}/{arm}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        assert not (dest/'complete.json').exists()
        if not have and arm=='context_invariant':
            reuse=OUT/f'fold{fold}/context_nll/seed{seed}'
            assert (reuse/'complete.json').exists()
            write(dest/'complete.json',dict(fold=fold,seed=seed,arm=arm,checkpoint_reuse=str(reuse),
                code_sha256=sha(__file__),protocol_sha256=sha(RULE),target_labels_used=False,new_rollouts=0))
            continue
        model=base.model('rest_motion')
        p=model.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
        masks={k:k[-2] in ('context_encoder','trunk1','trunk2','out') for k in traverse_util.flatten_dict(p)}
        mask=traverse_util.unflatten_dict(masks)
        opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
        def make_step(stage):
            @jax.jit
            def step(pp,oo,xx,ee,cc,ss,ff):
                def loss(params):
                    z=model.apply(params,xx,ee,cc);q=ss/jnp.maximum(ss+ff,1)
                    value=jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
                    if stage and have and arm=='context_invariant':
                        block=z.reshape(len(trainc),32)
                        value+=jnp.mean((block[-4:-2]-block[-2:])**2)
                    return value
                value,g=jax.value_and_grad(loss)(pp)
                if stage:g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
                update,oo=opt.update(g,oo,pp)
                if stage:update=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),update,mask)
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
                ref=OLD/f'cv/fold{fold}/trunk_only/seed{seed}/step25.msgpack'
                old=serialization.msgpack_restore(ref.read_bytes())
                assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(old)))
                frozen={k:np.array(v) for k,v in traverse_util.flatten_dict(p).items() if not masks[k]}
            if it not in base.STEPS:continue
            z=evaluate(p);predictions[f'step{it}']=z
            (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
            pp=expit(z)
            history.append(dict(step=it,held_native_VAL=base.metrics(z,d,va,held),
                causal_motion_VAL=base.causal(z,d,va),
                nuisance_probability_gap=float(abs(pp[12:14]-pp[14:16]).mean())))
        assert all(np.array_equal(v,traverse_util.flatten_dict(p)[k]) for k,v in frozen.items())
        for it in base.STEPS:
            pp=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
            for condition in ('wrong_controller','joint_state_context_shuffle','state_shuffle'):
                predictions[f'step{it}__{condition}']=evaluate(pp,condition)
        np.savez_compressed(dest/'predictions.npz',**predictions)
        write(dest/'history.json',history);write(dest/'normalization.json',norm)
        write(dest/'complete.json',dict(fold=fold,seed=seed,arm=arm,checkpoint_reuse=None,
            warm25_exact_replay=True,physical_eta_id_frozen=True,dataset_sha256=sha(base.OUT/'dataset.npz'),
            code_sha256=sha(__file__),protocol_sha256=sha(RULE),target_labels_used=False,new_rollouts=0))
        print(dict(fold=fold,seed=seed,arm=arm,completed=True),flush=True)


def summarize():
    d=dict(np.load(base.OUT/'dataset.npz'));va=np.flatnonzero(d['split']=='validation')
    rows=[];effects=[];chosen={}
    for arm in ARMS:
        loaded={}
        for fold in range(3):
            for seed in base.SEEDS:
                folder=OUT/f'fold{fold}/{arm}/seed{seed}';done=read(folder/'complete.json')
                assert done['code_sha256']==sha(__file__) and done['protocol_sha256']==sha(RULE)
                actual=Path(done['checkpoint_reuse']) if done['checkpoint_reuse'] else folder
                loaded[fold,seed]=dict(np.load(actual/'predictions.npz'))
        candidates=[]
        for it in base.STEPS:
            rr=[]
            for (fold,seed),pred in loaded.items():
                row=dict(arm=arm,step=it,fold=fold,seed=seed,**base.metrics(pred[f'step{it}'],d,va,base.folds()[fold]))
                rr.append(row);rows.append(row)
            candidates.append(dict(step=it,NLL=float(np.mean([r['NLL'] for r in rr])),
                B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==seed) for seed in base.SEEDS]))
        best=min(candidates,key=lambda r:r['NLL']);chosen[arm]=best
        for (fold,seed),pred in loaded.items():
            z=pred[f'step{best["step"]}'];p=expit(z)
            for condition in ('correct','wrong_controller','joint_state_context_shuffle','state_shuffle'):
                zz=pred[f'step{best["step"]}'+('' if condition=='correct' else f'__{condition}')]
                effects.append(dict(arm=arm,fold=fold,seed=seed,condition=condition,
                    **base.metrics(zz,d,va,base.folds()[fold]),**base.causal(zz,d,va),
                    nuisance_probability_gap=float(abs(expit(zz)[12:14]-expit(zz)[14:16]).mean())))
    base.csvwrite(OUT/'source_trajectories.csv',rows);base.csvwrite(OUT/'input_controls.csv',effects)
    write(OUT/'selection_frozen.json',dict(selection=chosen,criterion='pooledsourceheldcontrollerVALNLL',
        baseline_NLL=.5397345258129967,baseline_B15=[122,123,124],
        new_rollouts=0,target_labels_used=False,code_sha256=sha(__file__),protocol_sha256=sha(RULE)))
    print(chosen,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('train','summarize'))
    parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    train(args.index) if args.action=='train' else summarize()
