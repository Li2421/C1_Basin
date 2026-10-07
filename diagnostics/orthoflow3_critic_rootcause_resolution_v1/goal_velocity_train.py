"""Matched source-only rest/moving-goal input ablation; pure pair-equal NLL."""
import argparse
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .goal_velocity_response import OUT, SOURCE, read,write,sha
from .phase_factorial_train import load as old_load, metrics, causal, folds, STEPS,SEEDS
from .phase_factorial_support import OUT as DATA
from .goal_response_cv import csvwrite
KINDS=('rest_only','motion_only','rest_motion')


def model(kind):
    import flax.linen as nn
    import jax.numpy as jnp
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    class VelocityModel(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            a=context[:,24:88].reshape(-1,x['agents'].shape[1],16)
            a=jnp.broadcast_to(a.mean(1,keepdims=True),a.shape)
            xx={**x,'agents':jnp.concatenate([x['agents'],a],-1)}
            rest=context[:,88:104] if kind!='motion_only' else jnp.zeros_like(context[:,88:104])
            motion=context[:,104:120] if kind!='rest_only' else jnp.zeros_like(context[:,104:120])
            c=jnp.concatenate([context[:,:24],rest,motion],-1)
            return Critic(True,True,False,name='core')(xx,eta,c,jnp.zeros((len(eta),3)))
    return VelocityModel()


def load(fitc):
    d,x,tr,va,norm=old_load(fitc)
    values=[]
    for p in read(SOURCE/'protocol.json')['profiles']:
        a=np.load(OUT/f'inputs_{p["name"]}.npz')
        assert a['valid'].all()
        assert read(OUT/f'audit_{p["name"]}.json')['controller_sha256']==p['sha256']
        values.append(a['goal_motion_response'])
    # Both crossed controllers equal their late constituent at d=0.30,
    # at rest and at the new nonzero probe velocity; reuse exact measurements.
    mv=np.array(values)
    mv=np.concatenate([mv,mv[[1,0]]],0)
    center=mv[fitc][:,tr].mean((0,1));scale=np.maximum(mv[fitc][:,tr].std((0,1)),.05)
    norm.update(goal_motion_center=center.tolist(),goal_motion_scale=scale.tolist())
    d['input_context']=np.concatenate([d['input_context'],(mv-center)/scale],-1).astype(np.float32)
    return d,x,tr,va,norm


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold=index//9;kind=KINDS[(index%9)//3];seed=SEEDS[index%3]
    dest=OUT/'cv'/f'fold{fold}'/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    held=folds()[fold];fitc=[i for i in range(12) if i not in held]
    trainc=fitc+([12,13] if 0 in fitc and 1 in fitc else [])
    d,x,tr,va,norm=load(fitc);si=d['state_index'];e=d['normalized_eta'];c=d['input_context']
    m=model(kind);p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,c);q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=opt.update(g,o,p)
        return optax.apply_updates(p,u),o,v
    predict=jax.jit(lambda p,xx,e,c:m.apply(p,xx,e,c))
    def evaluate(pp,ii,condition='correct'):
        shift=np.roll(ii.reshape(-1,2),-1,axis=0).ravel();scores=[]
        for ci in range(14):
            cc=c[ci,ii].copy();ss=si[ii];ee=e[ii]
            if condition=='motion_wrong_controller':cc[:,104:]=c[(ci+1)%12,ii,104:]
            if condition=='motion_wrong_state':cc[:,104:]=c[ci,shift,104:]
            if condition=='rest_wrong_controller':cc[:,88:104]=c[(ci+1)%12,ii,88:104]
            if condition=='joint_state_context_shuffle':ss=si[shift];cc=c[ci,shift]
            if condition=='state_shuffle':ss=si[shift]
            if condition=='eta_shuffle':ee=e[ii.reshape(-1,2)[:,::-1].ravel()]
            scores.append(np.asarray(predict(pp,gather(x,ss),jnp.asarray(ee,jnp.float32),jnp.asarray(cc,jnp.float32))))
        return np.array(scores)
    rng=np.random.default_rng(seed);history=[];predictions={};dest.mkdir(parents=True,exist_ok=True)
    for it in range(1,1501):
        draw=rng.choice(tr,32);ii=np.tile(draw,len(trainc));ci=np.repeat(trainc,32)
        p,o,v=step(p,o,gather(x,si[ii]),jnp.asarray(e[ii],jnp.float32),jnp.asarray(c[ci,ii],jnp.float32),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in STEPS:continue
        z=evaluate(p,va);predictions[f'step{it}']=z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,held_native_VAL=metrics(z,d,va,held),seen_native_VAL=metrics(z,d,va,fitc),phase_VAL=metrics(z,d,va,[12,13]),causal_phase_VAL=causal(z,d,va)))
    for it in STEPS:
        pp=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in ('motion_wrong_controller','motion_wrong_state','rest_wrong_controller','joint_state_context_shuffle','state_shuffle','eta_shuffle'):
            predictions[f'step{it}__{condition}']=evaluate(pp,va,condition)
    write(dest/'normalization.json',norm);write(dest/'history.json',history)
    np.savez_compressed(dest/'predictions.npz',indices=va,**predictions)
    write(dest/'complete.json',dict(fold=fold,kind=kind,seed=seed,fit=trainc,held=held,
        data_sha256=sha(DATA/'dataset.npz'),protocol_sha256=sha(OUT/'protocol.json'),code_sha256=sha(__file__),
        parameter_count=int(sum(a.size for a in jax.tree.leaves(p))),new_rollouts=0,target_labels_used=False))
    print(dict(fold=fold,kind=kind,seed=seed,done=True),flush=True)


def summarize():
    rows=[];chosen={};controls=[]
    for kind in KINDS:
        candidates=[]
        for step in STEPS:
            rr=[]
            for fold in range(3):
                for seed in SEEDS:
                    path=OUT/'cv'/f'fold{fold}'/kind/f'seed{seed}';read(path/'complete.json')
                    r=next(a for a in read(path/'history.json') if a['step']==step)
                    rr.append(dict(kind=kind,fold=fold,seed=seed,step=step,**r['held_native_VAL']))
            rows+=rr
            candidates.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in SEEDS],cases_per_seed=192))
        best=min(candidates,key=lambda r:r['NLL']);chosen[kind]=best
        for fold in range(3):
            d,x,tr,va,n=load([i for i in range(12) if i not in folds()[fold]])
            for seed in SEEDS:
                pp=np.load(OUT/'cv'/f'fold{fold}'/kind/f'seed{seed}'/'predictions.npz')
                for condition in ('correct','motion_wrong_controller','motion_wrong_state','rest_wrong_controller','joint_state_context_shuffle','state_shuffle','eta_shuffle'):
                    key=f"step{best['step']}"+('' if condition=='correct' else f'__{condition}');z=pp[key]
                    for ev,cs in (('held_native',folds()[fold]),('phase',[12,13])):
                        controls.append(dict(kind=kind,fold=fold,seed=seed,step=best['step'],condition=condition,evaluation=ev,**metrics(z,d,va,cs),**causal(z,d,va)))
    csvwrite(OUT/'source_trajectories.csv',rows);csvwrite(OUT/'selected_input_controls.csv',controls)
    write(OUT/'source_selection.json',dict(selection=chosen,criterion='pooledsourceheldcontrollerVALNLL',new_rollouts=0,target_labels_used=False,independent_confirmation=False,
        caution='This sourceCV is for selection, not final evidence. No generator or old frozen tests changed.'))
    print(chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='train':train(a.index)
    else:summarize()
