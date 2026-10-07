"""Matched extra-information versus extra-capacity control, source only."""
import argparse
import hashlib
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import motion_factorial_train as base
from .controller_response_dispersion import OUT as INPUTS,RULE as INPUT_RULE
from .goal_motion_alias import weight

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'response_dispersion_training'
RULE=ROOT/'response_dispersion_training_protocol.json'
ARMS=('mean_only_capacity_control','mean_plus_dispersion')
CONTROLS=('correct','dispersion_wrong_controller','dispersion_wrong_state','context_wrong_controller','joint_state_context_shuffle')
read,write,sha=base.read,base.write,base.sha


def load(fit):
    d,x,tr,va,norm=base.load(fit)
    profiles=read(base.SOURCE/'protocol.json')['profiles'];values=[]
    for p in profiles:
        audit=read(INPUTS/f'audit_{p["name"]}.json')
        assert audit['controller_sha256']==p['sha256'] and audit['rule_sha256']==sha(INPUT_RULE)
        values.append(np.load(INPUTS/f'inputs_{p["name"]}.npz')['response_std'])
    native=np.asarray(values)
    assert native.shape==(12,160,32)
    # Ring maxspeed=.52 -> moving query speed=.26, above saturated gate .05.
    assert weight(.30,0.)==0. and weight(.30,.26)==1.
    phase=native[[1,0]]
    motion=np.concatenate((native[[0,1],:,:16],native[[1,0],:,16:]),-1)
    values=np.concatenate((native,phase,motion),0)
    center=values[fit][:,tr].mean((0,1));scale=np.maximum(values[fit][:,tr].std((0,1)),.01)
    norm.update(dispersion_center=center.tolist(),dispersion_scale=scale.tolist())
    d['input_context']=np.concatenate((d['input_context'],(values-center)/scale),-1).astype(np.float32)
    assert d['input_context'].shape==(16,160,152) and np.isfinite(d['input_context']).all()
    return d,x,tr,va,norm


def model(use_dispersion):
    import flax.linen as nn
    import jax.numpy as jnp
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    class DispersionModel(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            a=context[:,24:88].reshape(-1,x['agents'].shape[1],16)
            a=jnp.broadcast_to(a.mean(1,keepdims=True),a.shape)
            xx={**x,'agents':jnp.concatenate((x['agents'],a),-1)}
            extra=context[:,120:152] if use_dispersion else jnp.zeros_like(context[:,120:152])
            c=jnp.concatenate((context[:,:24],context[:,88:120],extra),-1)
            return Critic(True,True,False,name='core')(xx,eta,c,jnp.zeros((len(eta),3)))
    return DispersionModel()


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    arm=ARMS[index//9];fold=index%9//3;seed=base.SEEDS[index%3]
    held=base.folds()[fold];fit=[i for i in range(12) if i not in held]
    trainc=fit+([12,13,14,15] if 0 in fit and 1 in fit else [])
    d,x,tr,va,norm=load(fit);si,e,c=d['state_index'],d['normalized_eta'],d['input_context']
    dest=OUT/f'{arm}/fold{fold}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists()
    m=model(arm=='mean_plus_dispersion')
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,152)))
    init=hashlib.sha256(serialization.to_bytes(p)).hexdigest()
    masks={k:k[-2] in ('trunk1','trunk2','out') for k in traverse_util.flatten_dict(p)}
    mask=traverse_util.unflatten_dict(masks)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    def make_step(freeze):
        @jax.jit
        def step(pp,oo,xx,ee,cc,ss,ff):
            def loss(params):
                z=m.apply(params,xx,ee,cc);q=ss/jnp.maximum(ss+ff,1)
                return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
            value,g=jax.value_and_grad(loss)(pp)
            if freeze:g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
            update,oo=opt.update(g,oo,pp)
            if freeze:update=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),update,mask)
            return optax.apply_updates(pp,update),oo,value
        return step
    warm,fixed=make_step(False),make_step(True)
    predict=jax.jit(lambda pp,xx,ee,cc:m.apply(pp,xx,ee,cc))
    def evaluate(pp,condition='correct'):
        shift=np.roll(va.reshape(-1,2),-1,axis=0).ravel();scores=[]
        for ci in range(16):
            ss=si[va];cc=c[ci,va].copy()
            if condition=='dispersion_wrong_controller':cc[:,120:]=c[(ci+1)%12,va,120:]
            if condition=='dispersion_wrong_state':cc[:,120:]=c[ci,shift,120:]
            if condition=='context_wrong_controller':cc=c[(ci+1)%12,va]
            if condition=='joint_state_context_shuffle':ss,cc=si[shift],c[ci,shift]
            scores.append(np.asarray(predict(pp,gather(x,ss),jnp.asarray(e[va]),jnp.asarray(cc))))
        return np.asarray(scores)
    rng=np.random.default_rng(seed);history=[];predictions={};frozen=None;order=hashlib.sha256()
    for it in range(1,1501):
        draw=rng.choice(tr,32);order.update(draw.tobytes());ii,ci=np.tile(draw,len(trainc)),np.repeat(trainc,32)
        step=warm if it<=25 else fixed
        p,o,loss=step(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ci,ii]),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it==25:frozen={k:np.array(v) for k,v in traverse_util.flatten_dict(p).items() if not masks[k]}
        if it not in base.STEPS:continue
        z=evaluate(p);predictions[f'step{it}']=z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,held_native_VAL=base.metrics(z,d,va,held),causal_motion_VAL=base.causal(z,d,va)))
    assert all(np.array_equal(v,traverse_util.flatten_dict(p)[k]) for k,v in frozen.items())
    for it in base.STEPS:
        pp=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in CONTROLS[1:]:
            z=evaluate(pp,condition);predictions[f'step{it}__{condition}']=z
            if arm=='mean_only_capacity_control' and condition.startswith('dispersion'):
                assert np.array_equal(z,predictions[f'step{it}'])
    np.savez_compressed(dest/'predictions.npz',**predictions)
    write(dest/'history.json',history);write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(fold=fold,seed=seed,arm=arm,initial_parameters_sha256=init,
        batch_order_sha256=order.hexdigest(),encoders_frozen=True,
        dataset_sha256=sha(base.OUT/'dataset.npz'),rule_sha256=sha(RULE),code_sha256=sha(__file__),
        input_rule_sha256=sha(INPUT_RULE),target_labels_used=False,new_rollouts=0))
    print(dict(arm=arm,fold=fold,seed=seed,completed=True,new_rollouts=0),flush=True)


def summarize():
    d=dict(np.load(base.OUT/'dataset.npz'));va=np.flatnonzero(d['split']=='validation')
    rows=[];controls=[];chosen={}
    for fold in range(3):
        for seed in base.SEEDS:
            docs=[read(OUT/f'{a}/fold{fold}/seed{seed}/complete.json') for a in ARMS]
            for field in ('initial_parameters_sha256','batch_order_sha256','dataset_sha256','rule_sha256','code_sha256'):
                assert docs[0][field]==docs[1][field]
            assert docs[0]['code_sha256']==sha(__file__) and docs[0]['rule_sha256']==sha(RULE)
    for arm in ARMS:
        loaded={(f,s):dict(np.load(OUT/f'{arm}/fold{f}/seed{s}/predictions.npz')) for f in range(3) for s in base.SEEDS}
        candidates=[]
        for it in base.STEPS:
            rr=[]
            for (fold,seed),p in loaded.items():
                row=dict(arm=arm,fold=fold,seed=seed,step=it,**base.metrics(p[f'step{it}'],d,va,base.folds()[fold]))
                rows.append(row);rr.append(row)
            candidates.append(dict(step=it,NLL=float(np.mean([r['NLL'] for r in rr])),
                B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in base.SEEDS]))
        best=min(candidates,key=lambda r:r['NLL']);chosen[arm]=best
        for (fold,seed),p in loaded.items():
            for condition in CONTROLS:
                z=p[f'step{best["step"]}'+('' if condition=='correct' else f'__{condition}')]
                controls.append(dict(arm=arm,fold=fold,seed=seed,condition=condition,
                    **base.metrics(z,d,va,base.folds()[fold]),**base.causal(z,d,va)))
    base.csvwrite(OUT/'source_trajectories.csv',rows);base.csvwrite(OUT/'input_controls.csv',controls)
    write(OUT/'selection_frozen.json',dict(selection=chosen,criterion='pooledsourceheldcontrollerVALNLL',
        matched_initialization_and_order=True,new_rollouts=0,target_labels_used=False,
        code_sha256=sha(__file__),protocol_sha256=sha(RULE)))
    print(chosen,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('train','summarize'));parser.add_argument('--index',type=int,default=0)
    a=parser.parse_args();train(a.index) if a.action=='train' else summarize()
