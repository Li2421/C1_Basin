"""One source-only raw-context bypass; no new information, outcomes or targets."""
import argparse
import hashlib
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import state_breadth_train as data
from . import motion_factorial_train as base

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'context_raw_bypass'
RULE=ROOT/'context_raw_bypass_protocol.json'
read,write,sha=base.read,base.write,base.sha
CONTROLS=(*data.CONTROLS,'zero_raw_bypass')


def model(use_skip=True):
    import flax.linen as nn
    import jax.numpy as jnp
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    class Core(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context,raw):
            h=rep.Encoder(name='physical_encoder')(x)
            e=nn.silu(nn.Dense(32,name='eta_encoder')(eta))
            c=nn.silu(nn.Dense(32,name='context_encoder')(context))
            cid=nn.silu(nn.Dense(16,name='id_encoder')(jnp.zeros((len(eta),3))))
            a=nn.Dense(128,name='trunk1')(jnp.concatenate((h,e,c,cid),-1))
            skip=nn.Dense(128,use_bias=False,kernel_init=nn.initializers.zeros,name='raw_context_skip')(raw)
            a=a+skip if use_skip else a+0*skip
            z=nn.silu(a)
            z=nn.silu(nn.Dense(64,name='trunk2')(z))
            return nn.Dense(1,name='out')(z)[...,0]
    class Wrapper(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            agent=context[:,24:88].reshape(-1,x['agents'].shape[1],16)
            mean=agent.mean(1)
            xx={**x,'agents':jnp.concatenate((x['agents'],jnp.broadcast_to(mean[:,None],agent.shape)),-1)}
            c=jnp.concatenate((context[:,:24],context[:,88:120]),-1)
            return Core(name='core')(xx,eta,c,jnp.concatenate((c,mean),-1))
    return Wrapper()


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    assert sha(data.OUT/'dataset.npz')==read(RULE)['data_sha256']
    fold,seed=index//3,base.SEEDS[index%3]
    held=base.folds()[fold];fit=[i for i in range(12) if i not in held]
    trainc=fit+([12,13,14,15] if 0 in fit and 1 in fit else [])
    d,x,tr,va,norm=data.load(fit);si,e,c=d['state_index'],d['normalized_eta'],d['input_context']
    dest=OUT/f'fold{fold}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists()
    prior=data.OUT/f'expanded160/fold{fold}/seed{seed}'
    baseline_doc=read(prior/'complete.json')
    old=base.model('rest_motion');m=model()
    args=(gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
    oldp=old.init(jax.random.PRNGKey(seed),*args)
    assert hashlib.sha256(serialization.to_bytes(oldp)).hexdigest()==baseline_doc['initial_parameters_sha256']
    p=m.init(jax.random.PRNGKey(seed),*args)
    p['params']['core'].update(oldp['params']['core'])
    common=lambda pp:{'params':{'core':{k:v for k,v in pp['params']['core'].items() if k!='raw_context_skip'}}}
    assert hashlib.sha256(serialization.to_bytes(common(p))).hexdigest()==baseline_doc['initial_parameters_sha256']
    allkeys=traverse_util.flatten_dict(p)
    trainmask=traverse_util.unflatten_dict({k:k[-2] in ('trunk1','trunk2','out','raw_context_skip') for k in allkeys})
    warmmask=traverse_util.unflatten_dict({k:k[-2]!='raw_context_skip' for k in allkeys})
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    def make_step(mask):
        @jax.jit
        def step(pp,oo,xx,ee,cc,ss,ff):
            def loss(params):
                z=m.apply(params,xx,ee,cc);q=ss/jnp.maximum(ss+ff,1)
                return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
            value,g=jax.value_and_grad(loss)(pp)
            g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
            up,oo=opt.update(g,oo,pp)
            up=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),up,mask)
            return optax.apply_updates(pp,up),oo,value
        return step
    warm,fixed=make_step(warmmask),make_step(trainmask)
    predict=jax.jit(lambda pp,xx,ee,cc:m.apply(pp,xx,ee,cc))
    parity_args=(gather(x,si[va]),jnp.asarray(e[va]),jnp.asarray(c[fit[0],va]))
    init_error=float(np.max(abs(np.asarray(old.apply(oldp,*parity_args))-np.asarray(predict(p,*parity_args)))))
    assert init_error<2e-6,init_error
    def evaluate(pp,condition='correct'):
        shift=np.roll(va.reshape(-1,2),-1,axis=0).ravel();zz=[]
        # Keep the identical executable. A separate statically zeroed branch
        # triggers an empty CUDA graph in this installed XLA version.
        if condition=='zero_raw_bypass':
            pp={**pp,'params':{**pp['params'],'core':{**pp['params']['core'],
                'raw_context_skip':{'kernel':jnp.zeros_like(pp['params']['core']['raw_context_skip']['kernel'])}}}}
        for ci in range(16):
            ss=si[va];cc=c[ci,va]
            if condition=='wrong_controller':cc=c[(ci+1)%12,va]
            if condition=='joint_state_context_shuffle':ss,cc=si[shift],c[ci,shift]
            if condition=='state_shuffle':ss=si[shift]
            zz.append(np.asarray(predict(pp,gather(x,ss),jnp.asarray(e[va]),jnp.asarray(cc))))
        return np.asarray(zz)
    rng=np.random.default_rng(seed);order=hashlib.sha256();history=[];preds={};frozen=None;warm_error=None
    resume=(dest/'step1500.msgpack').exists()
    if resume:assert all((dest/f'step{it}.msgpack').exists() for it in base.STEPS)
    for it in range(1,1501):
        u=rng.random(32);order.update(u.tobytes())
        old_u=np.where(u<.4,u/.4,(u-.4)/.6)
        oi=np.minimum((old_u*128).astype(int),127);ni=np.minimum((u*320).astype(int),319)
        ii=np.concatenate([ni if ci<12 else oi for ci in trainc]);ci=np.repeat(trainc,32)
        if resume:
            if it not in base.STEPS:continue
            p=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        else:
            p,o,loss=(warm if it<=25 else fixed)(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ci,ii]),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it==25:
            baseline25=serialization.msgpack_restore((prior/'step25.msgpack').read_bytes())
            flat0=traverse_util.flatten_dict(baseline25);flat1=traverse_util.flatten_dict(common(p))
            warm_error=max(float(np.max(abs(np.asarray(flat0[k])-np.asarray(flat1[k])))) for k in flat0)
            assert warm_error<2e-5,warm_error
            assert np.max(abs(np.asarray(p['params']['core']['raw_context_skip']['kernel'])))==0
            frozen={k:np.asarray(v) for k,v in traverse_util.flatten_dict(p).items() if not traverse_util.flatten_dict(trainmask)[k]}
        if it not in base.STEPS:continue
        z=evaluate(p);preds[f'step{it}']=z
        if not resume:(dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,held_native_VAL=base.metrics(z,d,va,held),causal_motion_VAL=base.causal(z,d,va)))
    assert order.hexdigest()==baseline_doc['underlying_uniform_draws_sha256']
    assert all(np.array_equal(v,traverse_util.flatten_dict(p)[k]) for k,v in frozen.items())
    for it in base.STEPS:
        pp=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in CONTROLS[1:]:preds[f'step{it}__{condition}']=evaluate(pp,condition)
    np.savez_compressed(dest/'predictions.npz',**preds)
    write(dest/'history.json',history);write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(fold=fold,seed=seed,fit=trainc,held=held,init_parity_error=init_error,
        warm25_parameter_parity_error=warm_error,underlying_uniform_draws_sha256=order.hexdigest(),
        frozen_encoders_verified=True,extra_raw_skip_parameters=9216,
        parameter_count=sum(a.size for a in jax.tree.leaves(p)),dataset_sha256=sha(data.OUT/'dataset.npz'),
        baseline=str(prior),code_sha256=sha(__file__),protocol_sha256=sha(RULE),target_labels_used=False,new_rollouts=0,
        recovered_completed_training_after_evaluation_CUDA_error=resume,
        training_code_sha256='4c28da9ac5213f92fe7e3abd5ac34c9dc801757d75c75a334f0c4f0ff174b1f3' if resume else sha(__file__)))
    print(dict(fold=fold,seed=seed,complete=True,warm25_error=warm_error),flush=True)


def summarize():
    d=dict(np.load(data.OUT/'dataset.npz'));va=np.arange(320,352);rows=[];controls=[];choices=[]
    ps={}
    for fold in range(3):
        for seed in base.SEEDS:
            path=OUT/f'fold{fold}/seed{seed}';doc=read(path/'complete.json')
            assert doc['code_sha256']==sha(__file__)
            ps[(fold,seed)]=dict(np.load(path/'predictions.npz'))
    for step in base.STEPS:
        rr=[]
        for (fold,seed),p in ps.items():
            r=dict(step=step,fold=fold,seed=seed,**base.metrics(p[f'step{step}'],d,va,base.folds()[fold]));rows.append(r);rr.append(r)
        choices.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),
            B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==seed) for seed in base.SEEDS]))
    selected=min(choices,key=lambda r:r['NLL'])
    for (fold,seed),p in ps.items():
        for cond in CONTROLS:
            z=p[f'step{selected["step"]}'+('' if cond=='correct' else '__'+cond)]
            controls.append(dict(fold=fold,seed=seed,condition=cond,**base.metrics(z,d,va,base.folds()[fold]),**base.causal(z,d,va)))
    base.csvwrite(OUT/'trajectories.csv',rows);base.csvwrite(OUT/'input_controls.csv',controls)
    reference=read(data.OUT/'selection_frozen.json')['selection']['expanded160']
    write(OUT/'selection_frozen.json',dict(selected=selected,reference=reference,criterion='sourceheldcontrollerVALNLL',
        code_sha256=sha(__file__),protocol_sha256=sha(RULE),dataset_sha256=sha(data.OUT/'dataset.npz'),target_labels_used=False,new_rollouts=0))
    print(dict(selected=selected,reference=reference),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0)
    a=p.parse_args();train(a.index) if a.action=='train' else summarize()
