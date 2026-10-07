"""Source-only input deletions; do not alter deployed/frozen candidates."""
import argparse,hashlib,os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import context_raw_bypass as skip
from . import state_breadth_train as data
from . import motion_factorial_train as base
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'raw_input_ablation'
RULE=ROOT/'raw_input_ablation_protocol.json'
ARMS=('without_explicit_h','without_initial_response')
read,write,sha=base.read,base.write,base.sha


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    arm,fold,seed=ARMS[index//9],index%9//3,base.SEEDS[index%3]
    held=base.folds()[fold];fit=[i for i in range(12) if i not in held]
    trainc=fit+([12,13,14,15] if 0 in fit and 1 in fit else [])
    d,x,tr,va,norm=data.load(fit);si,e,c=d['state_index'],d['normalized_eta'],d['input_context'].copy()
    if arm=='without_explicit_h':
        x={k:(v if k in ('agent_mask','obstacle_mask') else np.zeros_like(v)) for k,v in x.items()}
    else:c[:,:,:88]=0
    dest=OUT/arm/f'fold{fold}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists()
    m=skip.model();old=base.model('rest_motion')
    args=(gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
    p=m.init(jax.random.PRNGKey(seed),*args);p['params']['core'].update(old.init(jax.random.PRNGKey(seed),*args)['params']['core'])
    init=hashlib.sha256(serialization.to_bytes(p)).hexdigest()
    fk=traverse_util.flatten_dict(p)
    masks={mode:traverse_util.unflatten_dict({k:(k[-2]!='raw_context_skip' if mode=='warm' else k[-2] in ('trunk1','trunk2','out','raw_context_skip')) for k in fk}) for mode in ('warm','fixed')}
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    def make_step(mask):
        @jax.jit
        def step(pp,oo,xx,ee,cc,s,f):
            def loss(params):
                z=m.apply(params,xx,ee,cc);q=s/jnp.maximum(s+f,1)
                return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
            v,g=jax.value_and_grad(loss)(pp)
            g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
            up,oo=opt.update(g,oo,pp);up=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),up,mask)
            return optax.apply_updates(pp,up),oo,v
        return step
    warm,fixed=make_step(masks['warm']),make_step(masks['fixed'])
    predict=jax.jit(lambda pp,xx,ee,cc:m.apply(pp,xx,ee,cc))
    def evaluate(pp,condition='correct'):
        shift=np.roll(va.reshape(-1,2),-1,axis=0).ravel();zz=[]
        for ci in range(16):
            ss=si[va];cc=c[ci,va]
            if condition=='wrong_controller':cc=c[(ci+1)%12,va]
            if condition=='joint_state_context_shuffle':ss,cc=si[shift],c[ci,shift]
            if condition=='state_shuffle':ss=si[shift]
            zz.append(np.asarray(predict(pp,gather(x,ss),jnp.asarray(e[va]),jnp.asarray(cc))))
        return np.asarray(zz)
    rng=np.random.default_rng(seed);order=hashlib.sha256();preds={};history=[]
    for it in range(1,1501):
        u=rng.random(32);order.update(u.tobytes());old_u=np.where(u<.4,u/.4,(u-.4)/.6)
        oi=np.minimum((old_u*128).astype(int),127);ni=np.minimum((u*320).astype(int),319)
        ii=np.concatenate([ni if ci<12 else oi for ci in trainc]);ci=np.repeat(trainc,32)
        p,o,v=(warm if it<=25 else fixed)(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ci,ii]),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in base.STEPS:continue
        z=evaluate(p);preds[f'step{it}']=z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,held_native_VAL=base.metrics(z,d,va,held),causal_motion_VAL=base.causal(z,d,va)))
    for it in base.STEPS:
        pp=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for cond in data.CONTROLS[1:]:preds[f'step{it}__{cond}']=evaluate(pp,cond)
    np.savez_compressed(dest/'predictions.npz',**preds);write(dest/'history.json',history);write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(arm=arm,fold=fold,seed=seed,fit=trainc,held=held,
        initial_parameters_sha256=init,draw_sha256=order.hexdigest(),code_sha256=sha(__file__),protocol_sha256=sha(RULE),
        dataset_sha256=sha(data.OUT/'dataset.npz'),new_rollouts=0,target_labels_used=False))
    print(dict(arm=arm,fold=fold,seed=seed,complete=True),flush=True)


def summarize():
    d=dict(np.load(data.OUT/'dataset.npz'));va=np.arange(320,352);selected={};rows=[];controls=[]
    for fold in range(3):
        for seed in base.SEEDS:
            docs=[read(OUT/a/f'fold{fold}/seed{seed}/complete.json') for a in ARMS]
            for key in ('initial_parameters_sha256','draw_sha256','dataset_sha256'):assert docs[0][key]==docs[1][key]
    for arm in ARMS:
        ps={(f,s):dict(np.load(OUT/arm/f'fold{f}/seed{s}/predictions.npz')) for f in range(3) for s in base.SEEDS};choices=[]
        for step in base.STEPS:
            rr=[]
            for (fold,seed),p in ps.items():
                r=dict(arm=arm,fold=fold,seed=seed,step=step,**base.metrics(p[f'step{step}'],d,va,base.folds()[fold]));rows.append(r);rr.append(r)
            choices.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in base.SEEDS]))
        best=min(choices,key=lambda r:r['NLL']);selected[arm]=best
        for (fold,seed),p in ps.items():
            for cond in data.CONTROLS:
                z=p[f'step{best["step"]}'+('' if cond=='correct' else '__'+cond)]
                controls.append(dict(arm=arm,fold=fold,seed=seed,condition=cond,**base.metrics(z,d,va,base.folds()[fold]),**base.causal(z,d,va)))
    base.csvwrite(OUT/'trajectories.csv',rows);base.csvwrite(OUT/'input_controls.csv',controls)
    write(OUT/'selection_frozen.json',dict(selected=selected,full_reference=read(skip.OUT/'selection_frozen.json')['selected'],
        criterion='sourceheldcontrollerVALNLL',new_rollouts=0,target_labels_used=False,code_sha256=sha(__file__),protocol_sha256=sha(RULE)))
    print(selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0)
    a=p.parse_args();train(a.index) if a.action=='train' else summarize()
