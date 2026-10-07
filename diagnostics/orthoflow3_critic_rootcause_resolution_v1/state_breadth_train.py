"""Matched64vs160family support, same controller/eta inputs and learning budget."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import motion_factorial_train as base
from .motion_state_breadth import OUT as NEW,RULE
from shared_rollout_db.src.rollout_db import connect,canonical

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'state_breadth_training'
ARMS=('original64_matched','expanded160')
read,write,sha=base.read,base.write,base.sha
CONTROLS=('correct','wrong_controller','joint_state_context_shuffle','state_shuffle')


def materialize():
    assert read(NEW/'working_state.json')['phase']=='postflight_complete'
    assert not (OUT/'dataset.npz').exists(),'Preserve materialized dataset'
    OUT.mkdir(exist_ok=True)
    old=dict(np.load(base.OUT/'dataset.npz'))
    op=read(base.SOURCE/'pairs.json');npairs=read(NEW/'pairs.json')
    os0=read(base.SOURCE/'states.json');ns=read(NEW/'states.json')
    tr=np.flatnonzero(old['split']=='train');va=np.flatnonzero(old['split']=='validation')
    assert len(tr)==128 and len(va)==32 and len(ns)==96
    states=[*os0[:64],*ns,*os0[64:]];assert len(states)==176
    lookup={s['uid']:i for i,s in enumerate(states)}
    assert len(lookup)==176
    pairs=[copy.deepcopy(p) for p in [*[op[i] for i in tr],*npairs,*[op[i] for i in va]]]
    for p in pairs:p['state_index']=lookup[p['state_uid']]
    n=len(pairs);assert n==352
    dest_old=np.r_[np.arange(128),np.arange(320,352)]
    oi=np.r_[tr,va]
    profiles=read(base.SOURCE/'protocol.json')['profiles']
    s=np.zeros((16,n),np.float32);f=s.copy();num=s.copy();keys=[]
    with connect(True) as db:
        for ci,profile in enumerate(profiles):
            for j,pair in enumerate(pairs):
                rows={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',
                    (pair['state_uid'],pair['eta_uid'],profile['controller_uid']))}
                rr=[rows[canonical({'future_index':k})] for k in range(16) if canonical({'future_index':k}) in rows]
                assert len(rr)>=int(pair['target_seeds'])
                assert all(not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in rr)
                valid=[r for r in rr if not r['numerical_failure']]
                s[ci,j]=sum(r['success'] for r in valid);f[ci,j]=len(valid)-s[ci,j];num[ci,j]=len(rr)-len(valid)
                keys.append(dict(controller_uid=profile['controller_uid'],state_uid=pair['state_uid'],eta_uid=pair['eta_uid'],
                    rollout_uids=[r['rollout_uid'] for r in rr],success=int(s[ci,j]),failure=int(f[ci,j]),numerical=int(num[ci,j])))
    for k,a in (('success',s),('failure',f),('numerical',num)):
        a[12:,dest_old]=old[k][12:,oi]
        assert np.array_equal(a[:,320:],old[k][:,va]),'VAL evidence changed'
    precision=np.load(ROOT/'motion_cached_precision/dataset.npz')
    for k,a in (('success',s),('failure',f),('numerical',num)):
        assert np.array_equal(a[:,dest_old],precision[k][:,oi]),'Original64all-cached-count baseline changed'
    d=dict(success=s,failure=f,numerical=num,standard_success=s.copy(),standard_failure=f.copy(),
        state_index=np.array([p['state_index'] for p in pairs]),eta=np.array([p['eta'] for p in pairs],np.float32),
        eta_index=np.array([p['eta_index'] for p in pairs]),split=np.array([p['split'] for p in pairs]),
        source_original64=np.r_[np.ones(128,bool),np.zeros(192,bool),np.ones(32,bool)])
    # Native inputs, exactly retaining old rows; the new rows have just been measured.
    for key,tail in (('context',(24,)),('agent_response',(4,16)),('goal_response',(16,)),('goal_motion_response',(16,))):
        a=np.zeros((16,n,*tail),np.float32);a[:,dest_old]=old[key][:,oi]
        for ci,profile in enumerate(profiles):
            name=profile['name']
            if key in ('context','agent_response'):v=np.load(NEW/f'inputs_{name}.npz')
            elif key=='goal_response':v=np.load(NEW/f'goal_rest/inputs_{name}.npz')
            else:v=np.load(NEW/f'goal_motion/inputs_{name}.npz')
            assert v['valid'].all()
            a[ci,128:320]=v[key]
        # These unlabelled program feature rows are never sampled for learning.
        # They merely implement the same saturated-gate query identities.
        if key in ('context','agent_response'):
            a[[12,14],128:320]=a[0,128:320];a[[13,15],128:320]=a[1,128:320]
        elif key=='goal_response':
            a[12:14,128:320]=a[[1,0],128:320];a[14:16,128:320]=a[[0,1],128:320]
        else:a[12:14,128:320]=a[14:16,128:320]=a[[1,0],128:320]
        d[key]=a
    ox=np.load(base.SOURCE/'frozen_model_entities.npz');nx=np.load(NEW/'frozen_model_entities.npz')
    x={k:np.concatenate((ox[k][:64],nx[k],ox[k][64:]),0) for k in ox.files}
    assert all(len(a)==176 and np.isfinite(a).all() for a in x.values())
    assert np.all(s[:12]+f[:12]>0) and np.all(s[12:,128:320]+f[12:,128:320]==0)
    assert np.array_equal(d['eta'][dest_old],old['eta'][oi])
    train_groups={state['source_group'] for state in states if state['split']=='train'}
    val_groups={state['source_group'] for state in states if state['split']=='validation'}
    assert len(train_groups)==160 and len(val_groups)==16 and not train_groups&val_groups
    np.savez_compressed(OUT/'dataset.npz',**d);np.savez_compressed(OUT/'entities.npz',**x)
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs);write(OUT/'native_DB_keys.json',keys)
    write(OUT/'dataset_manifest.json',dict(dataset_sha256=sha(OUT/'dataset.npz'),
        old_dataset_sha256=sha(base.OUT/'dataset.npz'),cached_precision_sha256=sha(ROOT/'motion_cached_precision/dataset.npz'),
        old_rows_and_VAL_exactly_unchanged=True,TRAIN_families=160,VAL_families=16,
        original_TRAIN_families=64,new_source_families=96,TRAIN_pairs=int(((s+f)[:,:320]>0).sum()),
        VAL_pairs=int(((s+f)[:,320:]>0).sum()),valid_observations=int((s+f).sum()),numerical=int(num.sum()),
        source_group_overlap=0,program_new_unlabelled_pairs_not_training=768,target_labels_used=False,
        protocol_sha256=sha(RULE),code_sha256=sha(__file__)))
    print(dict(materialized=True,TRAIN_families=160,VAL_families=16,pairs=int(((s+f)>0).sum())),flush=True)


def load(fit):
    manifest=read(OUT/'dataset_manifest.json');assert manifest['dataset_sha256']==sha(OUT/'dataset.npz')
    d=dict(np.load(OUT/'dataset.npz'));x=dict(np.load(OUT/'entities.npz'))
    _,_,_,_,norm=base.load(fit) # Intentionally freeze normalization to original64sourceTRAIN.
    d['normalized_eta']=(d['eta']-norm['eta_center'])/norm['eta_scale']
    blocks=[]
    for key,prefix in (('context','context'),('agent_response','agent'),('goal_response','goal'),('goal_motion_response','goal_motion')):
        a=(d[key]-norm[prefix+'_center'])/norm[prefix+'_scale']
        blocks.append(a.reshape(16,352,-1))
    d['input_context']=np.concatenate(blocks,-1).astype(np.float32)
    return d,x,np.arange(320),np.arange(320,352),norm


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
    m=base.model('rest_motion')
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
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
        shift=np.roll(va.reshape(-1,2),-1,axis=0).ravel();zz=[]
        for ci in range(16):
            ss=si[va];cc=c[ci,va]
            if condition=='wrong_controller':cc=c[(ci+1)%12,va]
            if condition=='joint_state_context_shuffle':ss,cc=si[shift],c[ci,shift]
            if condition=='state_shuffle':ss=si[shift]
            zz.append(np.asarray(predict(pp,gather(x,ss),jnp.asarray(e[va]),jnp.asarray(cc))))
        return np.asarray(zz)
    rng=np.random.default_rng(seed);history=[];predictions={};frozen=None;order=hashlib.sha256()
    for it in range(1,1501):
        u=rng.random(32);order.update(u.tobytes())
        # Uniform native160sampling, with40%exact matched old-family interventions.
        # Both arms/programs receive the identical uniform64-family remapping.
        old_u=np.where(u<.4,u/.4,(u-.4)/.6)
        old_idx=np.minimum((old_u*128).astype(int),127)
        new_idx=np.minimum((u*320).astype(int),319)
        ii=np.concatenate([new_idx if arm=='expanded160' and ci<12 else old_idx for ci in trainc])
        ci=np.repeat(trainc,32)
        assert np.all(d['success'][ci,ii]+d['failure'][ci,ii]>0)
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
        for condition in CONTROLS[1:]:predictions[f'step{it}__{condition}']=evaluate(pp,condition)
    np.savez_compressed(dest/'predictions.npz',**predictions)
    write(dest/'history.json',history);write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(arm=arm,fold=fold,seed=seed,fit=trainc,held=held,
        initial_parameters_sha256=init,underlying_uniform_draws_sha256=order.hexdigest(),
        normalization_from_original64_only=True,encoders_frozen=True,
        dataset_sha256=sha(OUT/'dataset.npz'),code_sha256=sha(__file__),protocol_sha256=sha(RULE),
        target_labels_used=False,new_rollouts=0))
    print(dict(arm=arm,fold=fold,seed=seed,complete=True),flush=True)


def summarize():
    d=dict(np.load(OUT/'dataset.npz'));va=np.arange(320,352);rows=[];controls=[];selected={}
    for fold in range(3):
        for seed in base.SEEDS:
            docs=[read(OUT/f'{a}/fold{fold}/seed{seed}/complete.json') for a in ARMS]
            for k in ('initial_parameters_sha256','underlying_uniform_draws_sha256','dataset_sha256','code_sha256','protocol_sha256'):
                assert docs[0][k]==docs[1][k]
            assert docs[0]['code_sha256']==sha(__file__)
    for arm in ARMS:
        ps={(f,s):dict(np.load(OUT/f'{arm}/fold{f}/seed{s}/predictions.npz')) for f in range(3) for s in base.SEEDS}
        candidates=[]
        for it in base.STEPS:
            rr=[]
            for (fold,seed),p in ps.items():
                r=dict(arm=arm,fold=fold,seed=seed,step=it,**base.metrics(p[f'step{it}'],d,va,base.folds()[fold]))
                rows.append(r);rr.append(r)
            candidates.append(dict(step=it,NLL=float(np.mean([r['NLL'] for r in rr])),
                B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in base.SEEDS]))
        best=min(candidates,key=lambda r:r['NLL']);selected[arm]=best
        for (fold,seed),p in ps.items():
            for condition in CONTROLS:
                z=p[f'step{best["step"]}'+('' if condition=='correct' else f'__{condition}')]
                controls.append(dict(arm=arm,fold=fold,seed=seed,condition=condition,
                    **base.metrics(z,d,va,base.folds()[fold]),**base.causal(z,d,va)))
    base.csvwrite(OUT/'source_trajectories.csv',rows);base.csvwrite(OUT/'input_controls.csv',controls)
    write(OUT/'selection_frozen.json',dict(selection=selected,criterion='sourceheldcontrollerVALNLL',
        matched_initialization_and_uniform_sampling=True,target_labels_used=False,new_training_rollouts=0,
        dataset_sha256=sha(OUT/'dataset.npz'),protocol_sha256=sha(RULE),code_sha256=sha(__file__)))
    print(selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('materialize','train','summarize'));p.add_argument('--index',type=int,default=0)
    a=p.parse_args()
    if a.action=='materialize':materialize()
    elif a.action=='train':train(a.index)
    else:summarize()
