"""Isolate encoder drift from late readout/interaction fitting, no new labels."""
import argparse
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from .motion_factorial_train import OUT as SOURCE, load, model, metrics, causal, folds, SEEDS, STEPS, CONTROLS, read, write, sha, csvwrite
OUT = SOURCE.parent/'motion_freeze_control'
RULE = SOURCE.parent/'motion_freeze_protocol.json'
ARMS = ('head_only', 'trunk_only')


def train(index):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization, traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold, arm, seed = index//6, ARMS[index%6//3], SEEDS[index%3]
    dest = OUT/'cv'/f'fold{fold}'/arm/f'seed{seed}'
    if (dest/'complete.json').exists():
        return
    held = folds()[fold]
    fitc = [i for i in range(12) if i not in held]
    have = 0 in fitc and 1 in fitc
    trainc = fitc+([12,13,14,15] if have else [])
    d,x,tr,va,norm = load(fitc)
    si,e,c = d['state_index'], d['normalized_eta'], d['input_context']
    m = model('rest_motion')
    p = m.init(jax.random.PRNGKey(seed), gather(x,si[:1]), jnp.zeros((1,3)), jnp.zeros((1,120)))
    flat = traverse_util.flatten_dict(p)
    allowed = {'out'} if arm=='head_only' else {'trunk1','trunk2','out'}
    masks = {key: key[-2] in allowed for key in flat}
    mask = traverse_util.unflatten_dict(masks)
    assert any(masks.values()) and not all(masks.values())
    opt = optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001))
    o = opt.init(p)
    def make_step(freeze):
        @jax.jit
        def step(p,o,xx,e,c,s,f):
            def loss(pp):
                z=m.apply(pp,xx,e,c); q=s/jnp.maximum(s+f,1)
                return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
            v,g=jax.value_and_grad(loss)(p)
            if freeze:
                g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
            u,o=opt.update(g,o,p)
            if freeze:
                u=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),u,mask)
            return optax.apply_updates(p,u),o,v
        return step
    warm_step,freeze_step=make_step(False),make_step(True)
    predict=jax.jit(lambda pp,xx,ee,cc:m.apply(pp,xx,ee,cc))
    def evaluate(pp,ii,condition='correct'):
        shift=np.roll(ii.reshape(-1,2),-1,axis=0).ravel(); scores=[]
        for ci in range(16):
            cc,ss,ee=c[ci,ii].copy(),si[ii],e[ii]
            if condition=='motion_parent_replacement' and ci in (14,15):cc[:,104:]=c[ci-14,ii,104:]
            if condition=='motion_wrong_controller':cc[:,104:]=c[(ci+1)%12,ii,104:]
            if condition=='motion_wrong_state':cc[:,104:]=c[ci,shift,104:]
            if condition=='rest_wrong_controller':cc[:,88:104]=c[(ci+1)%12,ii,88:104]
            if condition=='context_wrong_state':cc=c[ci,shift]
            if condition=='joint_state_context_shuffle':ss,cc=si[shift],c[ci,shift]
            if condition=='state_shuffle':ss=si[shift]
            if condition=='eta_shuffle':ee=e[ii.reshape(-1,2)[:,::-1].ravel()]
            scores.append(np.asarray(predict(pp,gather(x,ss),jnp.asarray(ee,jnp.float32),jnp.asarray(cc,jnp.float32))))
        return np.array(scores)
    rng=np.random.default_rng(seed);history=[];predictions={}
    dest.mkdir(parents=True,exist_ok=True)
    source=SOURCE/'cv'/f'fold{fold}'/'motion_intervention'/'rest_motion'/f'seed{seed}'
    frozen_parameters=None
    for it in range(1,1501):
        draw=rng.choice(tr,32);ii=np.tile(draw,len(trainc));ci=np.repeat(trainc,32)
        update=warm_step if it<=25 else freeze_step
        p,o,v=update(p,o,gather(x,si[ii]),jnp.asarray(e[ii],jnp.float32),jnp.asarray(c[ci,ii],jnp.float32),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it==25:
            old=serialization.msgpack_restore((source/'step25.msgpack').read_bytes())
            assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(old))), 'Warm-start must exactly replay baseline'
            frozen_parameters={k:np.array(v) for k,v in traverse_util.flatten_dict(p).items() if not masks[k]}
        if it not in STEPS:continue
        z=evaluate(p,va);predictions[f'step{it}']=z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        tz=evaluate(p,tr)[trainc];s,f=d['success'][trainc][:,tr],d['failure'][trainc][:,tr];q=s/(s+f)
        history.append(dict(step=it,held_native_VAL=metrics(z,d,va,held),motion_VAL=metrics(z,d,va,[14,15]),
            causal_motion_VAL=causal(z,d,va),TRAIN_NLL=float((q*np.logaddexp(0,-tz)+(1-q)*np.logaddexp(0,tz)).mean())))
    assert all(np.array_equal(v,traverse_util.flatten_dict(p)[k]) for k,v in frozen_parameters.items())
    for it in STEPS:
        pp=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in CONTROLS[1:]:
            predictions[f'step{it}__{condition}']=evaluate(pp,va,condition)
    write(dest/'history.json',history);write(dest/'normalization.json',norm)
    np.savez_compressed(dest/'predictions.npz',indices=va,**predictions)
    write(dest/'complete.json',dict(fold=fold,arm=arm,seed=seed,fit=trainc,held=held,
        warm25_exact_baseline_replay=True,frozen_parameters_unchanged=True,
        trainable_after25=int(sum(flat[k].size for k in flat if masks[k])),
        total_parameters=int(sum(a.size for a in flat.values())),
        data_sha256=sha(SOURCE/'dataset.npz'),rule_sha256=sha(RULE),code_sha256=sha(__file__),
        target_labels_used=False,new_rollouts=0))
    print(dict(fold=fold,arm=arm,seed=seed,done=True),flush=True)


def summarize():
    from .motion_factorial_analysis import causal_details, compare
    from .phase_learning_diagnosis import summarize as prior_diagnosis
    d=np.load(SOURCE/'dataset.npz');tr,va=[np.flatnonzero(d['split']==a) for a in ('train','validation')]
    s,f=[d[k][:,va].reshape(16,16,2) for k in ('success','failure')];q=s/(s+f)
    tq=(d['success'][:,tr]/(d['success'][:,tr]+d['failure'][:,tr])).reshape(16,64,2)
    prior=np.broadcast_to(tq.mean(1)[:,None],(16,16,2));boot=np.random.default_rng(202610042235).integers(16,size=(10000,16))
    rows,diagnostics,controls,choices=[],[],[],{}
    for arm in ('full_update',)+ARMS:
        def folder(fold,seed):
            return (SOURCE/'cv'/f'fold{fold}'/'motion_intervention'/'rest_motion'/f'seed{seed}' if arm=='full_update' else OUT/'cv'/f'fold{fold}'/arm/f'seed{seed}')
        candidates=[]
        for step in STEPS:
            rr=[]
            for fold in range(3):
                for seed in SEEDS:
                    p=folder(fold,seed);read(p/'complete.json')
                    h=next(r for r in read(p/'history.json') if r['step']==step)
                    rr.append(dict(arm=arm,fold=fold,seed=seed,step=step,**h['held_native_VAL']))
            rows+=rr
            candidates.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in SEEDS],cases_per_seed=192))
        choices[arm]=min(candidates,key=lambda a:a['NLL']);chosen=choices[arm]['step']
        for fold in range(3):
            for seed in SEEDS:
                pp=np.load(folder(fold,seed)/'predictions.npz')
                for step in sorted(set((25,75,150,300,500,700,1000,1500,chosen))):
                    z=pp[f'step{step}'].reshape(16,16,2)
                    for ev,cs in (('held_native',folds()[fold]),('motion',[14,15])):
                        diagnostics.append(dict(arm=arm,fold=fold,seed=seed,step=step,selected=step==chosen,
                            constituents_seen=fold!=2,evaluation=ev,**prior_diagnosis(z,q,s,f,prior,cs,boot),**causal_details(z,q,boot)))
                z=pp[f'step{chosen}'].reshape(16,16,2)
                for cond in CONTROLS[1:]:
                    zz=pp[f'step{chosen}__{cond}'].reshape(16,16,2)
                    for ev,cs in (('held_native',folds()[fold]),('motion',[14,15])):
                        controls.append(dict(arm=arm,fold=fold,seed=seed,step=chosen,condition=cond,
                            constituents_seen=fold!=2,evaluation=ev,**compare(z,zz,q,s,f,cs,boot)))
    csvwrite(OUT/'source_trajectories.csv',rows);csvwrite(OUT/'learning_diagnosis.csv',diagnostics);csvwrite(OUT/'selected_input_controls.csv',controls)
    write(OUT/'source_selection.json',dict(selection=choices,criterion='pooled source-held-native-controller VAL NLL',
        new_rollouts=0,target_labels_used=False,independent_confirmation=False))
    print(choices,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    train(a.index) if a.action=='train' else summarize()
