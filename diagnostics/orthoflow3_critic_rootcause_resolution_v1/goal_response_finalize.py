"""Source input-use gate and fixed-CV-step final fits; no target labels."""
import argparse,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from .goal_response_cv import OUT,DATA,RULE,KINDS,SEEDS,folds,load,model,metrics,read,write,sha,csvwrite


def controls():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    selection=read(OUT/'source_selection.json')['selection'];rows=[]
    conditions=('correct','goal_wrong_state','goal_wrong_controller','initial_context_wrong_state','all_context_wrong_state','state_shuffle')
    for fold in range(3):
        held=folds()[fold];fitc=[c for c in range(12) if c not in held];d,x,fit,val,norm=load(fitc);ctx=d['input_context'];si=d['state_index'];eta=d['normalized_eta']
        shift=np.roll(val.reshape(-1,2),-1,axis=0).ravel()
        for kind in KINDS:
            m=model(kind);predict=jax.jit(lambda p,xx,e,c:m.apply(p,xx,e,c));it=selection[kind]['held_controller_VAL']['step']
            for seed in SEEDS:
                path=OUT/'cv'/f'fold{fold}'/kind/f'seed{seed}';p=serialization.msgpack_restore((path/f'step{it}.msgpack').read_bytes())
                for condition in conditions:
                    z=np.zeros((12,len(val)),np.float32)
                    for c in held:
                        cx=ctx[c,val].copy()
                        if condition=='goal_wrong_state':cx[:,88:]=ctx[c,shift,88:]
                        if condition=='goal_wrong_controller':cx[:,88:]=ctx[(c+1)%12,val,88:]
                        if condition=='initial_context_wrong_state':cx[:,:88]=ctx[c,shift,:88]
                        if condition=='all_context_wrong_state':cx=ctx[c,shift]
                        state=si[shift] if condition=='state_shuffle' else si[val]
                        z[c]=np.asarray(predict(p,gather(x,state),jnp.asarray(eta[val],jnp.float32),jnp.asarray(cx,jnp.float32)))
                    rows.append(dict(fold=fold,kind=kind,seed=seed,step=it,condition=condition,**metrics(z,d,val,held)))
    csvwrite(OUT/'pooled_step_input_controls.csv',rows)
    pooled=[]
    for kind in KINDS:
      for condition in conditions:
       for seed in SEEDS:
        rr=[r for r in rows if r['kind']==kind and r['condition']==condition and r['seed']==seed]
        pooled.append(dict(kind=kind,condition=condition,seed=seed,NLL=float(np.mean([r['NLL'] for r in rr])),B15=sum(r['B15'] for r in rr),cases=192))
    csvwrite(OUT/'input_use_summary.csv',pooled)
    def nn(kind,condition='correct'):return np.mean([r['NLL'] for r in pooled if r['kind']==kind and r['condition']==condition])
    def bb(kind):return np.array([r['B15'] for r in pooled if r['kind']==kind and r['condition']=='correct'])
    gate=bool(nn('H20_goal')<min(nn('H20_only'),nn('eta_only')) and np.all(bb('H20_goal')>bb('eta_only')) and nn('H20_goal','goal_wrong_controller')>nn('H20_goal'))
    write(OUT/'source_gate.json',dict(passed=gate,full_NLL=float(nn('H20_goal')),H20_NLL=float(nn('H20_only')),eta_NLL=float(nn('eta_only')),
        wrong_goal_controller_NLL=float(nn('H20_goal','goal_wrong_controller')),wrong_goal_state_NLL=float(nn('H20_goal','goal_wrong_state')),
        B15_full=bb('H20_goal').tolist(),B15_eta=bb('eta_only').tolist(),cases=192,
        criterion='Source-held-controller NLL improves over H20 and eta, all3seeds B15 improve versuseta, correctgoalcontext NLL betterthanwrongcontrollergoalcontext. Eligibility for independentconfirmation only.',target_labels_used=False))
    print(read(OUT/'source_gate.json'));print(pooled)


def final_train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert read(OUT/'source_gate.json')['passed']
    kind=KINDS[index//3];seed=SEEDS[index%3];dest=OUT/'final_models'/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    dest.mkdir(parents=True,exist_ok=True);selection=read(OUT/'source_selection.json')['selection']
    steps=selection[kind]['held_controller_VAL']['step'];early=selection['H20_goal']['held_controller_VAL']['step']
    d,x,fit,val,norm=load(list(range(12)));si=d['state_index'];eta=d['normalized_eta'];ctx=d['input_context'];m=model(kind)
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,104)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,c);q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=opt.update(g,o,p)
        return optax.apply_updates(p,u),o,v
    rng=np.random.default_rng(seed)
    for it in range(1,steps+1):
        draw=rng.choice(fit,32);ii=np.tile(draw,12);ci=np.repeat(np.arange(12),32)
        p,o,_=step(p,o,gather(x,si[ii]),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[ci,ii],jnp.float32),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if kind=='H20_only' and it==early:(dest/'matched_early.msgpack').write_bytes(serialization.to_bytes(p))
    (dest/'fixed_step.msgpack').write_bytes(serialization.to_bytes(p));write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(kind=kind,seed=seed,steps=steps,step_chosen_only_by_source_controller_CV=True,
        dataset_sha256=sha(DATA/'dataset.npz'),protocol_sha256=sha(RULE),code_sha256=sha(__file__),target_labels_used=False))
    print(dict(kind=kind,seed=seed,steps=steps))


def freeze():
    assert read(OUT/'source_gate.json')['passed'] and (OUT/'independent_confirmation_protocol.json').exists()
    if (OUT/'models_frozen.json').exists():raise FileExistsError('Alreadyfrozen')
    entries=[]
    for kind in KINDS:
      for seed in SEEDS:
        p=OUT/'final_models'/kind/f'seed{seed}';done=read(p/'complete.json')
        entries.append(dict(variant=kind,kind=kind,seed=seed,path=str(p),checkpoint='fixed_step.msgpack',checkpoint_sha256=sha(p/'fixed_step.msgpack'),normalization_sha256=sha(p/'normalization.json'),steps=done['steps']))
        if kind=='H20_only':entries.append(dict(variant='H20_only_matched_early',kind=kind,seed=seed,path=str(p),checkpoint='matched_early.msgpack',checkpoint_sha256=sha(p/'matched_early.msgpack'),normalization_sha256=sha(p/'normalization.json'),steps=read(OUT/'source_selection.json')['selection']['H20_goal']['held_controller_VAL']['step']))
    for name in ('protocol.json','states.json','pairs.json'):write(OUT/name,read(DATA/name))
    write(OUT/'models_frozen.json',dict(models=entries,rule_sha256=sha(RULE),data_sha256=sha(DATA/'dataset.npz'),
        independent_confirmation_protocol_sha256=sha(OUT/'independent_confirmation_protocol.json'),target_not_created=True,target_labels_used=False,
        primary='H20_goal versus sourceCV-selected H20_only and eta-only;3seeds',secondary='H20-onlyatthesame75stepsasgoalmodel isolates earlierstopping from newinput'))
    print(dict(frozen=len(entries)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('controls','train','freeze'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='controls':controls()
    elif a.action=='train':final_train(a.index)
    else:freeze()
