"""Matched native-vs-crossed supervision; same architecture, order and NLL."""
import argparse,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.special import expit
from .phase_factorial_support import OUT, SOURCE, RULE, read, write, sha
from .goal_response_cv import model,folds,KINDS,SEEDS,csvwrite
from .support_train import STEPS
ARMS=('native_repeated','crossed_phase')


def load(fitc):
    d=dict(np.load(OUT/'dataset.npz'));x=dict(np.load(SOURCE/'frozen_model_entities.npz'));tr=np.flatnonzero(d['split']=='train');va=np.flatnonzero(d['split']=='validation')
    ec=d['eta'][tr].mean(0);es=np.maximum(d['eta'][tr].std(0),.1)
    cc=d['context'][fitc][:,tr].mean((0,1));cs=np.maximum(d['context'][fitc][:,tr].std((0,1)),.05)
    ac=d['agent_response'][fitc][:,tr].mean((0,1,2));ast=np.maximum(d['agent_response'][fitc][:,tr].std((0,1,2)),.01)
    gc=d['goal_response'][fitc][:,tr].mean((0,1));gs=np.maximum(d['goal_response'][fitc][:,tr].std((0,1)),.05)
    norm=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist(),agent_center=ac.tolist(),agent_scale=ast.tolist(),goal_center=gc.tolist(),goal_scale=gs.tolist(),fitting_native_controllers=fitc,normalization_intervention_data_used=False)
    d['normalized_eta']=(d['eta']-ec)/es;d['input_context']=np.concatenate([(d['context']-cc)/cs,((d['agent_response']-ac)/ast).reshape(14,160,-1),(d['goal_response']-gc)/gs],-1).astype(np.float32)
    return d,x,tr,va,norm


def metrics(z,d,ii,controllers):
    s=d['success'][controllers][:,ii];f=d['failure'][controllers][:,ii];q=s/(s+f);zz=z[controllers];p=expit(zz)
    g=(s>=15).reshape(len(controllers),-1,2);ch=zz.reshape(len(controllers),-1,2).argmax(-1);tq=q.reshape(len(controllers),-1,2);pp=p.reshape(len(controllers),-1,2)
    dt=tq[:,:,0]-tq[:,:,1];dp=pp[:,:,0]-pp[:,:,1];dt-=dt.mean(1,keepdims=True);dp-=dp.mean(1,keepdims=True)
    return dict(NLL=float((q*np.logaddexp(0,-zz)+(1-q)*np.logaddexp(0,zz)).mean()),MAE=float(abs(p-q).mean()),
        B15=int(g[np.arange(len(controllers))[:,None],np.arange(g.shape[1])[None,:],ch].sum()),oracle_B15=int(g.any(-1).sum()),cases=int(g.shape[0]*g.shape[1]),
        centered_contrast_correlation=float(np.corrcoef(dt.ravel(),dp.ravel())[0,1]) if dp.std()>1e-8 else None)


def causal(z,d,ii):
    q=d['success'][:,ii]/(d['success'][:,ii]+d['failure'][:,ii]);p=expit(z)
    dq=np.stack([q[12]-q[0],q[13]-q[1]]);dp=np.stack([p[12]-p[0],p[13]-p[1]])
    strong=abs(dq)>=.25
    # Causal eta-ranking interaction: difference-of-differences, per physicalfamily.
    tc=dq.reshape(2,-1,2);pc=dp.reshape(2,-1,2);td=tc[:,:,0]-tc[:,:,1];pd=pc[:,:,0]-pc[:,:,1]
    return dict(delta_MAE=float(abs(dq-dp).mean()),delta_correlation=float(np.corrcoef(dq.ravel(),dp.ravel())[0,1]) if dp.std()>1e-8 else None,
        mean_predicted_abs_delta=float(abs(dp).mean()),mean_true_abs_delta=float(abs(dq).mean()),
        strong_cells=int(strong.sum()),strong_sign_correct=int(((np.sign(dp)==np.sign(dq))&strong).sum()),
        interaction_delta_MAE=float(abs(td-pd).mean()),interaction_correlation=float(np.corrcoef(td.ravel(),pd.ravel())[0,1]) if pd.std()>1e-8 else None)


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold=index//18;arm=ARMS[(index%18)//9];kind=KINDS[(index%9)//3];seed=SEEDS[index%3]
    held=folds()[fold];fitc=[c for c in range(12) if c not in held];have_phase=0 in fitc and 1 in fitc
    trainc=fitc+(([0,1] if arm=='native_repeated' else [12,13]) if have_phase else [])
    dest=OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    dest.mkdir(parents=True,exist_ok=True);d,x,tr,va,norm=load(fitc);si=d['state_index'];e=d['normalized_eta'];c=d['input_context'];m=model(kind)
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,104)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,c);q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=opt.update(g,o,p);return optax.apply_updates(p,u),o,v
    predict=jax.jit(lambda p,xx,e,c:m.apply(p,xx,e,c))
    # Fold2 excludes both intervention constituents. Its TRAIN tensors,
    # optimizer and draw order are identical to the already-complete native
    # goal-response CV, so reuse those checkpoints rather than retraining.
    reuse_path=None
    if not have_phase:
        from .goal_response_cv import OUT as previous,load as previous_load
        od,ox,ot,ov,on=previous_load(fitc)
        assert np.array_equal(tr,ot) and np.array_equal(va,ov)
        assert all(np.array_equal(x[k],ox[k]) for k in x)
        for key in ('success','failure','context','agent_response','input_context'):
            assert np.array_equal(d[key][:12],od[key])
        assert np.array_equal(d['normalized_eta'],od['normalized_eta'])
        for key in ('eta_center','eta_scale','context_center','context_scale','agent_center','agent_scale','goal_center','goal_scale'):
            assert np.array_equal(norm[key],on[key])
        reuse_path=previous/'cv'/f'fold{fold}'/kind/f'seed{seed}';read(reuse_path/'complete.json')
    def eval_all(p,condition='correct'):
        scores=[];shift=np.roll(va.reshape(-1,2),-1,axis=0).ravel()
        for ci in range(14):
            cc=c[ci,va].copy()
            if condition=='goal_wrong_state':cc[:,88:]=c[ci,shift,88:]
            if condition=='goal_parent_replacement' and ci in (12,13):cc[:,88:]=c[ci-12,va,88:]
            if condition=='whole_context_wrong_state':cc=c[ci,shift]
            ix=si[shift] if condition=='joint_state_context_wrong_state' else si[va]
            if condition=='joint_state_context_wrong_state':cc=c[ci,shift]
            scores.append(np.asarray(predict(p,gather(x,ix),jnp.asarray(e[va],jnp.float32),jnp.asarray(cc,jnp.float32))))
        return np.array(scores)
    rng=np.random.default_rng(seed);history=[];snapshots={}
    for it in range(1,1501):
        if reuse_path is None:
            draw=rng.choice(tr,32);ii=np.tile(draw,len(trainc));ci=np.repeat(trainc,32)
            p,o,_=step(p,o,gather(x,si[ii]),jnp.asarray(e[ii],jnp.float32),jnp.asarray(c[ci,ii],jnp.float32),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in STEPS:continue
        if reuse_path is not None:p=serialization.msgpack_restore((reuse_path/f'step{it}.msgpack').read_bytes())
        zz=eval_all(p);snapshots[f'step{it}']=zz;(dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,held_native_VAL=metrics(zz,d,va,held),seen_native_VAL=metrics(zz,d,va,fitc),
            phase_VAL=metrics(zz,d,va,[12,13]),causal_phase_VAL=causal(zz,d,va)))
    write(dest/'history.json',history);write(dest/'normalization.json',norm);np.savez_compressed(dest/'validation_predictions.npz',indices=va,**snapshots)
    write(dest/'complete.json',dict(fold=fold,arm=arm,kind=kind,seed=seed,held_native_controllers=held,training_conditions=trainc,phase_labels_used=arm=='crossed_phase' and have_phase,
        target_labels_used=False,data_sha256=sha(OUT/'dataset.npz'),rule_sha256=sha(RULE),code_sha256=sha(__file__),
        exact_training_checkpoint_reuse=str(reuse_path) if reuse_path is not None else None))
    print(dict(fold=fold,arm=arm,kind=kind,seed=seed,done=True),flush=True)


def summarize():
    rows=[];selection={};causalrows=[]
    for arm in ARMS:
      selection[arm]={}
      for kind in KINDS:
        candidates=[]
        for step in STEPS:
            rr=[]
            for fold in range(3):
              for seed in SEEDS:
                path=OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}';read(path/'complete.json');h=next(r for r in read(path/'history.json') if r['step']==step)
                r=dict(arm=arm,kind=kind,step=step,fold=fold,seed=seed,**h['held_native_VAL']);rows.append(r);rr.append(r)
            candidates.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==seed) for seed in SEEDS],cases_per_seed=192))
        best=min(candidates,key=lambda a:a['NLL']);selection[arm][kind]=best
        for fold in range(3):
          for seed in SEEDS:
            path=OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}';h=next(r for r in read(path/'history.json') if r['step']==best['step'])
            causalrows.append(dict(arm=arm,kind=kind,step=best['step'],fold=fold,seed=seed,phase_constituents_seen=fold!=2,**h['causal_phase_VAL']))
    csvwrite(OUT/'source_cv_trajectories.csv',rows);csvwrite(OUT/'source_causal_predictions.csv',causalrows)
    write(OUT/'source_selection.json',dict(selection=selection,criterion='pooledheldsourcecontrollerNLL',target_labels_used=False,
        caution='Only2of3folds contain phaseconstituentsinTRAIN; fold2 is matchedno-intervention control and mustnot be called interventiontrained. SourceCV is selection, not independentconfirmation.'))
    print(selection)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='train':train(a.index)
    else:summarize()
