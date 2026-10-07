"""Diagnose early-stopping unit mismatch entirely inside source controllers."""
import argparse, hashlib, os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.special import expit
from .function_models import SEEDS,KINDS
from .function_support import OUT as DATA, read, write
from .support_train import model, STEPS
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
OUT=DATA.parent/'controller_validation_unit'
RULE=DATA.parent/'controller_validation_protocol.json'


def folds():
    profiles=read(DATA/'protocol.json')['profiles']
    order=sorted(range(12),key=lambda j:hashlib.sha256(('c1_controller_unit_cv_v1\0'+profiles[j]['controller_uid']).encode()).hexdigest())
    return [order[j:j+4] for j in range(0,12,4)]


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold=index//6;kind=KINDS[(index%6)//3];seed=SEEDS[index%3]
    held=folds()[fold];fitc=[c for c in range(12) if c not in held]
    dest=OUT/f'fold{fold}'/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    d=np.load(DATA/'dataset.npz');x=dict(np.load(DATA/'frozen_model_entities.npz'))
    fit=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='validation');si=d['state_index']
    ec=d['eta'][fit].mean(0);es=np.maximum(d['eta'][fit].std(0),.1)
    cc=d['context'][fitc][:,fit].mean((0,1));cs=np.maximum(d['context'][fitc][:,fit].std((0,1)),.05)
    ac=d['agent_response'][fitc][:,fit].mean((0,1,2));ast=np.maximum(d['agent_response'][fitc][:,fit].std((0,1,2)),.01)
    eta=(d['eta']-ec)/es
    ctx=np.concatenate([(d['context']-cc)/cs,((d['agent_response']-ac)/ast).reshape(12,160,-1)],-1).astype(np.float32)
    m=model(kind);p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,88)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,c);q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=opt.update(g,o,p)
        return optax.apply_updates(p,u),o,v
    predict=jax.jit(lambda p,xx,e,c:m.apply(p,xx,e,c))
    def eval_all(p,ii,condition='correct'):
        shifted=np.roll(ii.reshape(-1,2),-1,axis=0).ravel()
        state=si[shifted] if condition in ('state_shuffle','joint_state_context_shuffle') else si[ii]
        ci=shifted if condition in ('context_state_shuffle','joint_state_context_shuffle') else ii
        return np.array([np.asarray(predict(p,gather(x,state),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[c,ci],jnp.float32))) for c in range(12)])
    def metrics(z,ii,controllers):
        s=d['success'][controllers][:,ii];f=d['failure'][controllers][:,ii];q=s/(s+f);zz=z[controllers];pr=expit(zz)
        g=(s>=15).reshape(len(controllers),-1,2);ch=zz.reshape(len(controllers),-1,2).argmax(-1)
        dt=(q.reshape(len(controllers),-1,2)[:,:,0]-q.reshape(len(controllers),-1,2)[:,:,1])
        dp=(pr.reshape(len(controllers),-1,2)[:,:,0]-pr.reshape(len(controllers),-1,2)[:,:,1])
        dt=dt-dt.mean(1,keepdims=True);dp=dp-dp.mean(1,keepdims=True)
        return dict(NLL=float((q*np.logaddexp(0,-zz)+(1-q)*np.logaddexp(0,zz)).mean()),
            MAE=float(abs(pr-q).mean()),B15=int(g[np.arange(len(controllers))[:,None],np.arange(g.shape[1])[None,:],ch].sum()),
            oracle_B15=int(g.any(-1).sum()),cases=int(g.shape[0]*g.shape[1]),
            centered_state_eta_contrast_correlation=float(np.corrcoef(dt.ravel(),dp.ravel())[0,1]) if dp.std()>1e-8 else None)
    rng=np.random.default_rng(seed);history=[];snapshots={}
    for it in range(1,1501):
        draw=rng.choice(fit,32);ii=np.tile(draw,8);ci=np.repeat(fitc,32)
        p,o,_=step(p,o,gather(x,si[ii]),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[ci,ii],jnp.float32),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in STEPS:continue
        zz=eval_all(p,val);tt=eval_all(p,fit)
        snapshots[f'step{it}']=zz
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,TRAIN=metrics(tt,fit,fitc),seen_controller_VAL=metrics(zz,val,fitc),held_controller_VAL=metrics(zz,val,held)))
    write(dest/'history.json',history)
    normalization=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist(),agent_center=ac.tolist(),agent_scale=ast.tolist(),fitting_controllers=fitc)
    write(dest/'normalization.json',normalization)
    chosen={name:min(history,key=lambda a:a[name]['NLL'])['step'] for name in ('seen_controller_VAL','held_controller_VAL')}
    for selection,it in chosen.items():
        p=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in ('correct','state_shuffle','context_state_shuffle','joint_state_context_shuffle'):
            snapshots[f'{selection}__{condition}']=eval_all(p,val,condition)
    np.savez_compressed(dest/'validation_predictions.npz',indices=val,held_controllers=np.array(held),**snapshots)
    write(dest/'complete.json',dict(fold=fold,kind=kind,seed=seed,fit_controllers=fitc,held_controllers=held,
        selected_steps=chosen,rule_sha256=sha(RULE),data_sha256=sha(DATA/'dataset.npz'),code_sha256=sha(__file__),target_labels_used=False,new_rollouts=0))
    print(dict(fold=fold,kind=kind,seed=seed,selected_steps=chosen),flush=True)


def summarize():
    rows=[]
    for fold in range(3):
      for kind in KINDS:
       for seed in SEEDS:
        p=OUT/f'fold{fold}'/kind/f'seed{seed}';done=read(p/'complete.json');hist=read(p/'history.json')
        for a in hist:
          for evaluation in ('TRAIN','seen_controller_VAL','held_controller_VAL'):
            rows.append(dict(fold=fold,kind=kind,seed=seed,step=a['step'],evaluation=evaluation,**a[evaluation]))
    csvwrite(OUT/'trajectories.csv',rows)
    pooled=[]
    for kind in KINDS:
      for step in STEPS:
       for evaluation in ('seen_controller_VAL','held_controller_VAL'):
        rr=[r for r in rows if r['kind']==kind and r['step']==step and r['evaluation']==evaluation]
        pooled.append(dict(kind=kind,step=step,evaluation=evaluation,NLL=float(np.mean([r['NLL'] for r in rr])),
            B15_mean_over_seeds=float(sum(r['B15'] for r in rr)/3),cases_per_seed=sum(r['cases'] for r in rr)//3))
    csvwrite(OUT/'pooled_source_cv.csv',pooled)
    selection={}
    for kind in KINDS:
      selection[kind]={evaluation:min([r for r in pooled if r['kind']==kind and r['evaluation']==evaluation],key=lambda a:a['NLL']) for evaluation in ('seen_controller_VAL','held_controller_VAL')}
    write(OUT/'source_selection.json',dict(folds=folds(),selection=selection,source_only=True,target_labels_used=False,new_rollouts=0,
        warning='Source CV selects models/hyperparameters; these selected estimates are not final independent confirmation.'))
    print(selection)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='train':train(a.index)
    else:summarize()
