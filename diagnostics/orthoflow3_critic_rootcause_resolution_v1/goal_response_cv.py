"""Same-capacity H20 versus goal-neighborhood response, source controller CV."""
import argparse,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.special import expit
from .function_support import OUT as DATA,read,write
from .goal_response import OUT,RULE
from .controller_validation import folds
from .support_train import STEPS
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
KINDS=('eta_only','H20_only','H20_goal');SEEDS=(17,23,41)


def model(kind):
    import flax.linen as nn
    import jax.numpy as jnp
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    class GoalModel(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            a=context[:,24:88].reshape(-1,x['agents'].shape[1],16)
            a=jnp.broadcast_to(a.mean(1,keepdims=True),a.shape)
            if kind=='eta_only':a=jnp.zeros_like(a)
            xx={**x,'agents':jnp.concatenate([x['agents'],a],-1)}
            g=context[:,88:104] if kind=='H20_goal' else jnp.zeros_like(context[:,88:104])
            c=jnp.concatenate([context[:,:24],g],-1)
            return Critic(kind!='eta_only',kind!='eta_only',False,name='core')(xx,eta,c,jnp.zeros((len(eta),3)))
    return GoalModel()


def load(fitc):
    d=dict(np.load(DATA/'dataset.npz'));x=dict(np.load(DATA/'frozen_model_entities.npz'))
    fit=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='validation')
    profiles=read(DATA/'protocol.json')['profiles'];gg=[]
    for p in profiles:
        a=np.load(OUT/f'inputs_{p["name"]}.npz');assert a['valid'].all();gg.append(a['goal_response'])
    gg=np.array(gg)
    ec=d['eta'][fit].mean(0);es=np.maximum(d['eta'][fit].std(0),.1)
    cc=d['context'][fitc][:,fit].mean((0,1));cs=np.maximum(d['context'][fitc][:,fit].std((0,1)),.05)
    ac=d['agent_response'][fitc][:,fit].mean((0,1,2));ast=np.maximum(d['agent_response'][fitc][:,fit].std((0,1,2)),.01)
    gc=gg[fitc][:,fit].mean((0,1));gs=np.maximum(gg[fitc][:,fit].std((0,1)),.05)
    norm=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist(),agent_center=ac.tolist(),agent_scale=ast.tolist(),goal_center=gc.tolist(),goal_scale=gs.tolist(),fitting_controllers=fitc)
    d['normalized_eta']=(d['eta']-ec)/es
    d['input_context']=np.concatenate([(d['context']-cc)/cs,((d['agent_response']-ac)/ast).reshape(12,160,-1),(gg-gc)/gs],-1).astype(np.float32)
    return d,x,fit,val,norm


def metrics(z,d,ii,controllers):
    s=d['success'][controllers][:,ii];f=d['failure'][controllers][:,ii];q=s/(s+f);zz=z[controllers];p=expit(zz)
    g=(s>=15).reshape(len(controllers),-1,2);ch=zz.reshape(len(controllers),-1,2).argmax(-1)
    tq=q.reshape(len(controllers),-1,2);pp=p.reshape(len(controllers),-1,2);dt=tq[:,:,0]-tq[:,:,1];dp=pp[:,:,0]-pp[:,:,1]
    dt=dt-dt.mean(1,keepdims=True);dp=dp-dp.mean(1,keepdims=True)
    return dict(NLL=float((q*np.logaddexp(0,-zz)+(1-q)*np.logaddexp(0,zz)).mean()),MAE=float(abs(p-q).mean()),
        B15=int(g[np.arange(len(controllers))[:,None],np.arange(g.shape[1])[None,:],ch].sum()),oracle_B15=int(g.any(-1).sum()),cases=int(g.shape[0]*g.shape[1]),
        centered_state_eta_contrast_correlation=float(np.corrcoef(dt.ravel(),dp.ravel())[0,1]) if dp.std()>1e-8 else None)


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold=index//9;kind=KINDS[(index%9)//3];seed=SEEDS[index%3]
    held=folds()[fold];fitc=[c for c in range(12) if c not in held]
    dest=OUT/'cv'/f'fold{fold}'/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    dest.mkdir(parents=True,exist_ok=True);d,x,fit,val,norm=load(fitc);si=d['state_index'];eta=d['normalized_eta'];ctx=d['input_context']
    m=model(kind);p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,104)))
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
        shifted=np.roll(ii.reshape(-1,2),-1,axis=0).ravel();scores=[]
        for c in range(12):
            cx=ctx[c,ii].copy()
            if condition=='goal_wrong_state':cx[:,88:]=ctx[c,shifted,88:]
            if condition=='goal_wrong_controller':cx[:,88:]=ctx[(c+1)%12,ii,88:]
            if condition=='initial_context_wrong_state':cx[:,:88]=ctx[c,shifted,:88]
            states=si[shifted] if condition=='state_shuffle' else si[ii]
            scores.append(np.asarray(predict(p,gather(x,states),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(cx,jnp.float32))))
        return np.array(scores)
    rng=np.random.default_rng(seed);history=[];snapshots={}
    for it in range(1,1501):
        draw=rng.choice(fit,32);ii=np.tile(draw,8);ci=np.repeat(fitc,32)
        p,o,_=step(p,o,gather(x,si[ii]),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[ci,ii],jnp.float32),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in STEPS:continue
        zz=eval_all(p,val);tt=eval_all(p,fit);snapshots[f'step{it}']=zz
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,TRAIN=metrics(tt,d,fit,fitc),seen_controller_VAL=metrics(zz,d,val,fitc),held_controller_VAL=metrics(zz,d,val,held)))
    write(dest/'history.json',history);write(dest/'normalization.json',norm)
    chosen={name:min(history,key=lambda a:a[name]['NLL'])['step'] for name in ('seen_controller_VAL','held_controller_VAL')}
    # Per-fold controls are diagnosis only; pooled source-CV step is chosen later.
    for selection,it in chosen.items():
        p=serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in ('correct','goal_wrong_state','goal_wrong_controller','initial_context_wrong_state','state_shuffle'):
            snapshots[f'{selection}__{condition}']=eval_all(p,val,condition)
    np.savez_compressed(dest/'validation_predictions.npz',indices=val,held_controllers=np.array(held),**snapshots)
    write(dest/'complete.json',dict(fold=fold,kind=kind,seed=seed,fit_controllers=fitc,held_controllers=held,selected_steps=chosen,
        rule_sha256=sha(RULE),data_sha256=sha(DATA/'dataset.npz'),code_sha256=sha(__file__),target_labels_used=False,new_rollouts=0))
    print(dict(fold=fold,kind=kind,seed=seed,selected_steps=chosen),flush=True)


def summarize():
    rows=[]
    for fold in range(3):
      for kind in KINDS:
       for seed in SEEDS:
        p=OUT/'cv'/f'fold{fold}'/kind/f'seed{seed}';read(p/'complete.json')
        for a in read(p/'history.json'):
          for evaluation in ('TRAIN','seen_controller_VAL','held_controller_VAL'):
            rows.append(dict(fold=fold,kind=kind,seed=seed,step=a['step'],evaluation=evaluation,**a[evaluation]))
    csvwrite(OUT/'cv_trajectories.csv',rows);pooled=[]
    for kind in KINDS:
      for step in STEPS:
       for evaluation in ('seen_controller_VAL','held_controller_VAL'):
        rr=[r for r in rows if r['kind']==kind and r['step']==step and r['evaluation']==evaluation]
        pooled.append(dict(kind=kind,step=step,evaluation=evaluation,NLL=float(np.mean([r['NLL'] for r in rr])),
            B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==seed) for seed in SEEDS],cases_per_seed=sum(r['cases'] for r in rr)//3))
    csvwrite(OUT/'pooled_source_cv.csv',pooled)
    selection={kind:{evaluation:min([r for r in pooled if r['kind']==kind and r['evaluation']==evaluation],key=lambda a:a['NLL']) for evaluation in ('seen_controller_VAL','held_controller_VAL')} for kind in KINDS}
    write(OUT/'source_selection.json',dict(folds=folds(),selection=selection,source_only=True,target_labels_used=False,new_rollouts=0,
        warning='Selected sourceCV results are not independent confirmation. Need input-use controls and held-function selection advantage before any promotion.'))
    print(selection)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='train':train(a.index)
    else:summarize()
