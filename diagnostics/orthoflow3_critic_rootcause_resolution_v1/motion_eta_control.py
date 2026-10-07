"""Strong eta-only controls on the exact motion-factorial TRAIN evidence.

The continuous neural baseline keeps parameter shapes, seeds, order and NLL.
An analytic TRAIN-only two-eta MLE is also retained; it is a diagnostic for the
two observed probes, not a claim about continuous/unseen eta generalization.
"""
import argparse
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.special import logit
from .motion_factorial_train import OUT as SOURCE,load,folds,SEEDS,STEPS,metrics,read,write,sha,csvwrite
OUT=SOURCE.parent/'motion_eta_control'


def model():
    import flax.linen as nn
    import jax.numpy as jnp
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    class EtaOnly(nn.Module):
        @nn.compact
        def __call__(self,x,eta,context):
            a=jnp.zeros((len(eta),x['agents'].shape[1],16))
            xx={**x,'agents':jnp.concatenate([x['agents'],a],-1)}
            return Critic(False,False,False,name='core')(xx,eta,jnp.zeros((len(eta),56)),jnp.zeros((len(eta),3)))
    return EtaOnly()


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    fold,seed=index//3,SEEDS[index%3];dest=OUT/'cv'/f'fold{fold}'/f'seed{seed}'
    if (dest/'complete.json').exists():return
    held=folds()[fold];fitc=[i for i in range(12) if i not in held]
    trainc=fitc+([12,13,14,15] if 0 in fitc and 1 in fitc else [])
    d,x,tr,va,norm=load(fitc);si,e=d['state_index'],d['normalized_eta'];m=model()
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    @jax.jit
    def step(p,o,xx,e,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,jnp.zeros((len(e),120)));q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=opt.update(g,o,p)
        return optax.apply_updates(p,u),o,v
    predict=jax.jit(lambda pp,xx,ee:m.apply(pp,xx,ee,jnp.zeros((len(ee),120))))
    rng=np.random.default_rng(seed);history=[];predictions={};dest.mkdir(parents=True,exist_ok=True)
    for it in range(1,1501):
        draw=rng.choice(tr,32);ii=np.tile(draw,len(trainc));ci=np.repeat(trainc,32)
        p,o,_=step(p,o,gather(x,si[ii]),jnp.asarray(e[ii],jnp.float32),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in STEPS:continue
        zz=np.asarray(predict(p,gather(x,si[va]),jnp.asarray(e[va],jnp.float32)))
        assert np.array_equal(zz.reshape(-1,2),np.broadcast_to(zz[:2],(16,2)))
        z=np.broadcast_to(zz,(16,len(va)));predictions[f'step{it}']=z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        history.append(dict(step=it,held_native_VAL=metrics(z,d,va,held),motion_VAL=metrics(z,d,va,[14,15])))
    write(dest/'normalization.json',norm);write(dest/'history.json',history)
    np.savez_compressed(dest/'predictions.npz',indices=va,**predictions)
    write(dest/'complete.json',dict(fold=fold,seed=seed,fit=trainc,held=held,data_sha256=sha(SOURCE/'dataset.npz'),
        code_sha256=sha(__file__),parameter_count=int(sum(a.size for a in jax.tree.leaves(p))),target_labels_used=False,new_rollouts=0))
    print(dict(fold=fold,seed=seed,done=True),flush=True)


def summarize():
    from .motion_factorial_analysis import compare
    from .motion_freeze_train import OUT as FREEZE
    candidates=[];rows=[]
    for step in STEPS:
        rr=[]
        for fold in range(3):
            for seed in SEEDS:
                path=OUT/'cv'/f'fold{fold}'/f'seed{seed}';read(path/'complete.json')
                h=next(r for r in read(path/'history.json') if r['step']==step)
                rr.append(dict(fold=fold,seed=seed,step=step,**h['held_native_VAL']))
        rows+=rr;candidates.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),
            B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in SEEDS],cases_per_seed=192))
    selected=min(candidates,key=lambda a:a['NLL']);choice=read(FREEZE/'source_selection.json')['selection']
    comparisons=[];pooled={};analytic=[];boot=np.random.default_rng(202610042300).integers(16,size=(10000,16))
    for fold in range(3):
        held=folds()[fold];fitc=[i for i in range(12) if i not in held]
        trainc=fitc+([12,13,14,15] if 0 in fitc and 1 in fitc else [])
        d,x,tr,va,norm=load(fitc);s,f=[d[k][:,va].reshape(16,16,2) for k in ('success','failure')];q=s/(s+f)
        qt=d['success'][trainc][:,tr]/(d['success'][trainc][:,tr]+d['failure'][trainc][:,tr])
        prob=qt.reshape(len(trainc),64,2).mean((0,1));az=np.broadcast_to(logit(prob),(16,16,2))
        analytic.append(dict(fold=fold,TRAIN_probability_by_eta=prob.tolist(),**metrics(az.reshape(16,32),d,va,held)))
        for seed in SEEDS:
            eta=np.load(OUT/'cv'/f'fold{fold}'/f'seed{seed}'/'predictions.npz')[f"step{selected['step']}"].reshape(16,16,2)
            for arm in choice:
                folder=(SOURCE/'cv'/f'fold{fold}'/'motion_intervention'/'rest_motion'/f'seed{seed}' if arm=='full_update' else FREEZE/'cv'/f'fold{fold}'/arm/f'seed{seed}')
                z=np.load(folder/'predictions.npz')[f"step{choice[arm]['step']}"].reshape(16,16,2)
                for ref,zz in (('matched_neural_eta_only',eta),('analytic_TRAIN_MLE_eta_only',az)):
                    for ev,cs in (('held_native',held),('motion',[14,15])):
                        comparisons.append(dict(arm=arm,fold=fold,seed=seed,reference=ref,evaluation=ev,
                            constituents_seen=fold!=2,**compare(z,zz,q,s,f,cs,boot)))
                    pooled.setdefault((arm,seed,ref),[]).append((z[held],zz[held],q[held],s[held],f[held]))
    combined=[]
    for (arm,seed,ref),values in pooled.items():
        z,zz,q,s,f=[np.concatenate([v[i] for v in values],0) for i in range(5)]
        combined.append(dict(arm=arm,seed=seed,reference=ref,**compare(z,zz,q,s,f,list(range(12)),boot)))
    csvwrite(OUT/'source_trajectories.csv',rows);csvwrite(OUT/'paired_comparisons.csv',comparisons);csvwrite(OUT/'pooled_family_bootstrap_comparisons.csv',combined)
    write(OUT/'source_selection.json',dict(neural=selected,analytic=analytic,source_only=True,
        matched_TRAIN_evidence=True,target_labels_used=False,new_rollouts=0,
        family_bootstrap_note='Same16families shared acrosscontrollers andfolds; resample families jointly. These are source-selection diagnostics, not independent confirmation.'))
    print(selected,flush=True);print(combined,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    train(a.index) if a.action=='train' else summarize()
