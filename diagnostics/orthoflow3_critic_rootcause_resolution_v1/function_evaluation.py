"""Frozen4vs12functions confirmation; no target tuning or candidate changes."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from .function_confirmation import OUT,SOURCE,read,write
from .support_train import model
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite

def main():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    frozen=read(SOURCE/'models_frozen.json');d=np.load(OUT/'dataset.npz');x=dict(np.load(OUT/'entities.npz'));assert d['valid'].all()
    s=d['success'].reshape(64,2);f=d['failure'].reshape(64,2);n=s+f;q=s/n;good=s>=15;bad=f>=2;eligible=good.any(1);ix=np.arange(64)
    wrong={k:np.load(OUT/f'inputs_{k}.npz') for k in ('alt','second','extra_a','extra_b')};assert all(v['valid'].all() for v in wrong.values())
    shift=np.roll(ix,-1);pairshift=np.column_stack([2*shift,2*shift+1]).ravel()
    conditions=('correct',*(f'wrong_{k}' for k in wrong),'state_shuffle','context_state_shuffle','joint_state_context_shuffle')
    rows=[];decisions=[];preds={}
    for kind in ('eta_only','entity_response_mean'):
      m=model(kind);apply=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
      for entry in [r for r in frozen['models'] if r['kind']==kind]:
        path=Path(entry['path']);assert sha(path/'best.msgpack')==entry['checkpoint_sha256'];assert sha(path/'normalization.json')==entry['normalization_sha256']
        par=serialization.msgpack_restore((path/'best.msgpack').read_bytes());norm=read(path/'normalization.json');eta=(d['eta']-norm['eta_center'])/norm['eta_scale']
        for condition in conditions:
            inp=wrong[condition[6:]] if condition.startswith('wrong_') else d
            cc=(inp['context']-norm['context_center'])/norm['context_scale'];ar=(inp['agent_response']-norm['agent_center'])/norm['agent_scale'];c=np.concatenate([cc,ar.reshape(128,-1)],-1)
            si=shift[d['state_index']] if condition in ('state_shuffle','joint_state_context_shuffle') else d['state_index']
            if condition in ('context_state_shuffle','joint_state_context_shuffle'):c=c[pairshift]
            z=np.asarray(apply(par,gather(x,si),jnp.asarray(eta,jnp.float32),jnp.asarray(c,jnp.float32))).reshape(64,2);p=expit(z);ch=z.argmax(1)
            preds[f"{entry['size']}__{kind}__{entry['seed']}__{condition}"]=z
            dc=q[:,0]-q[:,1];pc=p[:,0]-p[:,1]
            # Unknownnumericaloutcomes receive intervalbounds, not imputedQ16.
            low=s/16;high=(16-f)/16;strong=np.where(low[:,0]-high[:,1]>=.25,1,np.where(high[:,0]-low[:,1]<=-.25,-1,0))
            rows.append(dict(arm=entry['size'],kind=kind,seed=entry['seed'],condition=condition,
                NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/n.sum()),MAE=float(abs(p-q).mean()),
                B15=int(good[ix,ch].sum()),unknown=int((~good[ix,ch]&~bad[ix,ch]).sum()),oracle_B15=int(eligible.sum()),cases=64,
                selected_Q=float(q[ix,ch].mean()),regret=float((q.max(1)-q[ix,ch]).mean()),selected_predicted=float(p[ix,ch].mean()),
                severe=int(((p[ix,ch]>.9)&(high[ix,ch]<=.5)).sum()),strong_cases=int((strong!=0).sum()),
                strong_correct=int(((np.sign(pc)==strong)&(strong!=0)).sum()),
                state_contrast_correlation=float(np.corrcoef(dc,pc)[0,1]) if pc.std()>1e-8 else None))
            for state in ix:decisions.append(dict(arm=entry['size'],kind=kind,seed=entry['seed'],condition=condition,state_index=int(state),
                eta_index=(10,15)[ch[state]],Q=float(q[state,ch[state]]),predicted=float(p[state,ch[state]]),
                B15=bool(good[state,ch[state]]),non_B15=bool(bad[state,ch[state]]),eligible=bool(eligible[state])))
    # Matched non-neural diagnostic, frozen before target controller creation.
    from . import function_local as local
    loc=SOURCE/'local_diagnostic'
    for name,expected in frozen['local_diagnostic'].items():assert sha(loc/name)==expected
    ld=np.load(loc/'knn.npz');lnorm=read(loc/'normalization.json');lk=read(loc/'complete.json')['k']
    for condition in conditions:
        inp=wrong[condition[6:]] if condition.startswith('wrong_') else d
        si=shift[d['state_index']] if condition in ('state_shuffle','joint_state_context_shuffle') else d['state_index']
        ci=pairshift if condition in ('context_state_shuffle','joint_state_context_shuffle') else np.arange(128)
        features=local.features(x,si,inp['context'][ci],inp['agent_response'][ci],lnorm)
        pp=local.predict(ld['train_z'],ld['train_eta'],ld['train_q'],features,d['eta_index'],lk).reshape(64,2)
        pp=np.clip(pp,1e-6,1-1e-6);z=np.log(pp)-np.log1p(-pp);ch=z.argmax(1);preds[f'twelve_controller__local_knn__0__{condition}']=z
        dc=q[:,0]-q[:,1];pc=pp[:,0]-pp[:,1];low=s/16;high=(16-f)/16
        strong=np.where(low[:,0]-high[:,1]>=.25,1,np.where(high[:,0]-low[:,1]<=-.25,-1,0))
        rows.append(dict(arm='twelve_controller',kind='local_knn',seed=0,condition=condition,
            NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/n.sum()),MAE=float(abs(pp-q).mean()),
            B15=int(good[ix,ch].sum()),unknown=int((~good[ix,ch]&~bad[ix,ch]).sum()),oracle_B15=int(eligible.sum()),cases=64,
            selected_Q=float(q[ix,ch].mean()),regret=float((q.max(1)-q[ix,ch]).mean()),selected_predicted=float(pp[ix,ch].mean()),
            severe=int(((pp[ix,ch]>.9)&(high[ix,ch]<=.5)).sum()),strong_cases=int((strong!=0).sum()),
            strong_correct=int(((np.sign(pc)==strong)&(strong!=0)).sum()),state_contrast_correlation=float(np.corrcoef(dc,pc)[0,1]) if pc.std()>1e-8 else None))
        for state in ix:decisions.append(dict(arm='twelve_controller',kind='local_knn',seed=0,condition=condition,state_index=int(state),eta_index=(10,15)[ch[state]],
            Q=float(q[state,ch[state]]),predicted=float(pp[state,ch[state]]),B15=bool(good[state,ch[state]]),non_B15=bool(bad[state,ch[state]]),eligible=bool(eligible[state])))
    csvwrite(OUT/'metrics.csv',rows);csvwrite(OUT/'decisions.csv',decisions);np.savez_compressed(OUT/'predictions.npz',**preds)
    rng=np.random.default_rng(202610048128);boot=rng.integers(64,size=(10000,64));comparisons=[]
    for entry in frozen['models']+[dict(size='twelve_controller',kind='local_knn',seed=0)]:
        arm,kind,seed=(entry[k] for k in ('size','kind','seed'));key=f'{arm}__{kind}__{seed}';a=preds[key+'__correct'];ca=a.argmax(1)
        refs={c:preds[key+'__'+c] for c in conditions[1:]}
        refs['eta_only']=preds[f'{arm}__eta_only__{seed}__correct'] if kind!='local_knn' else np.mean([preds[f'twelve_controller__eta_only__{s}__correct'] for s in (17,23,41)],axis=0)
        if arm=='twelve_controller' and kind!='local_knn':
            refs['matched_four_controller']=preds[f'four_controller__{kind}__{seed}__correct']
            refs['reweighted_four_controller']=preds[f'reweighted_four_controller__{kind}__{seed}__correct']
        for ref,b in refs.items():
            cb=b.argmax(1);ga=good[ix,ca];gb=good[ix,cb];ba=bad[ix,ca];bb=bad[ix,cb]
            la=(s*np.logaddexp(0,-a)+f*np.logaddexp(0,a)).sum(1)/n.sum(1);lb=(s*np.logaddexp(0,-b)+f*np.logaddexp(0,b)).sum(1)/n.sum(1)
            dl=la-lb;dq=q[ix,ca]-q[ix,cb];lci=np.quantile(dl[boot].mean(1),[.025,.975]);qci=np.quantile(dq[boot].mean(1),[.025,.975])
            rescue=int((ga&bb).sum());br=int((ba&gb).sum())
            comparisons.append(dict(arm=arm,kind=kind,seed=seed,reference=ref,NLL_delta=float(dl.mean()),NLL_CI_low=float(lci[0]),NLL_CI_high=float(lci[1]),
                selected_Q_delta=float(dq.mean()),selected_Q_CI_low=float(qci[0]),selected_Q_CI_high=float(qci[1]),rescue=rescue,breaks=br,net_rescue=rescue-br,
                paired_exact_p=float(binomtest(rescue,rescue+br,.5).pvalue) if rescue+br else 1.,changed_top1=int((ca!=cb).sum()),
                mean_probability_change=float(abs(expit(a)-expit(b)).mean())))
    csvwrite(OUT/'paired_comparisons.csv',comparisons)
    write(OUT/'evaluation_audit.json',dict(families=64,held_controller_seed=read(OUT/'protocol.json')['target_controller_seed'],source_controllers=12,
        oracle_B15=int(eligible.sum()),both_candidates_B15=int(good.all(1).sum()),no_known_B15=int((~eligible).sum()),
        fixed_eta10_B15=int(good[:,0].sum()),fixed_eta15_B15=int(good[:,1].sum()),numerical=int(d['numerical'].sum()),unknown_cells=int((~good&~bad).sum()),
        models_frozen_sha256=sha(SOURCE/'models_frozen.json'),target_labels_used_for_selection=False,
        scope=read(OUT/'protocol.json')['scope'],generator_modified=False))
    print([r for r in rows if r['condition']=='correct'])

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--replicate',type=int,default=0);args=ap.parse_args()
    from . import function_confirmation as target
    target.replica(args.replicate);OUT=target.OUT
    main()
