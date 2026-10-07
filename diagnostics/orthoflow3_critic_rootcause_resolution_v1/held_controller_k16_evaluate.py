"""Freeze predictions before labels; no selection using confirmation outcomes."""
import argparse, collections, os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import numpy as np
from scipy.special import expit
from .held_controller_k16_runtime import OUT,ROOT,DATA,read,write,sha,target,batchdir,SEEDS
from .db_transfer_evaluate import csvwrite
CONDITIONS=('correct','wrong_controller','state_shuffle','joint_state_context_shuffle','eta_tensor_shuffle')


def predict(index):
    import jax,jax.numpy as jnp
    from flax import serialization
    from .db_transfer_train import model
    from . import raw_bypass_confirmation as legacy
    dest=target(index);assert not (dest/'prediction_freeze.json').exists()
    frozen=read(OUT/'models_frozen.json');norm=read(frozen['normalization_path'])
    assert sha(frozen['normalization_path'])==frozen['normalization_sha256']
    for path,digest in frozen['dependencies'].items():assert sha(path)==digest
    pairs=read(dest/'pairs.json');states=read(dest/'states.json');n=len(states);K=16
    si=np.array([r['state_index'] for r in pairs]);eta=np.array([r['eta'] for r in pairs],np.float32)
    assert np.array_equal(si,np.repeat(np.arange(n),K))
    assert np.array_equal(eta.reshape(n,K,3),np.broadcast_to(eta[:K],(n,K,3)))
    entities=dict(np.load(dest/'entities.npz'));raw=dict(np.load(dest/'context.npz'))
    ee=((eta-norm['eta_center'])/norm['eta_scale']).astype(np.float32)
    contexts={k:((v-norm['context_center'])/norm['context_scale']).astype(np.float32) for k,v in raw.items()}
    for key in contexts:
        for a,b,flag in ((0,40,73),(40,56,74),(56,72,75)):contexts[key][:,a:b]*=raw[key][:,flag,None]>0
    shift=np.roll(np.arange(n),-1);pshift=(K*shift[:,None]+np.arange(K)).ravel();eshift=(K*np.arange(n)[:,None]+np.roll(np.arange(K),-1)).ravel()
    functions={};scores={};checks=[]
    for entry in frozen['models']:
        path=Path(entry['path'])/'checkpoint.msgpack';assert sha(path)==entry['checkpoint_sha256']
        par=serialization.msgpack_restore(path.read_bytes());kind=entry['kind']
        if kind not in functions:
            m=model(kind);functions[kind]=jax.jit(lambda p,x,e,c,m=m:m.apply(p,x,e,c))
        fn=functions[kind]
        for cond in CONDITIONS:
            ids=shift[si] if cond in ('state_shuffle','joint_state_context_shuffle') else si
            cc=contexts['wrong_controller' if cond=='wrong_controller' else 'correct']
            if cond=='joint_state_context_shuffle':cc=cc[pshift]
            e=ee[eshift] if cond=='eta_tensor_shuffle' else ee
            z=np.asarray(fn(par,{k:v[ids] for k,v in entities.items()},jnp.asarray(e),jnp.asarray(cc))).reshape(n,K)
            assert np.isfinite(z).all();scores[f'{entry["label"]}_{kind}__{entry["seed"]}__{cond}']=z
        if kind=='eta_only':
            z=scores[f'{entry["label"]}_{kind}__{entry["seed"]}__correct'];assert np.allclose(z,z[:1],atol=1e-6)
            checks.append(dict(arm=entry['label']+'_'+kind,seed=entry['seed'],state_independent=True))
    for entry in frozen['legacy_models']:
        par,ln=legacy.ev.load_entry(entry);e=(eta-ln['eta_center'])/ln['eta_scale']
        cc={}
        for key,a in raw.items():
            # Legacy model uses the same mean-broadcast response. Tiling the
            # physical mean is algebraically equivalent before its affine norm.
            values=dict(context=a[:,:24],agent_response=np.broadcast_to(a[:,None,24:40],(len(a),4,16)),goal_response=a[:,40:56],goal_motion_response=a[:,56:72])
            cc[key]=legacy.ev.normalized_context(values,values,ln)
        for cond in CONDITIONS:
            ids=shift[si] if cond in ('state_shuffle','joint_state_context_shuffle') else si
            c=cc['wrong_controller' if cond=='wrong_controller' else 'correct']
            if cond=='joint_state_context_shuffle':c=c[pshift]
            z=legacy.forward(par,{k:v[ids] for k,v in entities.items()},e[eshift] if cond=='eta_tensor_shuffle' else e,c,entry['variant']).reshape(n,K)
            assert np.isfinite(z).all();scores[f'legacy_{entry["variant"]}__{entry["seed"]}__{cond}']=z
    np.savez_compressed(dest/'predictions.npz',**scores)
    names=('protocol.json','pairs.json','states.json','physical.json','entities.npz','context.npz')
    write(dest/'prediction_freeze.json',dict(models_frozen_sha256=sha(OUT/'models_frozen.json'),protocol_sha256=sha(OUT/'protocol.json'),predictions_sha256=sha(dest/'predictions.npz'),runtime_sha256=sha(ROOT/'held_controller_k16_runtime.py'),evaluator_sha256=sha(__file__),input_sha256={name:sha(dest/name) for name in names},checks=checks,target_labels_read=False,new_rollouts=0,conditions=CONDITIONS))
    print(dict(target=SEEDS[index],models=len(scores)//len(CONDITIONS),states=n,candidates=K,predictions_frozen=True,new_rollouts=0),flush=True)


def truth(index,batches):
    from shared_rollout_db.src.rollout_db import connect,canonical
    dest=target(index);guard=read(dest/'prediction_freeze.json')
    assert guard['evaluator_sha256']==sha(__file__) and guard['predictions_sha256']==sha(dest/'predictions.npz')
    assert guard['models_frozen_sha256']==sha(OUT/'models_frozen.json')
    for name,digest in guard['input_sha256'].items():assert sha(dest/name)==digest
    for b in range(batches):
        audit=read(batchdir(index,b)/'alignment_audit.json');assert audit['missing']==audit['conflict']==0
        assert (batchdir(index,b)/'cache_postflight.json').exists()
    pairs=read(dest/'pairs.json')[:16*batches*16];cid=read(dest/'protocol.json')['profiles'][0]['controller_uid']
    arrays={k:np.zeros((16*batches,16),int) for k in ('success','failure','numerical','collision')};uids=[]
    with connect(True) as db:
        for i,p in enumerate(pairs):
            records={r['seed_key']:r for r in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',(p['state_uid'],p['eta_uid'],cid))}
            for seed in range(16):
                r=records[canonical({'future_index':seed})]
                assert r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined']
                key='numerical' if r['numerical_failure'] else 'success' if r['success'] else 'failure'
                arrays[key].flat[i]+=1;arrays['collision'].flat[i]+=r['collision'];uids.append(r['rollout_uid'])
    assert np.all(arrays['success']+arrays['failure']+arrays['numerical']==16)
    np.savez_compressed(dest/f'truth_stage{batches}.npz',**arrays)
    write(dest/f'truth_audit_stage{batches}.json',dict(seed_records=len(uids),rollout_uid_sha256=__import__('hashlib').sha256(canonical(uids).encode()).hexdigest(),numerical=int(arrays['numerical'].sum()),collision=int(arrays['collision'].sum()),target_labels_opened_only_after_prediction_freeze=True,all_labels_from_global_DB=True))
    return arrays


def eligibility():
    # This gate deliberately does not load any critic prediction array.
    rows=[]
    for index in (0,1):
        a=truth(index,1);good=a['success']>=15;bad=a['failure']>=2
        rows.append(dict(controller=SEEDS[index],states=16,oracle_B15=int(good.any(1).sum()),hindsight_best_global_eta_B15=int(good.sum(0).max()),mixed_states=int((good.any(1)&bad.any(1)).sum()),coverage_failures=int(bad.all(1).sum()),unresolved_oracle=int((~good.any(1)&~bad.all(1)).sum())))
    headroom=sum(r['oracle_B15']-r['hindsight_best_global_eta_B15'] for r in rows);mixed=sum(r['mixed_states'] for r in rows)
    result=dict(controllers=rows,oracle_minus_best_constant=headroom,mixed_states=mixed,extend_second_stage=bool(headroom>=4 and mixed>=8),rule_sha256=sha(OUT/'protocol.json'),model_predictions_used=False)
    write(OUT/'stage1_eligibility.json',result);print(result,flush=True)


def selected(z,a,subset):
    s,f,num=[a[k][:,subset] for k in ('success','failure','numerical')];zz=z[:,subset];ii=zz.argmax(1);rr=np.arange(len(ii))
    return dict(success=s[rr,ii],failure=f[rr,ii],numerical=num[rr,ii],p=expit(zz[rr,ii]),eta=np.asarray(subset)[ii])


def measures(z,a,subset):
    from scipy.stats import spearmanr
    s,f,num=[a[k][:,subset] for k in ('success','failure','numerical')];zz=z[:,subset];p=expit(zz);valid=s+f;q=s/np.maximum(valid,1)
    good=s>=15;bad=f>=2;eligible=good.any(1);chosen=selected(z,a,subset);yes=chosen['success']>=15;no=chosen['failure']>=2
    ii=zz.argmax(1);rr=np.arange(len(ii));nll=s*np.logaddexp(0,-zz)+f*np.logaddexp(0,zz)
    rank=np.argsort(-zz,axis=1);top=lambda k:bool(0) if len(subset)<k else float(np.take_along_axis(good,rank[:,:k],1).any(1)[eligible].mean()) if eligible.any() else None
    oracle=q.max(1);sel=q[rr,ii];high=(chosen['p']>.9)&(chosen['numerical']==0)&(chosen['success']<=8)
    reversal=[];reversal_states=set()
    clear=s+num<=8
    for i in range(len(subset)):
        for j in range(i):
            pos=good[:,i]&clear[:,j];neg=good[:,j]&clear[:,i]
            if not pos.any() or not neg.any():continue
            dz=zz[:,i]-zz[:,j]
            accuracy=.5*np.mean((dz[pos]>0)+.5*(dz[pos]==0))+.5*np.mean((dz[neg]<0)+.5*(dz[neg]==0))
            reversal.append(float(accuracy));reversal_states.update(np.flatnonzero(pos|neg).tolist())
    return dict(states=len(ii),eligible=int(eligible.sum()),B15=int(yes.sum()),nonB15=int(no.sum()),unresolved=int((~yes&~no).sum()),oracle_B15=int(eligible.sum()),oracle_gap=int(eligible.sum()-yes.sum()),eligible_selection_rate=float(yes[eligible].mean()) if eligible.any() else None,all_B15_underdiscriminative=int(good.all(1).sum()),coverage_failure=int(bad.all(1).sum()),oracle_unknown=int((~eligible&~bad.all(1)).sum()),NLL_observed_trials=float(nll.sum()/valid.sum()),MAE_observed_Q=float(abs(p-q)[valid>0].mean()),Spearman=float(spearmanr(p[valid>0],q[valid>0]).statistic),selected_Q_observed=float(sel.mean()),selected_Q16_lower=float((chosen['success']/16).mean()),selected_Q16_upper=float(((chosen['success']+chosen['numerical'])/16).mean()),oracle_Q_observed=float(oracle.mean()),regret_observed=float((oracle-sel).mean()),severe_FP=int(high.sum()),mean_selected_prediction=float(chosen['p'].mean()),top2_B15=top(2),top3_B15=top(3),oracle_top1_agreement=float(np.isclose(sel,oracle).mean()),strong_reversal_eta_pairs=len(reversal),strong_reversal_states=len(reversal_states),strong_reversal_balanced_accuracy=float(np.mean(reversal)) if reversal else None)


def paired(z,reference,a,subset):
    from scipy.stats import binomtest
    aa,bb=selected(z,a,subset),selected(reference,a,subset)
    ga,gb=aa['success']>=15,bb['success']>=15;ka=ga|(aa['failure']>=2);kb=gb|(bb['failure']>=2);known=ka&kb
    rescue=int((known&ga&~gb).sum());brk=int((known&~ga&gb).sum())
    delta=ga[known].astype(float)-gb[known].astype(float)
    boot=np.random.default_rng(2026100563).choice(delta,(5000,len(delta))).mean(1) if len(delta) else np.array([np.nan])
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(rescue=rescue,breaks=brk,net=rescue-brk,unresolved_pairs=int((~known).sum()),paired_rate_CI_low=float(lo),paired_rate_CI_high=float(hi),paired_exact_p=float(binomtest(min(rescue,brk),rescue+brk,.5).pvalue) if rescue+brk else 1.)


def evaluate(batches):
    rows=[];comparisons=[];panel=read(OUT/'candidate_panel.json')
    subsets={'K16':list(range(16)),'seen_eta_8':[r['index'] for r in panel if r['seen']],'unseen_eta_8':[r['index'] for r in panel if not r['seen']]}
    common=next(r['index'] for r in panel if r['source']=='source_TRAIN_strong_common_eta')
    for index in (0,1):
        a=truth(index,batches);n=16*batches;scores={k:v[:n] for k,v in np.load(target(index)/'predictions.npz').items()}
        for subset,ids in subsets.items():
            for key,z in scores.items():
                arm,seed,condition=key.split('__');rows.append(dict(controller=SEEDS[index],arm=arm,seed=seed,condition=condition,subset=subset,**measures(z,a,ids)))
                if condition!='correct' or arm not in ('controller_cv_full_context','family_cv_full_context','legacy_raw_bypass'):continue
                references={'controller_cv_eta':scores[f'controller_cv_eta_only__{seed}__correct'],'family_cv_eta':scores[f'family_cv_eta_only__{seed}__correct'],'wrong_controller':scores[f'{arm}__{seed}__wrong_controller'],'state_shuffle':scores[f'{arm}__{seed}__state_shuffle'],'joint_state_context_shuffle':scores[f'{arm}__{seed}__joint_state_context_shuffle']}
                for name,rz in references.items():comparisons.append(dict(controller=SEEDS[index],arm=arm,seed=seed,subset=subset,reference=name,**paired(z,rz,a,ids)))
        z=np.full((n,16),-100.);z[:,common]=100.
        mm=measures(z,a,subsets['K16'])
        for key in ('NLL_observed_trials','MAE_observed_Q','Spearman','mean_selected_prediction','severe_FP','top2_B15','top3_B15','strong_reversal_balanced_accuracy'):mm[key]=None
        rows.append(dict(controller=SEEDS[index],arm='fixed_source_common_eta',seed=0,condition='correct',subset='K16',**mm))
    csvwrite(OUT/f'metrics_stage{batches}.csv',rows);csvwrite(OUT/f'paired_stage{batches}.csv',comparisons)
    write(OUT/f'evaluation_audit_stage{batches}.json',dict(batches=batches,states=32*batches,models_refit_after_target_labels=False,generator_changed=False,scope='Held-controller/family, mixed seen+unseen eta; not cross-scene',numerical='observed-Q and Q16 bounds separately; unresolved robust labels never imputed',confirmation_protocol_sha256=sha(OUT/'protocol.json')))
    print([r for r in rows if r['arm'] in ('controller_cv_full_context','controller_cv_eta_only','family_cv_eta_only','fixed_source_common_eta') and r['condition']=='correct' and r['subset']=='K16'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('predict','eligibility','evaluate'));p.add_argument('--index',type=int,default=0);p.add_argument('--batches',type=int,default=1);a=p.parse_args()
    predict(a.index) if a.action=='predict' else evaluate(a.batches) if a.action=='evaluate' else eligibility()
