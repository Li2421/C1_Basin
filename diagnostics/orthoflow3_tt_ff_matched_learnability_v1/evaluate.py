"""One frozen paired confirmation; seed variability != independent sample size."""
from __future__ import annotations
import collections,csv,itertools,json
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest,spearmanr
from .design import ROOT,SCENES,read,write,freeze,sha
from .dataset import CHAINS
from .train import KINDS,SEEDS


def save_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def moments(values):
    v=np.array([x for x in values if x is not None and np.isfinite(x)],float)
    return (float(v.mean()),float(v.std(ddof=1)) if len(v)>1 else 0.) if len(v) else (None,None)


def interval(x,seed=120051):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return [None,None]
    r=np.random.default_rng(seed).integers(len(x),size=(20000,len(x)))
    return np.quantile(x[r].mean(1),[.025,.975]).tolist()


def wilson(s,n):
    n=np.asarray(n,float);p=s/np.maximum(n,1);z=1.959963984540054
    den=1+z*z/np.maximum(n,1);c=(p+z*z/(2*np.maximum(n,1)))/den
    d=z*np.sqrt(p*(1-p)/np.maximum(n,1)+z*z/(4*np.maximum(n,1)**2))/den
    return np.where(n>0,c-d,0.),np.where(n>0,c+d,1.)


def point_metrics(z,s,f,u,common_s,common_f):
    prob=expit(z);q=np.divide(s,s+f,out=np.full_like(s,np.nan,dtype=float),where=(s+f)>0)
    robust=s>=15;negative=f>=2;unknown=~(robust|negative)
    selected=np.argmax(z,1);ii=np.arange(len(z));hit=robust[ii,selected];bad=negative[ii,selected];un=unknown[ii,selected]
    oracle=robust.any(1);coverage_fail=negative.all(1);oracle_unknown=~(oracle|coverage_fail)
    nr=(common_s+common_f).sum();nll=float((common_s*np.logaddexp(0,-z)+common_f*np.logaddexp(0,z)).sum()/max(nr,1))
    qc=np.divide(common_s,common_s+common_f,out=np.full_like(s,np.nan,dtype=float),where=(common_s+common_f)>0);valid=np.isfinite(qc)
    top=np.argsort(-z,axis=1,kind='stable');regret=np.nanmax(q,1)-q[ii,selected]
    denom=max(1,int(oracle.sum()));within=[spearmanr(q[h],prob[h]).statistic for h in ii if np.isfinite(q[h]).all() and np.ptp(q[h])>0]
    r=dict(states=len(z),B15=int(hit.sum()),nonB15=int(bad.sum()),unresolved=int(un.sum()),oracle_B15=int(oracle.sum()),
        proposal_failure=int(coverage_fail.sum()),oracle_unknown=int(oracle_unknown.sum()),selection_failure=int((oracle&bad).sum()),
        selection_unresolved=int((oracle&un).sum()),oracle_gap_lower=int((oracle&bad).sum()),oracle_gap_upper=int((oracle&~hit).sum()),
        available_B15_selection=float(hit[oracle].sum()/denom),NLL=nll,MAE=float(np.mean(abs(qc[valid]-prob[valid]))),
        Brier=float(np.mean((qc[valid]-prob[valid])**2)),Spearman=finite(spearmanr(qc[valid],prob[valid]).statistic),
        selected_Q_observed=float(np.nanmean(q[ii,selected])),selected_Q16_lower=float((s[ii,selected]/16).mean()),
        selected_Q16_upper=float(((s[ii,selected]+u[ii,selected])/16).mean()),oracle_Q=float(np.nanmean(np.nanmax(q,1))),
        Q_regret=float(np.nanmean(regret)),severe_FP=int(((prob[ii,selected]>.9)&(f[ii,selected]>=8)).sum()),
        selected_p=float(prob[ii,selected].mean()),top2_B15=float(robust[ii[:,None],top[:,:2]].any(1)[oracle].sum()/denom),
        top3_B15=float(robust[ii[:,None],top[:,:3]].any(1)[oracle].sum()/denom),
        within_state_Spearman=finite(np.nanmean(within)) if within else None,all_B15_underdiscriminative=int(robust.all(1).sum()))
    return r,dict(hit=hit,bad=bad,unknown=un,selected=selected,oracle=oracle,q=q,prob=prob,robust=robust)


def reversals(s,f,z):
    n=s+f;q=np.divide(s,n,out=np.full_like(s,np.nan,dtype=float),where=n>0);lo,hi=wilson(s,n)
    cases=[];strong=[];covered=set();sc=[];qc=[]
    for a,b in itertools.combinations(range(16),2):
        diff=q[:,a]-q[:,b];up=np.flatnonzero(diff>=.25);down=np.flatnonzero(diff<=-.25)
        if not len(up) or not len(down):continue
        idx=np.r_[up,down];pred=z[idx,a]-z[idx,b];correct=np.where(pred>=0,1.,-1.)==np.sign(diff[idx]);balanced=.5*(correct[:len(up)].mean()+correct[len(up):].mean())
        cases.append(balanced);covered.update(idx.tolist());sc.extend((z[idx,a]-z[idx,b]).tolist());qc.extend(diff[idx].tolist())
        aa=np.flatnonzero((diff>=.25)&(lo[:,a]>hi[:,b]));bb=np.flatnonzero((diff<=-.25)&(hi[:,a]<lo[:,b]))
        if len(aa) and len(bb):strong.append(.5*((z[aa,a]>=z[aa,b]).mean()+(z[bb,a]<z[bb,b]).mean()))
    return dict(reversal_eta_pairs=len(cases),reversal_states=len(covered),reversal_balanced_accuracy=float(np.mean(cases)) if cases else None,
        stable_Wilson_reversal_eta_pairs=len(strong),stable_reversal_accuracy=float(np.mean(strong)) if strong else None,
        reversal_Q_difference_Spearman=float(spearmanr(qc,sc).statistic) if len(set(qc))>1 and len(set(sc))>1 else None)


def main():
    assert not (ROOT/'final_decision.json').exists(),'Frozen evaluation already produced'
    freeze0=read(ROOT/'test_predictions_frozen.json');assert freeze0['predictions_sha256']==sha(ROOT/'frozen_predictions.npz')
    pred=np.load(ROOT/'frozen_predictions.npz');truth=np.load(ROOT/'test_truth.npz');p=read(ROOT/'protocol.json');o=truth['outcomes'];indices=truth['indices']
    s=((o[...,0]==1)&(o[...,1]==0)).sum(-1).astype(float);u=o[...,1].sum(-1).astype(float);f=16-u-s
    valid=(o[0,...,1]==0)&(o[1,...,1]==0)
    cs=(o[...,0]*valid[None]).sum(-1).astype(float);cf=valid.sum(-1)[None]-cs
    metrics=[];individual=[];decisions={};support=[];reversal_rows=[]
    conditions=('correct','state_shuffle','context_shuffle','state_context_shuffle','eta_shuffle')
    for scene in SCENES:
        ix=np.flatnonzero([p['states'][i]['scenario']==scene for i in indices]);stateids=indices[ix]
        for ci,chain in enumerate(CHAINS):
            robust=s[ci,ix]>=15;negative=f[ci,ix]>=2;eligible=robust.any(1);q=s[ci,ix]/np.maximum(16-u[ci,ix],1)
            best_const=robust.sum(0).max();globalvar=np.var(q.mean(0));totalvar=np.var(q)
            gp=f'{scene}__{chain}__global_train_eta__0__test';assert np.array_equal(pred[gp+'__indices'],stateids)
            gm,ga=point_metrics(pred[gp+'__correct'],s[ci,ix],f[ci,ix],u[ci,ix],cs[ci,ix],cf[ci,ix])
            metrics.append(dict(scene=scene,chain=chain,kind='global_train_eta',seed=0,condition='correct',**gm))
            support.append(dict(scene=scene,chain=chain,states=len(ix),oracle_B15=int(eligible.sum()),
                hindsight_best_fixed_B15=int(best_const),oracle_minus_best_fixed=int(eligible.sum()-best_const),
                TRAIN_selected_fixed_B15=gm['B15'],TRAIN_selected_fixed_eta_index=int(ga['selected'][0]),
                globally_successful_eta_count=int(robust.all(0).sum()),median_B15_candidates=float(np.median(robust.sum(1))),
                eta_main_effect_fraction=float(globalvar/totalvar) if totalvar else None,
                state_main_effect_fraction=float(np.var(q.mean(1))/totalvar) if totalvar else None,
                interaction_variance_fraction=float(np.var(q-q.mean(0)[None]-q.mean(1)[:,None]+q.mean())/totalvar) if totalvar else None,
                numerical_unknown=int(u[ci,ix].sum()),collision=int(o[ci,ix,...,2].sum())))
            for kind,seed,condition in itertools.product(KINDS,SEEDS,conditions):
                prefix=f'{scene}__{chain}__{kind}__{seed}__test'
                assert np.array_equal(pred[prefix+'__indices'],stateids)
                z=pred[prefix+'__'+condition];m,a=point_metrics(z,s[ci,ix],f[ci,ix],u[ci,ix],cs[ci,ix],cf[ci,ix]);m.update(scene=scene,chain=chain,kind=kind,seed=seed,condition=condition)
                metrics.append(m);decisions[scene,chain,kind,seed,condition]=a
                rev=reversals(s[ci,ix],f[ci,ix],z);reversal_rows.append(dict(scene=scene,chain=chain,kind=kind,seed=seed,condition=condition,**rev))
                if condition=='correct':
                    for k,stateindex in enumerate(stateids):
                        sel=a['selected'][k];individual.append(dict(scene=scene,chain=chain,kind=kind,seed=seed,state_uid=p['states'][stateindex]['state_uid'],eta_index=int(sel),
                            success=int(s[ci,ix[k],sel]),observed_fail=int(f[ci,ix[k],sel]),numerical=int(u[ci,ix[k],sel]),
                            predicted_Q=float(a['prob'][k,sel]),B15=bool(a['hit'][k]),nonB15=bool(a['bad'][k]),unknown=bool(a['unknown'][k]),oracle_B15=bool(a['oracle'][k])))
    comparisons=[];contrasts=[];table=[]
    for scene in SCENES:
        ix=np.flatnonzero([p['states'][i]['scenario']==scene for i in indices]);n=len(ix);deltas={};bounds={}
        eligible=(s[:,ix]>=15).any(-1).all(0)&(f[:,ix]>=2).any(-1).all(0)
        for chain in CHAINS:
            diff=[];low=[];high=[]
            for seed in SEEDS:
                a=decisions[scene,chain,'full_context',seed,'correct'];b=decisions[scene,chain,'eta_only',seed,'correct']
                rescue=int((a['hit']&b['bad']).sum());brk=int((a['bad']&b['hit']).sum())
                comparisons.append(dict(scene=scene,chain=chain,seed=seed,rescue=rescue,break_count=brk,
                    unresolved_pairs=int((a['unknown']|b['unknown']).sum()),paired_binomial_p=float(binomtest(rescue,rescue+brk).pvalue) if rescue+brk else 1.))
                dd=a['hit'].astype(float)-b['hit'];diff.append(dd)
                low.append(a['hit'].astype(float)-(b['hit']|b['unknown']))
                high.append((a['hit']|a['unknown']).astype(float)-b['hit'])
            deltas[chain]=np.mean(diff,0);bounds[chain]=(np.mean(low,0),np.mean(high,0))
            for subset,mask in (('all',np.ones(n,bool)),('common_oracle_eligible_discriminative',eligible)):
                contrasts.append(dict(scene=scene,contrast='Delta_'+chain,subset=subset,states=int(mask.sum()),
                    estimate=float(deltas[chain][mask].mean()) if mask.any() else None,
                    family_bootstrap95=interval(deltas[chain][mask]),
                    numerical_lower=float(bounds[chain][0][mask].mean()) if mask.any() else None,
                    numerical_upper=float(bounds[chain][1][mask].mean()) if mask.any() else None))
            for kind in KINDS:
                mm=[m for m in metrics if (m['scene'],m['chain'],m['kind'],m['condition'])==(scene,chain,kind,'correct')]
                row=dict(scene=scene,chain=chain,kind=kind,states=n,B15_by_seed=[m['B15'] for m in mm],
                    unresolved_by_seed=[m['unresolved'] for m in mm],oracle=mm[0]['oracle_B15'],proposal_failure=mm[0]['proposal_failure'],
                    selection_failure_by_seed=[m['selection_failure'] for m in mm])
                for key in ('B15','NLL','MAE','Spearman','selected_Q_observed','Q_regret','severe_FP','top3_B15'):
                    row[key+'_mean'],row[key+'_std']=moments([m[key] for m in mm])
                table.append(row)
        for subset,mask in (('all',np.ones(n,bool)),('common_oracle_eligible_discriminative',eligible)):
            d=deltas['FF']-deltas['TT']
            contrasts.append(dict(scene=scene,contrast='Delta_FF_minus_Delta_TT',subset=subset,states=int(mask.sum()),
                estimate=float(d[mask].mean()) if mask.any() else None,family_bootstrap95=interval(d[mask]),
                numerical_lower=float((bounds['FF'][0]-bounds['TT'][1])[mask].mean()) if mask.any() else None,
                numerical_upper=float((bounds['FF'][1]-bounds['TT'][0])[mask].mean()) if mask.any() else None))
    save_csv(ROOT/'matched_metrics.csv',metrics);save_csv(ROOT/'paired_comparisons.csv',comparisons);save_csv(ROOT/'matched_table.csv',table)
    save_csv(ROOT/'per_state_decisions.csv',individual);save_csv(ROOT/'ranking_reversals.csv',reversal_rows);save_csv(ROOT/'data_structure_comparison.csv',support)
    write(ROOT/'matched_contrasts.json',contrasts)
    # Conservative preregistered adjudication: a numerical win alone is not
    # evidence that ranking is more learnable or that eta-only had headroom.
    ff_adv=[];tt_adv=[];did_adv=[];under=[];state_controls=[]
    for scene in SCENES:
        at=lambda c:next(r for r in contrasts if r['scene']==scene and r['contrast']==c and r['subset']=='all')
        def positive(c):
            r=at(c);return r['family_bootstrap95'][0] is not None and r['family_bootstrap95'][0]>0 and r['numerical_lower']>0
        ff_adv.append(positive('Delta_FF'));tt_adv.append(positive('Delta_TT'));did_adv.append(positive('Delta_FF_minus_Delta_TT'))
        ro=next(r for r in support if r['scene']==scene and r['chain']=='FF');under.append(ro['oracle_minus_best_fixed']<=1)
        correct=[m['B15'] for m in metrics if (m['scene'],m['chain'],m['kind'],m['condition'])==(scene,'FF','full_context','correct')]
        shuffled=[m['B15'] for m in metrics if (m['scene'],m['chain'],m['kind'],m['condition'])==(scene,'FF','full_context','state_context_shuffle')]
        state_controls.append(float(np.mean(correct))>float(np.mean(shuffled)) and float(np.mean(correct))>ro['TRAIN_selected_fixed_B15'])
    if any(a and b and not c and d for a,b,c,d in zip(ff_adv,did_adv,under,state_controls)):
        verdict='FIELD_LEARNABILITY_ADVANTAGE'
    elif any(tt_adv) and any(ff_adv):verdict='OLD_PROBLEM_ALSO_SOLVED'
    elif all(not u for u in under) and all(abs(r['estimate'])<=.05 and r['family_bootstrap95'][0]>=-.1 and r['family_bootstrap95'][1]<=.1 for r in contrasts if r['contrast']=='Delta_FF_minus_Delta_TT' and r['subset']=='all'):
        verdict='NO_CLEAR_DIFFERENCE'
    else:verdict='INCONCLUSIVE'
    freeze(ROOT/'final_decision.json',dict(classification=verdict,table=table,contrasts=contrasts,data_structure=support,
        no_paradigm_specific_tuning=True,models_frozen_before_test=True,same_model_initialization_and_pair_order=True,
        generator_modified=False,eta_or_lifting_modified=False,scope='Same fixed-scene held-out families on a shared continuous K16 panel, NOT unseen-eta/LOSO',
        source_ceilings_not_learnability_evidence=True,numerical_outcomes_not_imputed=True,
        family_bootstrap_not_seed_pseudoreplication=True,confirmation_sha256=sha(ROOT/'test_truth.npz')))
    print(json.dumps(dict(classification=verdict,table=table,contrasts=contrasts),allow_nan=False),flush=True)


if __name__=='__main__':main()
