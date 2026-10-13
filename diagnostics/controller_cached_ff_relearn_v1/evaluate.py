"""Development falsification on cached independent states; never final confirmation."""
from __future__ import annotations
import csv
import numpy as np
from scipy.special import expit
from .study import HERE, read, write, sha
from .learn import metrics, freeze_models


def correlate(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    good=np.isfinite(a)&np.isfinite(b);a=a[good];b=b[good]
    # float32 batched inference can differ by ~1e-8 for identical inputs.
    # Do not turn numerical roundoff into a reported controller correlation.
    if len(a)<3 or np.std(a)<1e-7 or np.std(b)<1e-7:return None
    return float(np.corrcoef(a,b)[0,1])


def center_eta(a):return a-a.mean(-1,keepdims=True)
def center_both(a):return a-a.mean(-1,keepdims=True)-a.mean(0,keepdims=True)+a.mean()


def run():
    freeze_models()
    p=read(HERE/'protocol.json');frozen=read(HERE/'models_frozen.json')
    if (HERE/'root_cause_scope_amendment.json').exists():
        p['models']=read(HERE/'root_cause_scope_amendment.json')['models']
    labels=np.load(HERE/'labels.npz');dev=np.flatnonzero(labels['split']=='development')
    success=labels['success'][dev];failure=labels['failure'][dev];unknown=labels['unknown'][dev]
    truth=success/np.maximum(success+failure,1);lower=success/16;upper=(success+unknown)/16
    n=len(dev);assert n==16
    rng=np.random.default_rng(2026100921);boot=rng.integers(n,size=(20000,n))
    permutations=[]
    while len(permutations)<100:
        perm=rng.permutation(n)
        if np.all(perm!=np.arange(n)):permutations.append(perm)
    permutations=np.stack(permutations)
    rows=np.arange(n)[:,None];controllers=np.arange(2)[None,:]
    def select_value(a,sel):return a[rows,controllers,sel]
    def interval(values):return [float(x) for x in np.quantile(values,[.025,.975])]
    def paired(a,b):
        difference=np.asarray(a,float)-np.asarray(b,float)
        if difference.ndim>1:difference=difference.mean(tuple(range(1,difference.ndim)))
        samples=difference[boot].mean(1)
        return dict(mean_difference=float(difference.mean()),ci95=interval(samples),states=n)
    qdelta=truth[:,1]-truth[:,0]
    lower_delta=lower[:,1]-upper[:,0];upper_delta=upper[:,1]-lower[:,0]
    strong_delta=(lower_delta>=.375)|(upper_delta<=-.375)
    a,b=np.triu_indices(16,1)
    raw=labels['outcome'][dev]
    disc=raw[...,:8];evalu=raw[...,8:]
    dl=(disc==1).mean(-1);dh=((disc==1)|(disc==-1)).mean(-1)
    ev=(evalu==1).sum(-1)/np.maximum((evalu>=0).sum(-1),1)
    direction=np.where(dl[...,a]-dh[...,b]>=.375,1,
        np.where(dl[...,b]-dh[...,a]>=.375,-1,0))
    reversal=(direction[:,0]*direction[:,1])==-1
    evdiff=ev[...,a]-ev[...,b]
    eval_valid=((evalu[:,:,a,:]>=0).all(-1)&(evalu[:,:,b,:]>=0).all(-1))
    eval_confirmed=reversal&(evdiff[:,0]*evdiff[:,1]<0)&eval_valid.all(1)
    reversal_rows,reversal_pairs=np.where(reversal)
    # Evaluation best-fixed policies are explicitly posthoc falsification bounds.
    fixed_mean=lower.mean((0,1));global_best=int(fixed_mean.argmax())
    per_controller_best=lower.mean(0).argmax(-1)
    fixed_global=np.full((n,2),global_best);fixed_by_c=np.broadcast_to(per_controller_best,(n,2))
    predictions={};seed_metrics={};summary={};bootstrap_arrays={}
    for kind in p['models']:
        logits=[];seed_metrics[kind]=[]
        for seed in p['fit_seeds']:
            z=np.load(HERE/'fits'/kind/f'seed{seed}'/'predictions.npz')['logits'][dev]
            logits.append(z);q=expit(z);delta=q[:,1]-q[:,0]
            seed_metrics[kind].append(dict(seed=seed,**metrics(z,success,failure),
                controller_delta_correlation=correlate(delta,qdelta),
                eta_centered_controller_delta_correlation=correlate(center_eta(delta),center_eta(qdelta))))
        q=expit(np.stack(logits)).mean(0);predictions[kind]=q
        delta=q[:,1]-q[:,0];choice=q.argmax(-1)
        selected=select_value(lower,choice);selected_upper=select_value(upper,choice)
        b15=select_value(success>=15,choice);b15_upper=select_value(success+unknown>=15,choice)
        wrong=q[:,::-1].argmax(-1);wrong_selected=select_value(lower,wrong)
        mismatch_choices=q[permutations].argmax(-1)
        mismatch_values=np.stack([select_value(lower,x) for x in mismatch_choices])
        mismatch_mean=mismatch_values.mean(0)
        mismatch_b15=np.stack([select_value(success>=15,x) for x in mismatch_choices]).mean(0)
        ranking=q[...,a]-q[...,b]
        follows=(np.sign(ranking[:,0])==direction[:,0])&(np.sign(ranking[:,1])==direction[:,1])
        num=(follows&reversal).sum(-1);den=reversal.sum(-1)
        numerator=num[boot].sum(-1);denominator=den[boot].sum(-1)
        switch_boot=numerator[denominator>0]/denominator[denominator>0]
        # Independently confirmed cases are secondary, never redefine the discovery grouping.
        follows_eval=(np.sign(ranking[:,0])==np.sign(evdiff[:,0]))&(np.sign(ranking[:,1])==np.sign(evdiff[:,1]))
        counts=np.stack([np.bincount(choice[:,c],minlength=16) for c in range(2)])
        prob=counts/n
        entropy=-np.sum(np.where(prob>0,prob*np.log2(np.maximum(prob,1e-30)),0),-1)
        matched_boot=selected.mean(-1)[boot].mean(-1)
        global_fixed_boot=lower[boot].mean((1,2)).max(-1)
        per_c_fixed_boot=lower[boot].mean(1).max(-1).mean(-1)
        conservative_fixed=upper[boot].mean((1,2)).max(-1)
        bootstrap_arrays[kind+'_minus_reselected_fixed']=matched_boot-global_fixed_boot
        bootstrap_arrays[kind+'_minus_reselected_controller_fixed']=matched_boot-per_c_fixed_boot
        bootstrap_arrays[kind+'_reversal_switch_accuracy']=switch_boot
        summary[kind]=dict(
            selected_success_lower=float(selected.mean()),selected_success_upper=float(selected_upper.mean()),
            selected_b15_lower_count=int(b15.sum()),selected_b15_upper_count=int(b15_upper.sum()),state_controller_count=n*2,
            per_controller_success=selected.mean(0).tolist(),per_controller_b15=b15.sum(0).tolist(),
            oracle_regret_lower=float((lower.max(-1)-selected).mean()),
            wrong_controller_success=float(wrong_selected.mean()),
            matched_minus_wrong_controller=paired(selected,wrong_selected),
            mean_mismatched_state_success=float(mismatch_mean.mean()),
            matched_minus_mismatched_state=paired(selected,mismatch_mean),
            b15_matched_minus_mismatched_state=paired(b15,mismatch_b15),
            global_posthoc_best_fixed_eta=global_best,per_controller_posthoc_best_fixed_eta=per_controller_best.tolist(),
            minus_global_posthoc_fixed=paired(selected,select_value(lower,fixed_global)),
            minus_controller_posthoc_fixed=paired(selected,select_value(lower,fixed_by_c)),
            minus_bootstrap_reselected_fixed_ci95=interval(matched_boot-global_fixed_boot),
            minus_bootstrap_reselected_controller_fixed_ci95=interval(matched_boot-per_c_fixed_boot),
            censored_adaptive_lower_minus_best_fixed_upper_ci95=interval(matched_boot-conservative_fixed),
            controller_delta_correlation=correlate(delta,qdelta),
            eta_centered_controller_delta_correlation=correlate(center_eta(delta),center_eta(qdelta)),
            state_eta_centered_controller_delta_correlation=correlate(center_both(delta),center_both(qdelta)),
            strong_controller_delta_direction=float(np.mean(np.sign(delta[strong_delta])==np.sign(qdelta[strong_delta]))) if strong_delta.any() else None,
            reversal_discovery_cases=int(reversal.sum()),reversal_discovery_states=int(reversal.any(-1).sum()),
            reversal_two_sided_accuracy=float(num.sum()/den.sum()) if den.sum() else None,
            reversal_two_sided_ci95=interval(switch_boot) if len(switch_boot) else None,
            independent_eval_confirmed_reversals=int(eval_confirmed.sum()),
            independent_eval_two_sided_accuracy=float(follows_eval[eval_confirmed].mean()) if eval_confirmed.any() else None,
            selection_counts=counts.tolist(),selection_entropy_bits=entropy.tolist(),
            model_controller_choice_switch_rate=float(np.mean(choice[:,0]!=choice[:,1])))
        np.savez_compressed(HERE/f'development_predictions_{kind}.npz',
            matched=q,wrong_controller=q[:,::-1],mismatched_state=q[permutations],
            choice=choice,mismatched_choice=mismatch_choices,
            per_seed=expit(np.stack(logits)),state_indices=dev,permutations=permutations)
    eta_sel=predictions['eta_only'].argmax(-1)
    for kind,q in predictions.items():
        sel=q.argmax(-1)
        summary[kind]['minus_eta_only']=paired(select_value(lower,sel),select_value(lower,eta_sel))
        summary[kind]['b15_minus_eta_only']=paired(select_value(success>=15,sel),select_value(success>=15,eta_sel))
    # Information-alias negative controls: descriptors are identical for the two programs.
    for kind in ('eta_only','state_only','current_field'):
        np.testing.assert_allclose(predictions[kind][:,0],predictions[kind][:,1],rtol=0,atol=1e-7)
    primary=frozen['primary'];q=predictions[primary];choice=q.argmax(-1)
    failures=[]
    for i,idx in enumerate(dev):
        for c in range(2):
            j=int(choice[i,c]);oracle=int(lower[i,c].argmax())
            if lower[i,c,oracle]>lower[i,c,j]:
                failures.append(dict(state_uid=str(labels['state_uid'][idx]),controller=('base','held')[c],
                    eta_selected=j,eta_oracle=oracle,selected_success_count=int(success[i,c,j]),
                    oracle_success_count=int(success[i,c,oracle]),selected_numerical_count=int(unknown[i,c,j]),
                    selected_collision_count=int(labels['collision'][idx,c,j].sum())))
    write(HERE/'development_results.json',dict(
        scope='Diagnostic only: old test re-used as DEVELOPMENT, same two controllers seen in TRAIN; no LOSO or unseen-controller claim.',
        primary=primary,physical_states=n,model_results=summary,seed_results=seed_metrics,
        global_fixed_successes=fixed_mean.tolist(),per_controller_fixed_successes=lower.mean(0).tolist(),
        oracle_success_lower=float(lower.max(-1).mean()),oracle_b15_lower_count=int((success.max(-1)>=15).sum()),
        numerical_unknown=int(unknown.sum()),collision_count=int(labels['collision'][dev].sum()),
        safety='No exact Safety cache for these test states verified; not equated with FF eta=0.',
        comparison_unit='Physical state; both controller conditions retained within the bootstrap cluster',
        protocol_sha256=sha(HERE/'protocol.json')))
    write(HERE/'development_failure_cases.json',failures)
    np.savez_compressed(HERE/'development_bootstrap.npz',indices=boot,**bootstrap_arrays)
    with (HERE/'development_table.csv').open('w') as f:
        fields=['model','B15','mean_success','wrong_controller_success','mismatched_state_success','delta_corr','centered_delta_corr','triple_interaction_corr','reversal_accuracy','mismatch_difference','mismatch_ci95']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for kind,r in summary.items():
            w.writerow(dict(model=kind,B15=f"{r['selected_b15_lower_count']}/32",mean_success=r['selected_success_lower'],
                wrong_controller_success=r['wrong_controller_success'],mismatched_state_success=r['mean_mismatched_state_success'],
                delta_corr=r['controller_delta_correlation'],centered_delta_corr=r['eta_centered_controller_delta_correlation'],
                triple_interaction_corr=r['state_eta_centered_controller_delta_correlation'],reversal_accuracy=r['reversal_two_sided_accuracy'],
                mismatch_difference=r['matched_minus_mismatched_state']['mean_difference'],mismatch_ci95=r['matched_minus_mismatched_state']['ci95']))
    print(dict(primary=primary,results={k:{x:r[x] for x in ('selected_success_lower','selected_b15_lower_count','eta_centered_controller_delta_correlation','matched_minus_mismatched_state')} for k,r in summary.items()}),flush=True)


if __name__=='__main__':run()
