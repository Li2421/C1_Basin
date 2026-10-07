"""One-shot independent-family evaluation of frozen source-only models."""
from __future__ import annotations
import csv,json
from collections import defaultdict
import numpy as np
from scipy.special import expit
from .pipeline import OUT,CONTROLLERS,read,write
from .train_models import KINDS,SEEDS

def _csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def q_and_status(s,f):
    if s+f==0:return None,'UNKNOWN'
    return float(s/(s+f)),'B15' if s>=15 else 'NONB15' if f>=2 else 'UNKNOWN'

def evaluate(pred,raw,indices):
    pair_idx={int(s):{int(raw['eta_index'][i]):i for i in indices
       if int(raw['state_index'][i])==s} for s in sorted(set(raw['state_index'][indices]))}
    rows=[];valid_s=[];valid_f=[];valid_p=[]
    for c,controller in enumerate(CONTROLLERS):
        for state,eta_pairs in pair_idx.items():
            assert sorted(eta_pairs)==[10,15]
            ii=eta_pairs[10];jj=eta_pairs[15]
            pi=float(expit(pred[c,np.where(indices==ii)[0][0]]));pj=float(expit(pred[c,np.where(indices==jj)[0][0]]))
            si,fi=float(raw['success'][c,ii]),float(raw['failure'][c,ii])
            sj,fj=float(raw['success'][c,jj]),float(raw['failure'][c,jj])
            qi,bi=q_and_status(si,fi);qj,bj=q_and_status(sj,fj)
            selected=10 if pi>=pj else 15
            selected_q=qi if selected==10 else qj
            selected_b=bi if selected==10 else bj
            one_b15=(bi=='B15')!=(bj=='B15') and 'UNKNOWN' not in (bi,bj)
            any_b15=bi=='B15' or bj=='B15'
            strong=abs(qi-qj)>=.25
            rows.append({'controller':controller,'state_index':state,'eta10_Q':qi,'eta15_Q':qj,
                'eta10_success':int(si),'eta10_failure':int(fi),
                'eta15_success':int(sj),'eta15_failure':int(fj),
                'eta10_B15':bi,'eta15_B15':bj,'pred_eta10':pi,'pred_eta15':pj,
                'true_delta':qi-qj,'pred_delta':pi-pj,'selected_eta_index':selected,
                'selected_Q':selected_q,'selected_B15':selected_b,
                'one_B15_one_nonB15':one_b15,'any_B15':any_b15,
                'strong_Q_contrast':strong,
                'correct_strong_ranking':bool(np.sign(pi-pj)==np.sign(qi-qj)) if strong else None,
                'severe_false_positive':bool(max(pi,pj)>.9 and selected_q<=.5)})
            valid_s.extend((si,sj));valid_f.extend((fi,fj));valid_p.extend((pi,pj))
    ss=np.array(valid_s);ff=np.array(valid_f);pp=np.clip(np.array(valid_p),1e-5,1-1e-5)
    eligible=[r for r in rows if r['one_B15_one_nonB15']]
    strong=[r for r in rows if r['strong_Q_contrast']]
    return rows,{'NLL':float(np.sum(-ss*np.log(pp)-ff*np.log1p(-pp))/sum(ss+ff)),
        'Q_MAE':float(np.mean(abs(pp-ss/(ss+ff)))),
        'total_controller_state_cases':len(rows),
        'oracle_B15_cases':sum(r['any_B15'] for r in rows),
        'one_B15_one_nonB15_cases':len(eligible),
        'selected_B15_all':sum(r['selected_B15']=='B15' for r in rows),
        'selected_B15_eligible':sum(r['selected_B15']=='B15' for r in eligible),
        'mean_selected_Q':float(np.mean([r['selected_Q'] for r in rows])),
        'strong_Q_contrasts':len(strong),
        'strong_contrast_sign_accuracy':float(np.mean([r['correct_strong_ranking'] for r in strong])) if strong else None,
        'severe_false_positive':sum(r['severe_false_positive'] for r in rows)}

def main():
    raw=dict(np.load(OUT/'dataset.npz'))
    indices=np.flatnonzero(raw['split']=='validation')
    assert len(indices)>=24
    rows=[];metrics=[];picks={}
    for kind in KINDS:
        for seed in SEEDS:
            base=OUT/'models'/kind/f'seed{seed}'
            assert read(base/'complete.json')['independent_validation_labels_used_for_checkpoint'] is False
            p=np.load(base/'frozen_validation_predictions.npz')
            np.testing.assert_array_equal(p['pair_indices'],indices)
            for condition,key in [('correct','correct_logits'),('wrong_controller','wrong_controller_logits'),
                                  ('state_shuffled','state_shuffled_logits')]:
                rr,m=evaluate(p[key],raw,indices)
                rows.extend({'kind':kind,'seed':seed,'condition':condition,**r} for r in rr)
                metrics.append({'kind':kind,'seed':seed,'condition':condition,**m})
                picks[(kind,seed,condition)]={(r['controller'],r['state_index']):r for r in rr}
    _csv(OUT/'frozen_validation_per_state.csv',rows)
    _csv(OUT/'frozen_validation_metrics.csv',metrics)
    # Analytic non-neural controller-by-eta prior, fitted only on source FIT.
    plan=read(OUT/'train_split.json')
    fit=np.flatnonzero(np.isin(raw['state_index'],plan['fit_state_indices'])&(raw['split']=='train'))
    prior=np.zeros((2,len(indices)),np.float32)
    for c in range(2):
        for eta in (10,15):
            z=fit[raw['eta_index'][fit]==eta]
            s=raw['success'][c,z].sum();f=raw['failure'][c,z].sum()
            rate=(s+1)/(s+f+2)
            prior[c,raw['eta_index'][indices]==eta]=rate
    prior_logits=np.log(prior/(1-prior))
    prior_rows,prior_metric=evaluate(prior_logits,raw,indices)
    _csv(OUT/'controller_eta_prior_per_state.csv',prior_rows)
    write(OUT/'controller_eta_prior_metrics.json',prior_metric)
    comparisons=[]
    for seed in SEEDS:
        ref=picks[('eta_only',seed,'correct')]
        controller_prior={(r['controller'],r['state_index']):r for r in prior_rows}
        for kind in ('additive','context_only','full','full_old_Q4','full_Q4_weight16'):
            cur=picks[(kind,seed,'correct')]
            controls=[('eta_only',ref),('controller_eta_prior',controller_prior)]
            if kind=='full':controls.append(('context_only',picks[('context_only',seed,'correct')]))
            for name,other in controls:
                eligible=[k for k in cur if cur[k]['one_B15_one_nonB15']]
                families=sorted({k[1] for k in eligible})
                effect=np.asarray([sum(int(cur[k]['selected_B15']=='B15')-int(other[k]['selected_B15']=='B15')
                    for k in eligible if k[1]==st) for st in families],int)
                count=np.asarray([sum(k[1]==st for k in eligible) for st in families],int)
                if families:
                    rng=np.random.default_rng(20261004)
                    draw=rng.integers(0,len(families),(10000,len(families)))
                    denom=np.maximum(1,count[draw].sum(1))
                    ci=np.quantile(effect[draw].sum(1)/denom,[.025,.975]).tolist()
                else:ci=[None,None]
                comparisons.append({'kind':kind,'seed':seed,'reference':name,
                    'eligible_cases':len(eligible),
                    'rescue':sum(cur[k]['selected_B15']=='B15' and other[k]['selected_B15']!='B15' for k in eligible),
                    'break':sum(other[k]['selected_B15']=='B15' and cur[k]['selected_B15']!='B15' for k in eligible),
                    'family_bootstrap_95pct_rate_difference_lower':ci[0],
                    'family_bootstrap_95pct_rate_difference_upper':ci[1],
                    'correct_context_NLL':next(r['NLL'] for r in metrics if r['kind']==kind and r['seed']==seed and r['condition']=='correct'),
                    'wrong_context_NLL':next(r['NLL'] for r in metrics if r['kind']==kind and r['seed']==seed and r['condition']=='wrong_controller')})
    _csv(OUT/'paired_selection_comparisons.csv',comparisons)
    # Natural controller and state ranking reversals are outcome-defined and
    # model-blind. The complete source validation cohort is always retained.
    truth={(r['controller'],r['state_index']):r for r in prior_rows}
    controller_reversals=[]
    for state in sorted(set(r['state_index'] for r in prior_rows)):
        a=truth[('alt',state)];b=truth[('second',state)]
        if abs(a['true_delta'])>=.25 and abs(b['true_delta'])>=.25 and np.sign(a['true_delta'])!=np.sign(b['true_delta']):
            controller_reversals.append(state)
    state_reversals={}
    for c in CONTROLLERS:
        positive=[r['state_index'] for r in prior_rows if r['controller']==c and r['true_delta']>=.25]
        negative=[r['state_index'] for r in prior_rows if r['controller']==c and r['true_delta']<=-.25]
        state_reversals[c]={'positive_families':len(positive),'negative_families':len(negative),
                            'cross_family_reversal_pairs':len(positive)*len(negative)}
    write(OUT/'interaction_evidence.json',{'controller_reversal_state_indices':controller_reversals,
         'controller_reversal_states':len(controller_reversals),
         'state_reversals':state_reversals,
         'construction':'outcome-defined, model-blind, full validation cohort retained',
         'no_model_selection_based_on_this_evidence':True})
    print(json.dumps({'metrics':metrics,'prior':prior_metric,
        'controller_reversals':len(controller_reversals),'state_reversals':state_reversals},indent=2))

if __name__=='__main__':main()
