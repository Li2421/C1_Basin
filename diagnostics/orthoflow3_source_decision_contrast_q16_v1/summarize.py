"""Out-of-family decision-contrast analysis; never selects checkpoints on outer labels."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.special import expit
from .pipeline import OUT,SRC,read,write
from .train_contrast import ARMS,SEEDS,FOLDS
from shared_rollout_db.src.rollout_db import connect

def _csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def _corr(a,b):
    if len(a)<3 or np.std(a)<1e-9 or np.std(b)<1e-9:return None
    return float(np.corrcoef(a,b)[0,1])

def _nll(p,s,f):
    p=np.clip(p,1e-5,1-1e-5)
    return float(np.sum(-s*np.log(p)-f*np.log1p(-p))/max(1,np.sum(s+f)))

def main():
    d=dict(np.load(SRC/'variants/large_20/dataset.npz'))
    evidence=dict(np.load(OUT/'full_Q16_evidence.npz'))
    states=read(SRC/'states.json');folds=read(OUT/'crossfit_splits.json')['folds']
    pairs=read(OUT/'pairs.json');pair_by_key={(r['state_index'],r['eta_index']):r for r in pairs}
    rows=[];summary=[];all_predictions={}
    for arm in ARMS:
      for seed in SEEDS:
        preds={}
        for fold in FOLDS:
            base=OUT/'crossfit_models'/f'fold{fold}'/arm/f'seed{seed}'
            assert (base/'complete.json').exists()
            pp=np.load(base/'outer_predictions.npz')
            for j,i in enumerate(pp['indices']):
                assert int(i) not in preds
                preds[int(i)]=float(expit(pp['logits'][2,j]))
        all_predictions[(arm,seed)]=preds
        score=[];truth=[];contrast_pred=[];contrast_true=[];wins=[];evals=[]
        severe=0;false_high=0
        for state in sorted(set(p['state_index'] for p in pairs)):
            i=state*16+1;j=state*16+3
            assert i in preds and j in preds
            pi,pj=preds[i],preds[j]
            si,fi=float(evidence['success'][2,i]),float(evidence['failure'][2,i])
            sj,fj=float(evidence['success'][2,j]),float(evidence['failure'][2,j])
            ni,nj=si+fi,sj+fj
            qi,qj=si/ni,sj/nj
            full=bool(evidence['full_16'][i] and evidence['full_16'][j])
            bi='B15' if si>=15 else 'NONB15' if fi>=2 else 'UNRESOLVED'
            bj='B15' if sj>=15 else 'NONB15' if fj>=2 else 'UNRESOLVED'
            pick=1 if pi>=pj else 3
            chosen_p=pi if pick==1 else pj;chosen_q=qi if pick==1 else qj
            chosen_b=bi if pick==1 else bj
            eligible=(bi=='B15') != (bj=='B15') and 'UNRESOLVED' not in (bi,bj)
            decided=abs(qi-qj)>=.25 and full
            row={'arm':arm,'seed':seed,'state_index':state,'source_group':states[state]['source_group'],
                 'fold':next(k for k,r in enumerate(folds) if state in r['outer_state_indices']),
                 'eta1_success':int(si),'eta1_failure':int(fi),'eta3_success':int(sj),'eta3_failure':int(fj),
                 'eta1_Q':qi,'eta3_Q':qj,'eta1_B15':bi,'eta3_B15':bj,
                 'eta1_pred':pi,'eta3_pred':pj,'predicted_delta':pi-pj,'true_delta':qi-qj,
                 'selected_eta_index':pick,'selected_true_Q':chosen_q,'selected_B15':chosen_b,
                 'eligible_B15_discrimination':eligible,'clear_Q_contrast':decided,'full_Q16_both':full}
            rows.append(row);score.extend((pi,pj));truth.extend(((si,fi),(sj,fj)))
            contrast_pred.append(pi-pj);contrast_true.append(qi-qj)
            if eligible:evals.append(int(chosen_b=='B15'))
            if decided:wins.append(int(np.sign(pi-pj)==np.sign(qi-qj)))
            severe+=int(chosen_p>.9 and chosen_q<=.5)
            false_high+=int((pi>.9 and qi<=.5)+(pj>.9 and qj<=.5))
        ss=np.array([v[0] for v in truth]);ff=np.array([v[1] for v in truth]);pp=np.array(score)
        summary.append({'arm':arm,'seed':seed,'families':len(contrast_true),
                        'two_probe_observed_NLL':_nll(pp,ss,ff),
                        'two_probe_Q_MAE':float(np.mean(abs(pp-ss/(ss+ff)))),
                        'Q_delta_MAE':float(np.mean(abs(np.array(contrast_pred)-contrast_true))),
                        'Q_delta_Pearson':_corr(contrast_pred,contrast_true),
                        'clear_contrasts':len(wins),'clear_contrast_sign_accuracy':float(np.mean(wins)) if wins else None,
                        'one_B15_one_nonB15_states':len(evals),
                        'eligible_B15_selected':int(sum(evals)),'eligible_B15_rate':float(np.mean(evals)) if evals else None,
                        'selected_high_confidence_false_Q_le_0p5':severe,
                        'all_probe_high_confidence_false_Q_le_0p5':false_high})
    _csv(OUT/'crossfit_out_of_family_predictions.csv',rows)
    _csv(OUT/'crossfit_metrics_by_seed.csv',summary)
    # Fold-local eta-only preference: fit the same two exact eta probes using
    # source FIT families only; never see an outer source family at fitting.
    eta_prior=[]
    for arm in ARMS:
        labels=dict(np.load(OUT/f'labels_{arm}.npz'))
        for fold,r in enumerate(folds):
            fit_states=r['fit_state_indices']
            rates=[]
            for eta_index in (1,3):
                ii=np.asarray([s*16+eta_index for s in fit_states],int)
                ss=labels['success'][2,ii].sum();ff=labels['failure'][2,ii].sum()
                rates.append(float(ss/(ss+ff)))
            pick=1 if rates[0]>=rates[1] else 3
            for state in r['outer_state_indices']:
                i=state*16+pick
                s=float(evidence['success'][2,i]);f=float(evidence['failure'][2,i])
                a=state*16+1;b=state*16+3
                sa,fa=float(evidence['success'][2,a]),float(evidence['failure'][2,a])
                sb,fb=float(evidence['success'][2,b]),float(evidence['failure'][2,b])
                eligible=((sa>=15 and fb>=2) or (sb>=15 and fa>=2))
                eta_prior.append({'arm':arm,'fold':fold,'state_index':state,'source_group':states[state]['source_group'],
                     'fit_eta1_rate':rates[0],'fit_eta3_rate':rates[1],
                     'selected_eta_index':pick,'selected_Q':s/(s+f),
                     'eligible_B15_discrimination':eligible,'selected_B15':bool(s>=15)})
    _csv(OUT/'fold_fit_eta_only_prior.csv',eta_prior)
    # The old Q4 labels are not a separate test set. This checks whether the
    # label upgrade actually resolves decision contrasts before interpreting models.
    decision=[]
    for st in sorted(set(p['state_index'] for p in pairs)):
        ii=st*16+1;jj=st*16+3
        q4=(d['success'][2,ii]/(d['success'][2,ii]+d['failure'][2,ii])-
            d['success'][2,jj]/(d['success'][2,jj]+d['failure'][2,jj]))
        q16=(evidence['success'][2,ii]/(evidence['success'][2,ii]+evidence['failure'][2,ii])-
             evidence['success'][2,jj]/(evidence['success'][2,jj]+evidence['failure'][2,jj]))
        decision.append({'state_index':st,'source_group':states[st]['source_group'],
                         'Q4_delta':float(q4),'Q16_delta':float(q16),'Q4_sign':int(np.sign(q4)),
                         'Q16_sign':int(np.sign(q16)),
                         'full_Q16_both':bool(evidence['full_16'][ii] and evidence['full_16'][jj])})
    _csv(OUT/'decision_contrast_label_audit.csv',decision)
    aggregate=[]
    for arm in ARMS:
        rr=[r for r in summary if r['arm']==arm]
        aggregate.append({'arm':arm,**{k+'_mean':float(np.mean([r[k] for r in rr]))
             for k in ('two_probe_observed_NLL','two_probe_Q_MAE','Q_delta_MAE','Q_delta_Pearson',
                       'clear_contrast_sign_accuracy','eligible_B15_rate') if all(r[k] is not None for r in rr)},
             **{k+'_std':float(np.std([r[k] for r in rr],ddof=1)) for k in
                ('two_probe_observed_NLL','two_probe_Q_MAE','Q_delta_MAE','Q_delta_Pearson',
                 'clear_contrast_sign_accuracy','eligible_B15_rate') if all(r[k] is not None for r in rr)},
             'seeds':list(SEEDS)})
    _csv(OUT/'crossfit_arm_comparison.csv',[{k:v for k,v in r.items() if k!='seeds'} for r in aggregate])
    labels={'Q4_Q16_contrast_correlation':_corr([r['Q4_delta'] for r in decision],[r['Q16_delta'] for r in decision]),
            'Q4_non_tied':sum(r['Q4_sign']!=0 for r in decision),
            'Q16_clear_abs_delta_ge_0p25':sum(abs(r['Q16_delta'])>=.25 for r in decision),
            'Q4_Q16_sign_agreement_when_both_nonzero':float(np.mean([r['Q4_sign']==r['Q16_sign'] for r in decision
              if r['Q4_sign'] and r['Q16_sign']])) if any(r['Q4_sign'] and r['Q16_sign'] for r in decision) else None}
    controller_uid=read(OUT/'protocol.json')['selected_controller_uid']
    seed_rates={}
    with connect(True) as db:
        for pair in pairs:
            good={}
            for rr in db.execute('''SELECT seed_key,success,numerical_failure FROM rollout
                WHERE state_uid=? AND eta_uid=? AND controller_uid=?''',
                (pair['state_uid'],pair['eta_uid'],controller_uid)):
                future=json.loads(rr['seed_key']).get('future_index',-1)
                if 0<=future<16 and not rr['numerical_failure']:
                    good[future]=int(rr['success'])
            assert len(good)>=14
            seed_rates[(pair['state_index'],pair['eta_index'])]=(
                np.mean([v for k,v in good.items() if k<8]),
                np.mean([v for k,v in good.items() if k>=8]))
    halves=[];half_by_state={}
    for state in sorted(set(p['state_index'] for p in pairs)):
        e1=seed_rates[(state,1)];e3=seed_rates[(state,3)]
        half_by_state[state]=(e1[0]-e3[0],e1[1]-e3[1])
        halves.append(half_by_state[state])
    labels['Q8_halves_contrast_correlation']=_corr([v[0] for v in halves],[v[1] for v in halves])
    labels['Q8_halves_both_nonzero_sign_agreement']=float(np.mean([
        np.sign(a)==np.sign(b) for a,b in halves if a and b])) if any(a and b for a,b in halves) else None
    labels['Q8_halves_both_nonzero_count']=int(sum(bool(a and b) for a,b in halves))
    clear_halves=[half_by_state[r['state_index']] for r in decision if abs(r['Q16_delta'])>=.25]
    labels['clear_Q16_contrast_Q8_halves_same_sign']=int(sum(np.sign(a)==np.sign(b) for a,b in clear_halves))
    labels['clear_Q16_contrast_count']=len(clear_halves)
    write(OUT/'decision_label_summary.json',labels)
    write(OUT/'crossfit_summary.json',{'arms':aggregate,'label_audit':labels,
        'eta_only_fit_prior':{arm:{'selected_B15_eligible':int(sum(r['selected_B15'] for r in eta_prior if r['arm']==arm and r['eligible_B15_discrimination'])),
                                'eligible_states':int(sum(r['eligible_B15_discrimination'] for r in eta_prior if r['arm']==arm))}
                              for arm in ARMS},
        'scope':'Source TRAIN family crossfit only. No held-controller or LOSO target labels.',
        'causal_interpretation':'Q16_rate_weight4 versus original_Q4 isolates label precision; Q4_weight16 isolates increased weight.'})
    print(json.dumps({'arms':aggregate,'labels':labels},indent=2))

if __name__=='__main__':main()
