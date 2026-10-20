"""One-time locked endpoints; choices and all model code must already be frozen."""
from common import *
from selector import ARMS,BASELINE,few_labels,paired_bounds,crossfit_oracle,best_arm
from analysis_statistics import evaluate
from transition_summary import summarize
import csv


def main(evaluation_phase,tuning_phase):
    assert not (HERE/'CONFIRMATORY_RESULTS.json').exists(),'Locked analysis is written once'
    for name in ('FROZEN_PIPELINE.receipt.json','LOCKED_CHOICES.receipt.json','LOCKED_COMPARATOR_CHOICES.receipt.json','LOCKED_BUNDLE.receipt.json'):
        for path,digest in read(HERE/name)['files'].items():assert sha(path)==digest
    chosen=read(HERE/'LOCKED_CHOICES.json');choice_time=read(HERE/'LOCKED_CHOICES.receipt.json')['utc']
    times=[]
    for path in HERE.glob('*_SUBMISSION.json'):
        receipt=read(path)
        if set(receipt.get('phases',[])) & {evaluation_phase,tuning_phase}:times.append(receipt['utc'])
    assert times and choice_time<min(times),'Choices must precede the first locked evaluation/tuning submission'
    with np.load(HERE/'LOCKED_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert set(data['types']).issubset({'ring_ippo','implicit'})
    assert np.isin(data['labels'],[-1,0,1]).all() and np.isin(data['nominal'],[-1,0,1]).all()
    names=list(data['names']);types=data['types'];n=len(names);lookup={r['controller']:r for r in chosen['controllers']}
    assert set(names)==set(lookup)
    fixed={f'all_{arm}':np.full(n,i,int) for i,arm in enumerate(ARMS)}
    fixed['pipeline']=np.array([ARMS.index(lookup[name]['arm']) for name in names])
    fixed['r5_pca2_pipeline']=np.array([ARMS.index(lookup[name]['r5_arm']) for name in names])
    comparators=read(HERE/'LOCKED_COMPARATOR_CHOICES.json')['methods']
    for method,record in comparators.items():
        fixed[method]=np.array([ARMS.index(record['choices'][str(name)]) if record['choices'][str(name)] in ARMS else -1 for name in names])
    for k in (2,4,8):fixed[f'few_labels_{k}']=few_labels(data['tuning']==1,k)
    summaries=evaluate(data,fixed)
    labels=data['labels'];lo=(labels==1).astype(float);hi=(labels!=0).astype(float)
    definitions=read(HERE/'LOCKED_ENDPOINT_DEFINITIONS.json')
    assert definitions['utc']<min(times)
    repair={}
    for t in sorted(set(types)):
        mask=types==t
        comparison=paired_bounds(lo[mask,:,BASELINE],hi[mask,:,BASELINE],lo[mask,:,0],hi[mask,:,0],types[mask])
        repair[str(t)]=dict(**comparison,status='TESTED' if mask.sum()>=4 else 'INSUFFICIENT_INSTANCES',
                           passed=bool(mask.sum()>=4 and comparison['CI95_pp'][0]>0))
    ppo=types=='ring_ippo';ppo_low=lo[ppo,:,0].mean((1,2)) if ppo.any() else np.array([])
    ppo_rare=bool(len(ppo_low)>=4 and np.sum(ppo_low>=definitions['ppo_safety_success_threshold'])>len(ppo_low)/2)
    expected={'implicit'} if ppo_rare else {'implicit','ring_ippo'}
    available=set(types.tolist());primary_complete=expected.issubset(available) and all(np.sum(types==t)>=4 for t in expected)
    if ppo_rare and 'implicit' in available:primary=summaries['implicit']['type_equal']['methods']['pipeline']
    else:primary=summaries['combined']['type_equal']['methods']['pipeline']
    full=summaries['combined']['type_equal']['methods']['pipeline']
    with np.load(HERE/'DEVELOPMENT_FEATURES.npz',allow_pickle=False) as z:source=z['features']
    lower=np.nanmin(source,axis=0);upper=np.nanmax(source,axis=0);outlier=[]
    for name,t,x in zip(names,types,data['features']):
        valid=np.isfinite(x)&np.isfinite(lower)&np.isfinite(upper)
        outlier.append(dict(controller=str(name),type=str(t),valid_features=int(valid.sum()),
            outside_development_range_fraction=float(np.mean((x[valid]<lower[valid])|(x[valid]>upper[valid]))) if valid.any() else None))
    endpoints=dict(repair_vs_safety=repair,
        primary_zero_shot=dict(status='TESTED' if primary_complete else 'INCOMPLETE_LOCKED_COHORT',
            passed=bool(primary_complete and primary['gain_CI95_pp'][0]>0),expected_types=sorted(expected),
            missing_types=sorted(expected-available),gain_pp=primary['gain_vs_schedule_pp'],CI95_pp=primary['gain_CI95_pp']),
        near_oracle_and_no_harm=dict(status='TESTED' if all(np.sum(types==t)>=4 for t in ('implicit','ring_ippo')) else 'INCOMPLETE_LOCKED_COHORT',
            available_cohort_regret_pp=full['crossfit_regret_pp'],available_cohort_CI95_pp=full['crossfit_regret_CI95_pp'],
            available_cohort_harm_possible=full['harm_possible'],available_cohort_numerical_target_met=full['crossfit_regret_pp']<=3 and full['harm_possible']==0),
        ppo_mostly_safety_at_least_98_percent=ppo_rare,
        ppo_interpretation='In this evaluation, most PPO instances already succeed >=98% with safety projection' if ppo_rare else 'No rare-repair exception applied')
    transition=[]
    for i,name in enumerate(names):
        for s in range(32):
            for q in range(2):
                before='unknown' if data['nominal_unknown'][i,s,q] else 'collision' if data['nominal_collision'][i,s,q] else 'success' if data['nominal'][i,s,q]==1 else 'timeout' if data['nominal_timeout'][i,s,q] else 'other'
                after='unknown' if data['unknown'][i,s,0,q] else 'collision' if data['collision'][i,s,0,q] else 'success' if labels[i,s,0,q]==1 else 'timeout' if data['timeout'][i,s,0,q] else 'other'
                transition.append(dict(controller=str(name),type=str(types[i]),state_uid=str(data['state_uids'][s]),seed=q,nominal_outcome=before,safety_outcome=after))
    with (HERE/'LOCKED_NOMINAL_TO_SAFETY_TRANSITIONS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=transition[0].keys());writer.writeheader();writer.writerows(transition)
    write(HERE/'LOCKED_NOMINAL_TO_SAFETY_SUMMARY.json',summarize(transition))
    cross,crosschoices=crossfit_oracle(lo);rates=lo.mean((1,3));details=[];descriptive={}
    for i,name in enumerate(names):
        arm=int(fixed['pipeline'][i]);selected=lo[i,:,arm]
        details.append(dict(controller=str(name),type=str(types[i]),pipeline_arm=ARMS[arm],
            in_sample_best_arm=ARMS[int(best_arm(rates[i]))],success_pct=float(selected.mean()*100),
            crossfit_regret_pp=float((cross[i]-selected).mean()*100),in_sample_regret_pp=float((rates[i].max()-selected.mean())*100),
            crossfit_choices=[ARMS[int(a)] for a in crosschoices[i]],
            per_arm_lower_success_pct={a:float(rates[i,j]*100) for j,a in enumerate(ARMS)}))
    for t in sorted(available):
        mask=types==t;averages=rates[mask].mean(0);x=data['features'][mask]
        descriptive[str(t)]=dict(in_sample_best_fixed_arms=[a for a,v in zip(ARMS,averages) if abs(v-averages.max())<1e-12],
            per_arm_lower_success_pct={a:float(v*100) for a,v in zip(ARMS,averages)},
            mean_safety_probe_strict_freeze=float(np.nanmean(x[:,0])) if np.isfinite(x[:,0]).any() else None,
            mean_safety_probe_moving_no_progress=float(np.nanmean(x[:,1])) if np.isfinite(x[:,1]).any() else None,
            diagnostic_interpretation='Outcome-free probe behavior; a timeout is not a verified deadlock')
    write(HERE/'LOCKED_CONTROLLER_DETAILS.json',details)
    write(HERE/'CONFIRMATORY_RESULTS.json',dict(utc=utc(),evidence_level='CONFIRMATORY',endpoints=endpoints,summaries=summaries,
        choices_utc=choice_time,first_label_submission_utc=min(times),out_of_range=outlier,
        comparator_gaps={k:v for k,v in comparators.items() if not v['available_for_complete_cohort']},
        missing_locked_types=sorted({'implicit','ring_ippo'}-available),descriptive_mechanisms_and_preferences=descriptive,
        timeout_not_called_deadlock=True))
    seal('CONFIRMATORY_RESULTS.receipt.json',[HERE/'CONFIRMATORY_RESULTS.json',HERE/'LOCKED_NOMINAL_TO_SAFETY_TRANSITIONS.csv',
        HERE/'LOCKED_NOMINAL_TO_SAFETY_SUMMARY.json',HERE/'LOCKED_CONTROLLER_DETAILS.json',HERE/'transition_summary.py',
        HERE/'confirmatory_analysis.py',HERE/'analysis_statistics.py',HERE/'LOCKED_ENDPOINT_DEFINITIONS.json',
        HERE/'LOCKED_CHOICES.receipt.json',HERE/'LOCKED_BUNDLE.receipt.json'],analysis_run_once=True)
    print('One-time locked endpoint analysis recorded; missing types remain explicit',flush=True)

if __name__=='__main__':main(*sys.argv[1:])
