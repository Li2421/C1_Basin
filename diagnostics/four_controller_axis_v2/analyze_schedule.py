"""Round19b schedule effects, grouped selection, and frozen-probe E learning."""
from __future__ import annotations
import csv,datetime,hashlib,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import connect
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def finite(x):return None if not np.isfinite(x) else float(x)
def argbest(values,indices):
    return max(indices,key=lambda j:(values[j],-j))
def main():
    fixed=read(HERE/'FIXED_RESULTS.json');assert fixed['headroom_gate_passed'] is True
    assert read(HERE/'SCHEDULE_MERGE.receipt.json')['experiment_uid']=='exp_round19b_four_controller_axis_schedule_v1'
    p=read(HERE/'SCHEDULE_PROTOCOL.json');cases=read(HERE/'SCHEDULE_CASES.json');assert len(cases)==8000
    names=p['controller_order'];arms=p['arms'];Y=np.zeros((25,5,32,2),dtype=np.int8);U=np.zeros_like(Y);unknown=[]
    with connect(True) as db:
        for c in cases:
            r=read(HERE/'stage3/raw'/f"{c['rollout_uid']}.json")
            assert r['case']==c and r['protocol_sha256']==sha(HERE/'SCHEDULE_PROTOCOL.json') and r['trace_sha256']==sha(r['trace_path'])
            stored=db.execute('SELECT success,numerical_failure,outcome,episode_length FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert stored and tuple(stored)==(r['success'],r['numerical_failure'],r['outcome'],r['episode_length'])
            i=names.index(c['base_controller_name']);j=arms.index(c['arm']);Y[i,j,c['state_index'],c['seed']]=r['success'];U[i,j,c['state_index'],c['seed']]=r['numerical_failure']
            if r['numerical_failure']:unknown.append(dict(rollout_uid=c['rollout_uid'],controller=names[i],arm=c['arm'],state_index=c['state_index'],seed=c['seed'],error=r['error']))
    post=preflight(HERE/'SCHEDULE_PLANNED_ROLLOUTS.json')
    assert post['summary']['ambiguous']==0 and post['summary']['genuinely_missing']==len(unknown)
    write(HERE/'SCHEDULE_POSTFLIGHT.json',dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        official_planner='shared_rollout_db.src.planner.preflight',summary=post['summary'],unknown_not_rerun=len(unknown)))
    fs=np.array([r['success_rates'] for r in fixed['table']],dtype=float);fr=np.array([r['success_counts'] for r in fixed['table']],dtype=float)
    known=64-U.sum((2,3));sr=np.divide(Y.sum((2,3)),known,out=np.full((25,5),np.nan),where=known>0)
    allrates=np.concatenate((fs,sr),axis=1);allarms=['F10','F13','F15']+arms
    assert np.isfinite(allrates).all(), 'entire controller-arm unknown: selection inconclusive'
    run_for=[r['run'] for r in fixed['table']];unique_runs=list(dict.fromkeys(run_for));folds=[]
    for run in unique_runs:
        source=[i for i,x in enumerate(run_for) if x!=run];held=[i for i,x in enumerate(run_for) if x==run]
        fixed_choice=argbest(allrates[source].mean(0),range(3))
        expanded_choice=argbest(allrates[source].mean(0),range(8))
        folds.append(dict(run=run,source_controllers=[names[i] for i in source],held_controllers=[names[i] for i in held],
            fixed_choice=allarms[fixed_choice],expanded_choice=allarms[expanded_choice],
            held=[dict(controller=names[i],fixed_regret_pp=100*(np.max(allrates[i,:3])-allrates[i,fixed_choice]),
                       expanded_regret_pp=100*(np.max(allrates[i])-allrates[i,expanded_choice]),
                       fixed_oracle=allarms[int(np.argmax(allrates[i,:3]))],expanded_oracle=allarms[int(np.argmax(allrates[i]))]) for i in held]))
    fixed_regrets=np.array([x['fixed_regret_pp'] for f in folds for x in f['held']]);expanded_regrets=np.array([x['expanded_regret_pp'] for f in folds for x in f['held']])
    pair_defs=[('S15to10_471',15,10),('S15to10_786',15,10),('S10to15_471',10,15),('S10to15_786',10,15),('S13to10_786',13,10)]
    effects=[];both=set()
    for i,name in enumerate(names):
        for arm,early,late in pair_defs:
            s=allrates[i,allarms.index(arm)];e=allrates[i,allarms.index('F'+str(early))];l=allrates[i,allarms.index('F'+str(late))]
            E=100*(s-l);L=100*(s-e);N=100*(e-l)
            assert abs((E-L)-N)<1e-8
            effects.append(dict(controller=name,arm=arm,early=early,late=late,E_pp=E,L_pp=L,N_pp=N))
            if E>=10 and L>=10:both.add(name)
    h_compress=bool(expanded_regrets.mean()<=fixed_regrets.mean()/3 and len(both)>=math.ceil(len(names)/2))
    votes=[f['expanded_choice'] for f in folds if f['expanded_choice'].startswith('S')]
    selected=max(arms,key=lambda a:(votes.count(a),-arms.index(a))) if votes else None
    r5=np.load(HERE/'R5_FROZEN.npz');assert r5['controller_names'].tolist()==names
    X=np.asarray(r5['scores'],dtype=float);assert X.shape==(25,2)
    learning=None
    if selected:
        _,early,late=next(x for x in pair_defs if x[0]==selected)
        E=100*(allrates[:,allarms.index(selected)]-allrates[:,allarms.index('F'+str(late))])
        preds=np.full(25,np.nan);baseline=np.full(25,np.nan);fold_details=[]
        for run in unique_runs:
            source=np.array([i for i,x in enumerate(run_for) if x!=run]);held=np.array([i for i,x in enumerate(run_for) if x==run])
            mean=X[source].mean(0);scale=X[source].std(0);scale=np.where(scale<1e-6,1.,scale)
            train=(X[source]-mean)/scale;test=(X[held]-mean)/scale
            ymean=E[source].mean();beta=np.linalg.solve(train.T@train+np.eye(2),train.T@(E[source]-ymean))
            preds[held]=ymean+test@beta;baseline[held]=ymean
            fold_details.append(dict(run=run,source_indices=source.tolist(),held_indices=held.tolist(),source_mean=ymean,
                                     feature_mean=mean.tolist(),feature_scale=scale.tolist(),coefficients=beta.tolist()))
        rho=float(spearmanr(E,preds).statistic);rmse=float(np.sqrt(np.mean((E-preds)**2)));base_rmse=float(np.sqrt(np.mean((E-baseline)**2)))
        learn_status=('INCONCLUSIVE' if not np.isfinite(rho) or base_rmse==0 else
                      'SUPPORTED' if rho>=.6 and rmse<=.7*base_rmse else 'REJECTED')
        learning=dict(status=learn_status,selected_schedule=selected,
                      selected_from_source_folds=votes,late_fixed_arm='F'+str(late),E_targets_pp=E.tolist(),predictions_pp=preds.tolist(),
                      source_mean_predictions_pp=baseline.tolist(),Spearman=finite(rho),RMSE_pp=rmse,baseline_RMSE_pp=base_rmse,
                      RMSE_ratio=rmse/base_rmse if base_rmse else None,folds=fold_details)
    else:learning=dict(status='INCONCLUSIVE',reason='No expanded source-average fold selected a schedule',selected_from_source_folds=[])
    result=dict(stage='schedule_and_E',status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                schedule_rollouts=8000,schedule_unknown=len(unknown),rates_definition='successes / known seeds; unknown excluded',
                all_arms=allarms,controller_order=names,success_rates=allrates.tolist(),schedule_success_counts=Y.sum((2,3)).tolist(),
                schedule_unknown_counts=U.sum((2,3)).tolist(),schedule_known_counts=known.tolist(),
                effects=effects,folds=folds,fixed_mean_regret_pp=float(fixed_regrets.mean()),fixed_max_regret_pp=float(fixed_regrets.max()),
                expanded_mean_regret_pp=float(expanded_regrets.mean()),expanded_max_regret_pp=float(expanded_regrets.max()),
                both_E_L_ge10pp_controllers=sorted(both),H_compress_Four='SUPPORTED' if h_compress else 'REJECTED',
                E_learning=learning,r5_freeze_sha256=sha(HERE/'R5_FREEZE.receipt.json'),postflight_sha256=sha(HERE/'SCHEDULE_POSTFLIGHT.json'))
    write(HERE/'SCHEDULE_RESULTS.json',result);write(HERE/'SCHEDULE_UNKNOWN.json',unknown)
    with (HERE/'E_L_N.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
    print(json.dumps(dict(status='COMPLETE',H_compress_Four=result['H_compress_Four'],E_learning=learning['status'],
        fixed_mean_regret_pp=result['fixed_mean_regret_pp'],expanded_mean_regret_pp=result['expanded_mean_regret_pp'],
        both_E_L=len(both),unknown=len(unknown))),flush=True)
if __name__=='__main__':main()
