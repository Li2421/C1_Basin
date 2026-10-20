from common import *
from statistics_pipeline import evaluate_group,equivalence,best
from baselines import LEARNED,NONLEARNED,repair_class,CLASSES
from zoo_design import LOCKED
import csv


def main():
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'NEW_EVAL_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    data['uid_class']=equivalence(data['uids']);types=data['types'];nc=len(types);locked=np.isin(types,LOCKED);newdev=np.isin(types,[*NONLEARNED,'ring_ippo_conservative']);learned=np.isin(types,list(LEARNED))
    groups={str(t):types==t for t in sorted(set(types))};groups.update(development=~locked,learned=learned,nonlearned=np.isin(types,list(NONLEARNED)),locked=locked,new_development=newdev,locked_and_new_development=locked|newdev,old_learning_regression=np.isin(types,list(LEARNED-{'ring_ippo_conservative'})))
    results={}
    for group,mask in groups.items():
        if not mask.any():continue
        subset={k:v[mask] for k,v in data.items() if k!='state_uids'};selected={k:v[mask] for k,v in methods.items()}
        results[group]={'type_equal':evaluate_group(subset,selected,True,5000)}
        if group in ('development','learned','nonlearned','locked','new_development','locked_and_new_development','old_learning_regression'):results[group]['controller_equal']=evaluate_group(subset,selected,False,5000)
    paired=lambda group,key:results[group]['type_equal']['paired_comparisons']['pipeline__minus__'+key]
    primary=paired('locked','best_fixed');combined=results['locked_and_new_development']['type_equal']['methods']['pipeline']
    endpoints=dict(endpoint1={str(t):dict(**paired(str(t),'all_safety_baseline'),passed=paired(str(t),'all_safety_baseline')['CI95_pp'][0]>0) for t in sorted(set(types[locked]))},endpoint2=dict(**primary,passed=primary['CI95_pp'][0]>0,implicit_type_missing=True),endpoint3={b:paired('new_development',b) for b in ('all_Sprog','source_fixed')},endpoint4=dict(regret_pp=combined['state_crossfit_regret_pp'],regret_unknown_bounds_pp=combined['state_crossfit_regret_unknown_bounds_pp'],harm_possible=combined['harm_possible'],harm_certain=combined['harm_certain'],passed=combined['state_crossfit_regret_unknown_bounds_pp'][1]<=3 and combined['harm_possible']==0),endpoint5={g:{m:paired(g,m) for m in ('no_controller','no_state')} for g in ('locked','new_development')})
    by_group={}
    for group in ('locked','new_development','locked_and_new_development'):
        value=results[group]['type_equal']['methods']['pipeline']
        by_group[group]=dict(regret_pp=value['state_crossfit_regret_pp'],regret_unknown_bounds_pp=value['state_crossfit_regret_unknown_bounds_pp'],
            regret_CI95_pp=value['state_crossfit_regret_CI95_pp'],harm_possible=value['harm_possible'],harm_certain=value['harm_certain'],
            passed=value['state_crossfit_regret_unknown_bounds_pp'][1]<=3 and value['harm_possible']==0)
    endpoints['endpoint4'].update(by_group=by_group,combined_passed=by_group['locked_and_new_development']['passed'],
        passed=all(by_group[g]['passed'] for g in ('locked','new_development')),
        criterion='Locked types and new development types must each meet <=3pp average regret and zero possible harms; combined estimate also reported')
    lower=(data['labels']==1).astype(float);controller_best=best(lower[:,:,:16].mean((1,3)));repair=[];details=[]
    candidates=read(HERE/'CANDIDATES.json')['states'][48:]
    for i,name in enumerate(data['names']):
        classes=repair_class(methods['pipeline'][i]);scores=lower[i,:,:16].mean((0,2));tied=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-12))
        repair.append(dict(controller=str(name),type=str(types[i]),best_candidate=int(controller_best[i]),best_repair_class=CLASSES[int(repair_class(controller_best[i]))],optimal_candidates=tied.tolist(),optimal_repair_classes=sorted({CLASSES[int(repair_class(a))] for a in tied}),selected_class_fractions={c:float(np.mean(classes==j)) for j,c in enumerate(CLASSES)}))
        for s in range(24):
            chosen=int(methods['pipeline'][i,s]);oracle=int(best(lower[i,s,:16].mean(-1)))
            for a,action in enumerate(candidates[s]['actions']):
                value=data['labels'][i,s,a];details.append(dict(controller=str(name),type=str(types[i]),state_uid=str(data['state_uids'][s]),state_index=s,candidate=a,candidate_name=action['name'],selected=a==chosen,pipeline_choice=chosen,successes=int((value==1).sum()),known_seeds=int((value>=0).sum()),unknown_seeds=int((value==-1).sum()),seed0=int(value[0]),seed1=int(value[1]),rollout_uid_seed0=str(data['uids'][i,s,a,0]),rollout_uid_seed1=str(data['uids'][i,s,a,1]),eta_a=json.dumps(action['eta_a']),eta_b=json.dumps(action['eta_b']),switch_mode=action['switch_mode'],rhr_rule=json.dumps(action.get('rhr_rule')),state_in_sample_regret_pp=float((lower[i,s,oracle].mean()-lower[i,s,chosen].mean())*100)))
    write(HERE/'CONFIRMATORY_RESULTS.json',dict(utc=utc(),evidence='CONFIRMATORY',results=results,endpoints=endpoints,repair_class_selection=repair,locked_types=sorted(set(types[locked])),optional_implicit_gap=True,new_states=24,seeds=[0,1],bootstrap_replicates=5000,oracle_and_best_fixed_reselected_every_draw=True))
    with (HERE/'CONTROLLER_STATE_DETAILS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    seal('CONFIRMATORY_RESULTS.receipt.json',[HERE/'CONFIRMATORY_RESULTS.json',HERE/'CONTROLLER_STATE_DETAILS.csv',HERE/'confirmatory_statistics.py',HERE/'NEW_EVAL_BUNDLE.receipt.json',HERE/'NEW_EVAL_CHOICES.receipt.json'])
    print(json.dumps(dict(primary_endpoint=endpoints['endpoint2'],regret_and_harm=endpoints['endpoint4'])))


if __name__=='__main__':main()
