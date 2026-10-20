"""Separate uniform known24/unseen48 cohorts; ragged physical action menus."""
from common import *
from statistics_pipeline import oracle_choices
import csv

def cohort_pack(cohort,maps,chosen):
    names=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')[cohort+'_controllers'];ns=48 if cohort=='unseen' else 24;nc=len(names)
    if not names:return None,[]
    maps=[r for r in maps if r['cohort']==cohort];chosen=[r for r in chosen if r['cohort']==cohort];method_names=list(maps[0]['method_uids']);orders=[];actions=[]
    for r,a in zip(maps,chosen):
        assert r['controller']==a['controller'] and r['state_uid']==a['state_uid'];unique=[];metadata=[]
        for method,pair in r['method_uids'].items():
            pair=tuple(pair)
            if pair not in unique:unique.append(pair);metadata.append(a['methods'][method])
        orders.append(unique);actions.append(metadata)
    maximum=max(map(len,orders));shape=(nc,ns,maximum,2);labels=np.full(shape,-2,np.int8);uids=np.full(shape,'',dtype='U96');events={k:np.zeros(shape,np.int8) for k in ('collision','timeout','unknown')};oracle_mask=np.zeros(shape[:3],bool);non_diag=np.zeros(shape[:3],bool);methods={k:np.full((nc,ns),-1,np.int32) for k in method_names};details=[];types=['']*nc
    with rollout_db.connect(True) as db:
        for record,order,metadata in zip(maps,orders,actions):
            i=names.index(record['controller']);s=record['state_index'];types[i]=record['type']
            for method,pair in record['method_uids'].items():methods[method][i,s]=order.index(tuple(pair))
            for a,(pair,action) in enumerate(zip(order,metadata)):
                non_diag[i,s,a]=not np.array_equal(action['eta_a'],action['eta_b']);oracle_mask[i,s,a]=action['name']!='nominal_only';actual_switch=[]
                for q,rid in enumerate(pair):
                    row=db.execute('SELECT success,collision,timeout,numerical_failure FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();assert row is not None;labels[i,s,a,q]=-1 if row['numerical_failure'] else row['success'];uids[i,s,a,q]=rid
                    for k in events:events[k][i,s,a,q]=row['numerical_failure'] if k=='unknown' else row[k]
                    raw=HERE/'new_evaluation/raw'/f'{rid}.json';actual_switch.append(read(raw).get('progress_switch_step') if action['switch_mode']=='progress' and raw.exists() else action['switch_step'] if non_diag[i,s,a] and action['switch_mode'] in ('step','step500') else None)
                details.append(dict(cohort=cohort,controller=record['controller'],type=record['type'],state_uid=record['state_uid'],state_index=s,evaluated_action_index=a,selected_by_primary=a==methods['pipeline'][i,s],selected_by_methods=json.dumps([k for k in method_names if methods[k][i,s]==a]),candidate_index=action.get('candidate_index',-1),eta_early=json.dumps(action['eta_a']),eta_late=json.dumps(action['eta_b']),switch_mode=action['switch_mode'],switch_step=action['switch_step'] if non_diag[i,s,a] and action['switch_mode']!='progress' else None,actual_switch_step_seed0=actual_switch[0],actual_switch_step_seed1=actual_switch[1],non_diagonal=bool(non_diag[i,s,a]),successes=int((labels[i,s,a]==1).sum()),known_seeds=int((labels[i,s,a]>=0).sum()),unknown_seeds=int((labels[i,s,a]==-1).sum()),seed0=int(labels[i,s,a,0]),seed1=int(labels[i,s,a,1]),rollout_uid_seed0=pair[0],rollout_uid_seed1=pair[1],oracle_eligible=bool(oracle_mask[i,s,a])))
    assert all(np.all(v>=0) for v in methods.values());assert np.array_equal(labels!=-2,uids!='')
    oracle=oracle_choices(labels[None],oracle_mask[None])['evaluated_state_crossfit_oracle'][0];lower=(labels==1).astype(float)
    for row in details:
        i=names.index(row['controller']);s=row['state_index'];a=row['evaluated_action_index'];inside=max(lower[i,s,oracle_mask[i,s]].mean(-1));row['evaluated_state_in_sample_regret_pp']=float((inside-lower[i,s,a].mean())*100);point=[];lo=[];hi=[]
        for q in (0,1):
            o=int(oracle[i,s,q]);same=uids[i,s,o,q]==uids[i,s,a,q];ol=lower[i,s,o,q];al=lower[i,s,a,q];ou=labels[i,s,o,q]!=0;au=labels[i,s,a,q]!=0;point.append(ol-al);lo.append(0 if same else ol-au);hi.append(0 if same else ou-al)
        row['evaluated_state_crossfit_regret_pp']=float(np.mean(point)*100);row['crossfit_regret_unknown_lower_pp']=float(np.mean(lo)*100);row['crossfit_regret_unknown_upper_pp']=float(np.mean(hi)*100);row['oracle_action_seed0']=int(oracle[i,s,0]);row['oracle_action_seed1']=int(oracle[i,s,1])
    prefix=cohort.upper();np.savez_compressed(HERE/f'{prefix}_EVAL_BUNDLE.npz',names=names,types=types,labels=labels,uids=uids,oracle_mask=oracle_mask,non_diagonal=non_diag,**events);np.savez_compressed(HERE/f'{prefix}_EVAL_ANALYSIS_CHOICES.npz',names=names,types=types,**methods)
    return dict(cohort=cohort,controllers=nc,states=ns,maximum_unique_actions=maximum,valid_seed_records=int((uids!='').sum()),bundle=str(HERE/f'{prefix}_EVAL_BUNDLE.npz'),choices=str(HERE/f'{prefix}_EVAL_ANALYSIS_CHOICES.npz')),details

def main():
    assert (HERE/'new_evaluation_COMPLETE.receipt.json').exists();maps=read(HERE/'NEW_EVAL_METHOD_UIDS.json');chosen=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];summaries=[];details=[]
    for cohort in ('known','unseen'):
        summary,rows=cohort_pack(cohort,maps,chosen)
        if summary is not None:summaries.append(summary);details.extend(rows)
    assert sum(r['valid_seed_records'] for r in summaries)==len(read(HERE/'new_evaluation_CASES.json'))
    for name,rows in [('CONTROLLER_STATE_DETAILS.csv',details),('CONTROLLER_STATE_CROSSFIT_REGRET.csv',[r for r in details if r['selected_by_primary']])]:
        with (HERE/name).open('w') as f:writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(rows)
    write(HERE/'NEW_EVAL_BUNDLE_INDEX.json',dict(utc=utc(),cohorts=summaries))
    seal('NEW_EVAL_BUNDLE.receipt.json',[HERE/'NEW_EVAL_BUNDLE_INDEX.json',HERE/'CONTROLLER_STATE_DETAILS.csv',HERE/'CONTROLLER_STATE_CROSSFIT_REGRET.csv',HERE/'assemble_evaluation.py',HERE/'new_evaluation_COMPLETE.receipt.json',HERE/'NEW_EVAL_CHOICES.receipt.json']+[Path(r[k]) for r in summaries for k in ('bundle','choices')],ragged_padding_missing_not_unknown=True,all_selected_methods_two_seeds_complete=True)
if __name__=='__main__':main()
