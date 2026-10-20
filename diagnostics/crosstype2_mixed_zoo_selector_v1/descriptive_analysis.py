from common import *
from zoo_design import LOCKED
from baselines import LEARNED,NONLEARNED,repair_class,CLASSES
from statistics_pipeline import best,oracle_choices
import csv


def main():
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'NEW_EVAL_ANALYSIS_CHOICES.npz',allow_pickle=False) as z:chosen={k:z[k] for k in z.files if k not in ('names','types')}
    rows=[];transitions={};all_transitions={}
    for i,name in enumerate(data['names']):
        for s in range(24):
            for q in (0,1):
                value=data['labels'][i,s,10,q]
                status='unknown' if value==-1 else 'success' if value==1 else 'timeout' if data['timeout'][i,s,10,q] else 'collision' if data['collision'][i,s,10,q] else 'other_known_failure'
                nominal_value=data['labels'][i,s,16,q];nominal_unknown=nominal_value<0
                nominal_status='unknown' if nominal_unknown else 'success' if nominal_value==1 else 'timeout' if data['timeout'][i,s,16,q] else 'collision' if data['collision'][i,s,16,q] else 'other_known_failure'
                hit=bool(data['collision'][i,s,16,q] and not nominal_unknown);rows.append(dict(controller=str(name),type=str(data['types'][i]),state_uid=str(data['state_uids'][s]),seed=q,nominal_status=nominal_status,nominal_collision=hit,nominal_unknown=bool(nominal_unknown),safety_status=status,nominal_uid=str(data['uids'][i,s,16,q]),safety_uid=str(data['uids'][i,s,10,q])))
                table=all_transitions.setdefault(str(data['types'][i]),{});pair=nominal_status+'__to__'+status;table[pair]=table.get(pair,0)+1
                if hit:
                    group=transitions.setdefault(str(data['types'][i]),dict(nominal_collision=0,success=0,timeout=0,collision=0,unknown=0,other_known_failure=0));group['nominal_collision']+=1;group[status]+=1
    for group in transitions.values():group['conditional_proportions']={k:v/group['nominal_collision'] for k,v in group.items() if k!='nominal_collision'}
    with (HERE/'NOMINAL_TO_SAFETY_TRANSITIONS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(HERE/'NOMINAL_TO_SAFETY_SUMMARY.json',transitions)
    write(HERE/'NOMINAL_TO_SAFETY_ALL_TRANSITIONS.json',dict(counts=all_transitions,denominator='Raw paired controller-state-seed cases by type; unknown retained as its own event'))
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:features=z['features'];feature_names=z['names'];types=z['types']
    ranges=[]
    for i,name in enumerate(data['names']):
        ci=int(np.flatnonzero(feature_names==name)[0]);kind=str(types[ci]);source_mask=~np.isin(types,LOCKED)
        if kind not in LOCKED:source_mask&=types!=kind
        source=features[source_mask];low=np.nanmin(source,0);high=np.nanmax(source,0);x=features[ci];valid=np.isfinite(x)&np.isfinite(low)&np.isfinite(high);out=valid&((x<low)|(x>high))
        ranges.append(dict(controller=str(name),type=kind,observed_dimensions=int(valid.sum()),outside_dimensions=int(out.sum()),outside_fraction_observed=float(out.sum()/max(valid.sum(),1)),missing_fraction=float(np.mean(~np.isfinite(x))),outside_dimension_indices=np.flatnonzero(out).tolist(),range_source='all development' if kind in LOCKED else 'other development types'))
    write(HERE/'FEATURE_RANGE_REPORT.json',ranges)
    candidates=read(HERE/'CANDIDATES.json')['states'][48:];variation=[]
    for i,name in enumerate(data['names']):
        choices=chosen['pipeline'][i];counts=np.bincount(choices,minlength=16);identities=[]
        for s,a in enumerate(choices):
            action=candidates[s]['actions'][int(a)]
            identities.append(json.dumps(dict(early=np.asarray(action['eta_a'],np.float64).tobytes().hex(),late=np.asarray(action['eta_b'],np.float64).tobytes().hex(),switch=action['switch_mode'],rhr=action.get('rhr_rule')),sort_keys=True))
        _,identity_counts=np.unique(identities,return_counts=True)
        variation.append(dict(controller=str(name),type=str(data['types'][i]),distinct_candidate_indices=int(np.sum(counts>0)),distinct_complete_action_identities=len(identity_counts),candidate_index_pair_disagreement=float(1-np.sum(counts*(counts-1))/(24*23)),actual_action_pair_disagreement=float(1-np.sum(identity_counts*(identity_counts-1))/(24*23))))
    write(HERE/'STATE_ACTION_VARIATION.json',variation)
    # State-wise privileged reference uses the opposite seed to choose each arm.
    cf=oracle_choices((data['labels'][None]==1).astype(float))['state_crossfit_oracle'][0]
    own=chosen['pipeline'][...,None,None]
    actual=np.take_along_axis(data['labels'],own,axis=2)[:,:,0,:]
    actual_uid=np.take_along_axis(data['uids'],own,axis=2)[:,:,0,:]
    reference=np.take_along_axis(data['labels'],cf[:,:,None,:],axis=2)[:,:,0,:]
    reference_uid=np.take_along_axis(data['uids'],cf[:,:,None,:],axis=2)[:,:,0,:]
    same=actual_uid==reference_uid;alo=actual==1;ahi=actual!=0;rlo=reference==1;rhi=reference!=0
    regret=(rlo.astype(float)-alo).mean(-1)*100
    regret_lower=np.where(same,0.,rlo.astype(float)-ahi).mean(-1)*100
    regret_upper=np.where(same,0.,rhi.astype(float)-alo).mean(-1)*100
    state_rows=[]
    for i,name in enumerate(data['names']):
        for s in range(24):
            state_rows.append(dict(controller=str(name),type=str(data['types'][i]),state_uid=str(data['state_uids'][s]),
                selected_candidate=int(chosen['pipeline'][i,s]),pipeline_seed0=int(actual[i,s,0]),pipeline_seed1=int(actual[i,s,1]),
                oracle_choice_for_seed0=int(cf[i,s,0]),oracle_choice_for_seed1=int(cf[i,s,1]),
                oracle_seed0=int(reference[i,s,0]),oracle_seed1=int(reference[i,s,1]),
                crossfit_regret_pp=float(regret[i,s]),regret_unknown_lower_pp=float(regret_lower[i,s]),regret_unknown_upper_pp=float(regret_upper[i,s])))
    with (HERE/'CONTROLLER_STATE_CROSSFIT_REGRET.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(state_rows[0]));writer.writeheader();writer.writerows(state_rows)
    # Descriptive timeout tails use real simulation evidence; never relabel events.
    index=read(HERE/'NEW_EVAL_TRACE_INDEX.json');cfg=read(HERE/'BASE_PROTOCOL.json')['native_manifest']['scenario_config'];diagnoses={};details=[]
    lookup={str(n):i for i,n in enumerate(data['names'])}
    for r in index:
        if r['candidate_index']!=10:continue
        i=lookup[r['controller']];s=r['state_index'];q=r['seed'];kind=str(data['types'][i])
        if kind not in {*NONLEARNED,'ring_ippo_conservative',*LOCKED}:continue
        if data['labels'][i,s,10,q]<0 or not data['timeout'][i,s,10,q]:continue
        assert sha(r['trace_path'])==r['trace_sha256']
        with np.load(r['trace_path'],allow_pickle=False) as z:p=z['positions'];v=z['velocities'];g=z['goals']
        d=np.linalg.norm(g[None]-p,axis=-1);speed=np.linalg.norm(v,axis=-1);window=round(1/cfg['dt']);start=max(window,len(p)-100);tt=np.arange(start,len(p));active=d[tt]>cfg['goal_tolerance']
        freeze=active&(speed[tt]<.05);moving=active&(speed[tt]>=.05)&((d[tt-window]-d[tt])<.05)
        counts=dict(remaining_robot_steps=int(active.sum()),strict_freeze_robot_steps=int(freeze.sum()),moving_no_progress_robot_steps=int(moving.sum()))
        item=dict(type=kind,controller=r['controller'],state_index=s,seed=q,trace_path=r['trace_path'],**counts);details.append(item)
        group=diagnoses.setdefault(kind,dict(timeouts=0,remaining_robot_steps=0,strict_freeze_robot_steps=0,moving_no_progress_robot_steps=0));group['timeouts']+=1
        for key,value in counts.items():group[key]+=value
    for group in diagnoses.values():
        denominator=max(group['remaining_robot_steps'],1);group['strict_freeze_fraction']=group['strict_freeze_robot_steps']/denominator;group['moving_no_progress_fraction']=group['moving_no_progress_robot_steps']/denominator
    write(HERE/'TIMEOUT_DIAGNOSTICS.json',dict(utc=utc(),groups=diagnoses,details=details,denominator='Remaining robots in last100 physical states; speed<.05 strict freeze, speed>=.05 and 1s progress<.05 moving without progress',deadlock_claim=False,new_rollouts=0))
    # Blinded capability rates may now be unsealed, after confirmation completion.
    assert (HERE/'new_evaluation_COMPLETE.receipt.json').exists();gates={}
    with rollout_db.connect(True) as db:
        for c in read(HERE/'gate_locked_CASES.json'):
            row=db.execute('SELECT success,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            group=gates.setdefault(c['base_controller_name'],{}).setdefault(c['arm'],dict(success=0,unknown=0,n=0));group['success']+=int(row['success'] and not row['numerical_failure']);group['unknown']+=int(row['numerical_failure']);group['n']+=1
    write(HERE/'LOCKED_CAPABILITY_UNBLINDED_AFTER_CONFIRMATION.json',dict(utc=utc(),before_freeze_rates_not_emitted=True,after_confirmation=True,controllers={name:dict(arms=arms,no_repair_needed=arms['safety_baseline']['success']/16>=.98) for name,arms in gates.items()}))
    checks=[];by_type={}
    for i,name in enumerate(data['names']):
        kind=str(data['types'][i])
        if kind not in LOCKED:continue
        selected=chosen['pipeline'][i];take=selected[:,None,None]
        actual=np.take_along_axis(data['labels'][i],take,axis=1)[:,0,:]
        actual_uid=np.take_along_axis(data['uids'][i],take,axis=1)[:,0,:]
        safety=data['labels'][i,:,10,:];same=actual_uid==data['uids'][i,:,10,:]
        lo=actual==1;hi=actual!=0;slo=safety==1;shi=safety!=0
        lower=float(np.where(same,0.,lo.astype(float)-shi).mean())
        upper=float(np.where(same,0.,hi.astype(float)-slo).mean())
        no_repair=gates[str(name)]['safety_baseline']['success']/16>=.98
        checks.append(dict(controller=str(name),type=kind,pilot_no_repair_needed=bool(no_repair),
            safety_success_lower_pct=float(slo.mean()*100),pipeline_success_lower_pct=float(lo.mean()*100),
            pipeline_minus_safety_pp=float((lo.astype(float)-slo).mean()*100),
            gain_unknown_bounds_pp=[lower*100,upper*100],
            selected_no_correction_fraction=float(np.mean(selected==10)),
            harm_possible=lower<=-.1+1e-12,harm_certain=upper<=-.1+1e-12))
    for kind in sorted(set(r['type'] for r in checks)):
        all_rows=[r for r in checks if r['type']==kind];rows=[r for r in all_rows if r['pilot_no_repair_needed']]
        by_type[kind]=dict(controllers=len(all_rows),pilot_no_repair_needed=len(rows),
            majority_no_repair_needed=len(rows)>len(all_rows)/2,
            no_repair_subset_selected_safety_fraction=float(np.mean([r['selected_no_correction_fraction'] for r in rows])) if rows else None,
            no_repair_subset_harm_possible=sum(r['harm_possible'] for r in rows),
            no_repair_subset_harm_certain=sum(r['harm_certain'] for r in rows))
    write(HERE/'LOCKED_NO_REPAIR_CHECK.json',dict(utc=utc(),after_confirmation=True,types=by_type,controllers=checks,
        status_definition='Pilot safety success >=98%; fresh-state outcomes reported separately',
        harm_definition='Pipeline >=10pp below safety; same-UID unknowns cancel'))
    seal('DESCRIPTIVE_ANALYSIS.receipt.json',[HERE/n for n in ('NOMINAL_TO_SAFETY_TRANSITIONS.csv','NOMINAL_TO_SAFETY_SUMMARY.json','NOMINAL_TO_SAFETY_ALL_TRANSITIONS.json','FEATURE_RANGE_REPORT.json','STATE_ACTION_VARIATION.json','CONTROLLER_STATE_CROSSFIT_REGRET.csv','TIMEOUT_DIAGNOSTICS.json','LOCKED_CAPABILITY_UNBLINDED_AFTER_CONFIRMATION.json','LOCKED_NO_REPAIR_CHECK.json','descriptive_analysis.py')],new_rollouts=0,simulation_evidence_only=True)
    print('Transition, feature-range and timeout diagnostics complete; locked gate rates unsealed after confirmation')


if __name__=='__main__':main()
