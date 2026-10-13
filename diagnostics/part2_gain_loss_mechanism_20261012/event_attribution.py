"""Paired cached event audit; seed-level outcomes remain distinct from B15."""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from extract_cached import HERE,PREV,PAIRS,monitor,measure

def readcsv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def main():
    b15={(r['uid'],int(r['eta_index']),r['placement']):r['status'] for r in readcsv(PREV/'b15_by_candidate.csv')}
    rows=[r for r in readcsv(PREV/'gain_loss_audit.csv') if (r['old'],r['new'])==PAIRS.get(r['scene'])]
    assert len(rows)==12288
    metadata={};out=[];category=Counter();ring_type=Counter();example=[]
    for r in rows:
        uid=r['uid'];ei=int(r['eta_index']);sc=r['scene'];old,new=PAIRS[sc]
        ao=b15[(uid,ei,old)];bn=b15[(uid,ei,new)]
        bo='unknown' if 'UNKNOWN' in (ao,bn) else 'gain' if ao=='FALSE' and bn=='TRUE' else 'loss' if ao=='TRUE' and bn=='FALSE' else 'both_success' if ao=='TRUE' else 'both_failure'
        for p in (r['old_result'],r['new_result']):
            if p and p not in metadata:metadata[p]=json.loads(Path(p).read_text())
        a=metadata.get(r['old_result'],{});b=metadata.get(r['new_result'],{})
        out.append(dict(scene=sc,uid=uid,eta_index=ei,seed=int(r['seed']),b15_outcome=bo,seed_outcome=r['outcome'],
            old_termination=a.get('termination','missing'),new_termination=b.get('termination','missing'),
            old_steps=a.get('steps'),new_steps=b.get('steps'),
            old_ever_candidate=a.get('plant_summary',{}).get('ever_candidate_deadlock'),
            new_ever_candidate=b.get('plant_summary',{}).get('ever_candidate_deadlock'),
            old_max_stuck=a.get('plant_summary',{}).get('max_stuck_timer'),
            new_max_stuck=b.get('plant_summary',{}).get('max_stuck_timer')))
        category[(sc,bo,r['outcome'],a.get('termination'),b.get('termination'))]+=1
        if sc=='ring_exchange' and bo=='gain' and r['outcome']=='gain' and a.get('error') is None and b.get('error') is None:
            pa=Path(r['old_result']).with_suffix('.npz');pb=Path(r['new_result']).with_suffix('.npz')
            with np.load(pa) as z:apos=z['positions'].copy();goal=z['goals'].copy()
            with np.load(pb) as z:bpos=z['positions'].copy()
            err=np.linalg.norm(apos[1:]-goal[None],axis=2)
            individually=(err<=.09).any(axis=0)
            kind='early_traversal_failure' if not individually.all() else 'late_synchronization_failure'
            ring_type[kind]+=1
            if len(example)<20:
                example.append(dict(uid=uid,eta_index=ei,seed=int(r['seed']),kind=kind,
                    individual_target_ever=individually.tolist(),tt_steps=len(apos)-1,tf_steps=len(bpos)-1))
    fields=list(out[0])
    with (HERE/'event_pair_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(out)
    # Seed-0 first separation, monitor and geometric checkpoints, all outcome groups.
    summaries={}
    for sc,old,new in [('toy_give_way','FT','FF'),('ring_exchange','TT','TF')]:
        name='toy_loss_analysis.csv' if sc=='toy_give_way' else 'ring_gain_analysis.csv'
        a=readcsv(HERE/name)
        for group in ('gain','loss','both_success','both_failure'):
            s=[x for x in a if x['b15_outcome']==group]
            def median(k):
                vals=[float(x[k]) for x in s if x.get(k) not in ('',None)]
                return float(np.median(vals)) if vals else None
            summaries[f'{sc}:{group}']={
                'candidates':len(s),'valid_seed0':sum(bool(x['first_action_l2']) for x in s),
                'median_first_separation_step_5cm':median('first_separation_step_5cm'),
                'median_first_action_l2':median('first_action_l2'),
                'median_H20_worst_progress_delta':median('H20_worst_progress_delta'),
                'median_H80_worst_progress_delta':median('H80_worst_progress_delta'),
                'median_old_active_cbf_fraction80':median(old+'_active_cbf_fraction_80'),
                'median_new_active_cbf_fraction80':median(new+'_active_cbf_fraction_80'),
                'median_old_goal_step':median(old+'_first_goal_step'),
                'median_new_goal_step':median(new+'_first_goal_step')}
            if sc=='toy_give_way':
                summaries[f'{sc}:{group}'].update(old_bay_full=None,new_bay_full=None,
                    median_old_first_candidate=median(old+'_first_candidate'),
                    median_new_first_candidate=median(new+'_first_candidate'),
                    median_old_max_stuck=median(old+'_max_stuck_timer'),
                    median_new_max_stuck=median(new+'_max_stuck_timer'))
            else:
                summaries[f'{sc}:{group}'].update(median_old_H80_angular_sum=median(old+'_H80_angular_sum_abs'),
                    median_new_H80_angular_sum=median(new+'_H80_angular_sum_abs'))
    result=dict(seed_pairs=len(out),new_rollouts=0,category_counts={':'.join(map(str,k)):v for k,v in category.items()},
        ring_gain_seed_level_failure_types=dict(ring_type),ring_examples=example,seed0_group_summaries=summaries)
    (HERE/'event_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('seed pairs',len(out),'ring gain failure types',dict(ring_type))

if __name__=='__main__':main()
