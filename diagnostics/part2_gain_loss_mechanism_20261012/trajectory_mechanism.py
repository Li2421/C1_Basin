"""Retrospective full-trajectory event ordering for the two target flip groups."""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from extract_cached import HERE,PREV,measure,monitor

def readcsv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def first(mask):
    where=np.flatnonzero(mask)
    return int(where[0]) if len(where) else None

def main():
    b15={(r['uid'],r['eta_index'],r['placement']):r['status'] for r in readcsv(PREV/'b15_by_candidate.csv')}
    source=readcsv(PREV/'gain_loss_audit.csv');out=[]
    toy_cfg=dict(dt=.05,progress_window_seconds=2.,deadlock_hold_seconds=5.,progress_epsilon=.01,
        max_speed=.5,speed_epsilon_fraction=.05,goal_tolerance=.08)
    for r in source:
        sc=r['scene'];old,new=r['old'],r['new']
        if (sc,old,new) not in (('toy_give_way','FT','FF'),('ring_exchange','TT','TF')):continue
        if r['outcome']!=('loss' if sc=='toy_give_way' else 'gain'):continue
        a=b15[(r['uid'],r['eta_index'],old)];b=b15[(r['uid'],r['eta_index'],new)]
        if (a,b)!=(('TRUE','FALSE') if sc=='toy_give_way' else ('FALSE','TRUE')):continue
        arrays={}
        for ch,path in ((old,r['old_result']),(new,r['new_result'])):
            with np.load(Path(path).with_suffix('.npz')) as z:
                arrays[ch]={k:z[k].copy() for k in ('positions','actions','goals','certificates','noise_keys')}
        A=arrays[old];B=arrays[new];assert np.array_equal(A['noise_keys'][0],B['noise_keys'][0])
        n=min(len(A['positions']),len(B['positions']))
        gap=np.linalg.norm((A['positions'][:n]-B['positions'][:n]).reshape(n,-1),axis=1)
        row=dict(scene=sc,uid=r['uid'],eta_index=int(r['eta_index']),seed=int(r['seed']),
            seed_outcome=r['outcome'],old_termination=r['old_termination'],new_termination=r['new_termination'],
            first_joint_separation_5cm=first(gap>.05),first_action_l2=float(np.linalg.norm(A['actions'][0]-B['actions'][0])),
            old_active_cbf_fraction80=float(np.mean(A['certificates'][:80,0]<1e-6)),
            new_active_cbf_fraction80=float(np.mean(B['certificates'][:80,0]<1e-6)))
        for h in (10,20,80):
            ma=measure(A['positions'],A['actions'],A['goals'],h)
            mb=measure(B['positions'],B['actions'],B['goals'],h)
            row[f'H{h}_worst_progress_delta']=mb['progress_min']-ma['progress_min']
            row[f'H{h}_joint_progress_delta']=mb['progress_sum']-ma['progress_sum']
            row[f'H{h}_position_gap']=float(gap[min(h,n-1)])
            if sc=='ring_exchange':
                row[f'H{h}_angular_advantage']=mb['angular_sum_abs']-ma['angular_sum_abs']
        if sc=='toy_give_way':
            for ch,d in arrays.items():
                p=d['positions'];bay=(np.abs(p[:,:,0])<=.2)&(p[:,:,1]>.2)
                row[ch+'_first_bay_step']=first(bay.any(axis=1))
                row[ch+'_first_x_order_reversal']=first(p[:,0,0]-p[:,1,0]>0)
                mon=monitor(p,d['actions'],d['goals'],toy_cfg)
                row[ch+'_first_candidate_step']=mon['first_candidate']
                row[ch+'_max_stuck_timer']=mon['max_stuck_timer']
        else:
            for ch,d in arrays.items():
                err=np.linalg.norm(d['positions'][1:]-d['goals'][None],axis=2)
                row[ch+'_individual_goal_ever_count']=int((err<=.09).any(axis=0).sum())
                row[ch+'_first_individual_goal_step']=first((err<=.09).any(axis=1))
        out.append(row)
    fields=list(dict.fromkeys(k for r in out for k in r))
    with (HERE/'critical_event_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(out)
    summary={}
    for sc in ('toy_give_way','ring_exchange'):
        r=[x for x in out if x['scene']==sc]
        def rate(name,predicate=lambda v:bool(v)):
            return sum(predicate(x[name]) for x in r)/len(r)
        def med(name):
            a=[float(x[name]) for x in r if x.get(name) is not None]
            return float(np.median(a)) if a else None
        z=dict(n=len(r),median_first_joint_separation_5cm=med('first_joint_separation_5cm'),
               median_first_action_l2=med('first_action_l2'),
               median_H20_worst_progress_delta=med('H20_worst_progress_delta'),
               median_H80_worst_progress_delta=med('H80_worst_progress_delta'),
               median_old_active_cbf_fraction80=med('old_active_cbf_fraction80'),
               median_new_active_cbf_fraction80=med('new_active_cbf_fraction80'))
        if sc=='toy_give_way':
            z.update(old_bay_rate=rate('FT_first_bay_step',lambda v:v is not None),
                new_bay_rate=rate('FF_first_bay_step',lambda v:v is not None),
                old_order_reversal_rate=rate('FT_first_x_order_reversal',lambda v:v is not None),
                new_order_reversal_rate=rate('FF_first_x_order_reversal',lambda v:v is not None),
                median_old_first_bay_step=med('FT_first_bay_step'),
                median_new_first_candidate_step=med('FF_first_candidate_step'),
                new_deadlock_rate=sum(x['new_termination']=='deadlock' for x in r)/len(r),
                old_bay_new_no_bay=sum(x['FT_first_bay_step'] is not None and x['FF_first_bay_step'] is None for x in r))
        else:
            z.update(early_traversal_failure_count=sum(x['TT_individual_goal_ever_count']<4 for x in r),
                late_synchronization_failure_count=sum(x['TT_individual_goal_ever_count']==4 for x in r),
                median_H80_angular_advantage=med('H80_angular_advantage'),
                median_new_first_individual_goal_step=med('TF_first_individual_goal_step'))
        summary[sc]=z
    (HERE/'critical_event_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print('critical pairs',len(out),summary)

if __name__=='__main__':main()
