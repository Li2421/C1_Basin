"""Predeclared score orientations, state-disjoint descriptive validation, eta prior."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
from analyze_geometry import HERE,OLD,readcsv,writecsv

def ap(y,score):
    y=np.asarray(y,bool);s=np.asarray(score,float)
    if not y.any():return None
    order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order]
    cuts=np.r_[np.flatnonzero(np.diff(ss)!=0)+1,len(ss)]
    start=0;seen=0;out=0.
    for stop in cuts:
        hits=int(yy[start:stop].sum());seen+=hits
        out+=hits/np.sum(y)*seen/stop;start=stop
    return float(out)

def bootstrap(rows,metric,seed):
    by=defaultdict(list)
    for r in rows:by[r['uid']].append(r)
    states=sorted(by);rng=np.random.default_rng(seed);out=[]
    for _ in range(1000):
        z=[r for u in rng.choice(states,len(states),replace=True) for r in by[u]]
        v=ap([r['target'] for r in z],[r[metric] for r in z])
        if v is not None:out.append(v)
    return np.quantile(out,[.025,.975]).tolist() if out else [None,None]

def within_eta_p(rows,metric,seed):
    by=defaultdict(list)
    for r in rows:by[r['eta_index']].append(r)
    observed=0.
    for z in by.values():
        y=np.array([r['target'] for r in z]);s=np.array([r[metric] for r in z]);observed+=float(np.dot(y,s-s.mean()))
    rng=np.random.default_rng(seed);extreme=0
    for _ in range(2000):
        stat=0.
        for z in by.values():
            y=rng.permutation([r['target'] for r in z]);s=np.array([r[metric] for r in z]);stat+=float(np.dot(y,s-s.mean()))
        extreme+=stat>=observed-1e-12
    return (extreme+1)/2001,observed

def main():
    action=readcsv(HERE/'ring_directional_mechanism.csv')+readcsv(HERE/'toy_yielding_mechanism.csv')
    traj={(r['scene'],r['uid'],int(r['eta_index'])):r for r in readcsv(HERE/'trajectory_geometry.csv')}
    prior=json.loads((OLD/'eta_prior_baseline.json').read_text())
    rows=[]
    for a in action:
        sc=a['scene'];uid=a['uid'];j=int(a['eta_index']);t=traj[(sc,uid,j)]
        z=dict(scene=sc,uid=uid,eta_index=j,split=a['split'],b15_outcome=a['b15_outcome'],
               target=int(a['b15_outcome']==('gain' if sc=='ring_exchange' else 'loss')),
               seed0_trajectory_valid=int(t['old_trajectory_steps'] and t['new_trajectory_steps']),
               score_raw_action_l2=float(a['action_l2']))
        if sc=='ring_exchange':
            z.update(ambiguous_agents=float(a['ambiguous_agents']),
                score_shortest_signed_finite=float(a['shortest_action_difference']),
                score_nominal_sign_finite=float(a['nominal_sign_action_difference']),
                score_absolute_angular_finite=float(a['angular_abs_action_difference']),
                score_projection_retention=float(a['projection_angular_change_advantage']),
                score_shortest_signed_local=float(a['shortest_local_differential']),
                score_early_H20_absolute_angular=float(t['H20_absolute_travel_delta']),
                score_early_H80_absolute_angular=float(t['H80_absolute_travel_delta']),
                score_early_H20_shortest_signed=float(t['H20_shortest_signed_net_delta']),
                score_early_H80_shortest_signed=float(t['H80_shortest_signed_net_delta']))
        else:
            z.update(score_goal_finite=-float(a['goal_mean_difference']),
                score_clearance_finite=-float(a['clearance_hdot_difference']),
                score_upward_capacity_finite=-float(a['upward_capacity_difference']),
                score_lateral_asymmetry_finite=-float(a['lateral_asymmetry_difference']),
                score_lateral_asymmetry_local=-float(a['lateral_asymmetry_local_difference']),
                score_goal_local=-float(a['goal_local_differential']),
                score_early_H20_worst_goal=-float(t['H20_worst_goal_progress_delta']) if t['H20_worst_goal_progress_delta'] else None,
                score_early_H80_worst_goal=-float(t['H80_worst_goal_progress_delta']) if t['H80_worst_goal_progress_delta'] else None,
                score_early_H20_bay_rise=-float(t['H20_max_y_inside_bottleneck_delta']),
                score_early_H80_bay_rise=-float(t['H80_max_y_inside_bottleneck_delta']),
                score_early_H20_lateral_separation=-float(t['H20_max_lateral_separation_delta']),
                score_early_H80_lateral_separation=-float(t['H80_max_lateral_separation_delta']))
        z['score_eta_prior']=float(prior[f"{sc}:heldout_test"]['discovery_eta_rates'][str(j)])
        rows.append(z)
    writecsv(HERE/'heldout_validation.csv',rows)
    out=[]
    for sc in ('ring_exchange','toy_give_way'):
        for split in ('discovery','validation','heldout_test'):
            z=[r for r in rows if r['scene']==sc and r['split']==split]
            for metric in [k for k in z[0] if k.startswith('score_')]:
                q=[r for r in z if r[metric] is not None and np.isfinite(r[metric])]
                y=np.array([r['target'] for r in q],bool);s=np.array([r[metric] for r in q],float)
                p,stat=within_eta_p(q,metric,20261012) if split=='heldout_test' else (None,None)
                out.append(dict(scene=sc,split=split,metric=metric,n=len(q),positives=int(y.sum()),
                    prevalence=float(y.mean()),average_precision=ap(y,s),
                    state_bootstrap_ap_95=bootstrap(q,metric,20261013) if split=='heldout_test' else None,
                    within_eta_permutation_p=p,within_eta_centered_statistic=stat,
                    median_positive=float(np.median(s[y])) if y.any() else None,
                    median_negative=float(np.median(s[~y])) if (~y).any() else None))
    writecsv(HERE/'heldout_metric_summary.csv',out)
    for r in out:
        if r['split']=='heldout_test':print(r['scene'],r['metric'],'AP',round(r['average_precision'],3),
                                        'eta-p',round(r['within_eta_permutation_p'],3))

if __name__=='__main__':main()
