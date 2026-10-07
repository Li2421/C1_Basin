"""Classify frozen Flow + safety traces conservatively, preserving all evidence."""
from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np


ROOT=Path('diagnostics/gap_flow_safety_audit_v1')
MODES=('solo_even','solo_odd','solo_odd_remote','one_way','temporal_even_first','temporal_odd_first','opposing')


def metrics(path):
    with np.load(path,allow_pickle=False) as z:
        d={key:z[key] for key in z.files if key!='metadata_json'}
        meta=json.loads(str(z['metadata_json'].item()))
    p=d['positions'];g=d['goals'];cfg=meta['config'];dt=cfg['dt'];n=len(g)
    t=len(p)-1
    errors=np.linalg.norm(p-g[None],axis=2)
    speed=np.linalg.norm(d['u_safe'],axis=2)
    progress=-(np.diff(errors,axis=0))/dt
    gate=(np.abs(p[:,:,0])<1.5)&(np.abs(p[:,:,1])<1.)
    crossing=((p[0,:,0]<-.3)&(p[:,:,0]>.3).any(axis=0))|((p[0,:,0]>.3)&(p[:,:,0]<-.3).any(axis=0))
    active=np.arange(n) if not meta['control_mode'].startswith('solo_') else (
        np.arange(0,n,2) if meta['control_mode']=='solo_even' else np.arange(1,n,2))
    closest=float(np.linalg.norm(p,axis=2).min())
    encounter=None
    if meta['control_mode']=='opposing':
        left=np.arange(0,n,2);right=np.arange(1,n,2)
        sep=np.linalg.norm(p[:,left,None,:]-p[:,None,right,:],axis=-1)
        adjacent=(sep<.8)&(np.abs(p[:,:,0][:,left,None])<1.5)&(np.abs(p[:,:,0][:,None,right])<1.5)
        hits=np.flatnonzero(adjacent.any(axis=(1,2)))
        encounter=int(hits[0]) if len(hits) else None
    window=int(round(2/dt));hold=int(round(5/dt))
    candidates=np.zeros(t+1,dtype=bool)
    for k in range(window,t+1):
        candidates[k]=(np.max(np.abs(errors[k]-errors[k-window]))<.01 and
                       np.max(speed[k-1])<.025 and
                       np.any(errors[k]>cfg['goal_tolerance']))
    strict=None;streak=0
    for k in range(t+1):
        streak=streak+1 if candidates[k] else 0
        if streak>=hold:
            strict=k;break
    final_start=max(0,t-hold)
    pair_clear=np.asarray(d['swept_agent_clearance'])
    wall_clear=np.asarray(d['swept_wall_clearance'])
    wall_rows=n*len(d['walls'])
    pair_rows=n*(n-1)//2
    corrected=np.asarray(d['projection_norm'])
    goal_direction=(g[None]-p[:-1])/np.maximum(errors[:-1,:,None],1e-12)
    flow_goal=np.sum(d['u_flow']*goal_direction,axis=2)
    safe_goal=np.sum(d['u_safe']*goal_direction,axis=2)
    recent_speed=speed[final_start:,active]
    recent_u=d['u_safe'][final_start:,active]
    if len(recent_u)>1:
        turn=np.sum(recent_u[1:]*recent_u[:-1],axis=2)
        flips=(turn<-.5*recent_speed[1:]*recent_speed[:-1])&(
            recent_speed[1:]>.05)&(recent_speed[:-1]>.05)
        reversal_fraction=float(flips.mean())
    else:
        reversal_fraction=None
    def span_mean(values,start,end):
        return float(np.asarray(values)[start:end].mean()) if end>start else None
    early_end=encounter if encounter is not None else None
    return dict(rollout_id=meta['rollout_id'],mode=meta['control_mode'],N=n,
        termination=meta['termination'],steps=t,seconds=t*dt,
        initial_mean_goal_distance=float(errors[0,active].mean()),
        final_mean_goal_distance=float(errors[-1,active].mean()),
        minimum_mean_goal_distance=float(errors[:,active].mean(axis=1).min()),
        mean_agent_travel=float(np.linalg.norm(np.diff(p,axis=0),axis=2).sum(axis=0)[active].mean()),
        active_agents_enter_gate=int(gate[:,active].any(axis=0).sum()),
        active_agents_cross_gate=int(crossing[active].sum()),
        closest_agent_to_gate_center=closest,
        first_opposing_gate_encounter_step=encounter,
        first_strict_tt_stall_step=strict,
        strict_tt_stall_after_encounter=bool(encounter is not None and strict is not None and strict>encounter),
        minimum_swept_wall_clearance=float(wall_clear.min()) if len(wall_clear) else None,
        minimum_center_to_wall_distance=(float(wall_clear.min()+cfg['agent_radius']+
            cfg['wall_radius']+cfg['wall_collision_margin']) if len(wall_clear) else None),
        minimum_swept_agent_clearance=float(pair_clear.min()) if len(pair_clear) else None,
        projection_active_fraction=float((corrected>1e-6).mean()) if len(corrected) else None,
        mean_projection_norm=float(corrected.mean()) if len(corrected) else None,
        wall_constraint_active_fraction=float((d['active_wall_count']>0).mean()) if t else None,
        pair_constraint_active_fraction=float((d['active_pair_count']>0).mean()) if t else None,
        speed_constraint_active_fraction=float((d['active_speed_count']>0).mean()) if t else None,
        mean_active_wall_row_fraction=float(d['active_wall_count'].mean()/wall_rows) if t else None,
        mean_active_pair_row_fraction=float(d['active_pair_count'].mean()/pair_rows) if t else None,
        mean_active_speed_ball_fraction=float(d['active_speed_count'].mean()/n) if t else None,
        final_5s_mean_speed=span_mean(speed[:,active],final_start,t),
        final_5s_mean_progress_rate=span_mean(progress[:,active],final_start,t),
        # A temporally masked control changes the active subset mid-episode;
        # a static all-agent Flow-vs-safe average would confound the mask.
        final_5s_mean_flow_goal_component=(None if meta['control_mode'].startswith('temporal_') else
                                           span_mean(flow_goal[:,active],final_start,t)),
        final_5s_mean_safe_goal_component=(None if meta['control_mode'].startswith('temporal_') else
                                           span_mean(safe_goal[:,active],final_start,t)),
        final_5s_mean_projection_norm=span_mean(corrected,final_start,t),
        final_5s_high_speed_reversal_fraction=reversal_fraction,
        final_5s_wall_constraint_active_fraction=float((d['active_wall_count'][final_start:]>0).mean()) if t else None,
        final_5s_pair_constraint_active_fraction=float((d['active_pair_count'][final_start:]>0).mean()) if t else None,
        final_5s_active_wall_row_fraction=float(d['active_wall_count'][final_start:].mean()/wall_rows) if t else None,
        final_5s_active_pair_row_fraction=float(d['active_pair_count'][final_start:].mean()/pair_rows) if t else None,
        pre_encounter_mean_speed=span_mean(speed[:,active],0,early_end) if early_end is not None else None,
        post_encounter_mean_speed=span_mean(speed[:,active],early_end,t) if early_end is not None else None,
        pre_encounter_mean_progress_rate=span_mean(progress[:,active],0,early_end) if early_end is not None else None,
        post_encounter_mean_progress_rate=span_mean(progress[:,active],early_end,t) if early_end is not None else None,
        pre_encounter_mean_projection_norm=span_mean(corrected,0,early_end) if early_end is not None else None,
        post_encounter_mean_projection_norm=span_mean(corrected,early_end,t) if early_end is not None else None,
        pre_encounter_active_wall_row_fraction=(span_mean(d['active_wall_count'],0,early_end)/wall_rows
                                                if early_end is not None and early_end>0 else None),
        post_encounter_active_wall_row_fraction=(span_mean(d['active_wall_count'],early_end,t)/wall_rows
                                                 if early_end is not None else None),
        pre_encounter_active_pair_row_fraction=(span_mean(d['active_pair_count'],0,early_end)/pair_rows
                                                if early_end is not None and early_end>0 else None),
        post_encounter_active_pair_row_fraction=(span_mean(d['active_pair_count'],early_end,t)/pair_rows
                                                 if early_end is not None else None),
        passive_max_displacement=(float(np.linalg.norm(p[:,1::2]-p[0,1::2],axis=2).max()) if meta['control_mode']=='solo_even' else
                                  float(np.linalg.norm(p[:,0::2]-p[0,0::2],axis=2).max()) if meta['control_mode'] in ('solo_odd','solo_odd_remote') else None),
        trace=str(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    root=parser.parse_args().root
    all_rows=[]
    by_n={}
    for n in (2,10,50):
        folder=root/f'n{n}'
        if not folder.exists():continue
        by_mode={}
        for mode in MODES:
            summary=folder/mode/'summary.json'
            if not summary.exists():continue
            source=json.loads(summary.read_text())
            rows=[]
            for entry in source['rollouts']:
                row=metrics(folder/mode/'traces'/f"{entry['rollout_id']}.npz")
                row['numerical_error']=entry['numerical_error']
                rows.append(row)
            by_mode[mode]={row['rollout_id'].removesuffix('_'+mode):row for row in rows}
            all_rows.extend(rows)
        by_n[n]=by_mode
    for row in all_rows:
        if row['termination']=='success':
            category='success';reason='all agents reached goals safely'
        elif row['termination']=='collision':
            category='C';reason='simulator swept collision'
        elif row['termination']=='numerical_failure':
            category='D';reason='projection or numerical exception'
        elif row['termination']=='timeout':
            category='B';reason='timeout without certified opposing-interaction deadlock'
            if row['mode']=='opposing' and row['first_opposing_gate_encounter_step'] is not None:
                key=row['rollout_id'].removesuffix('_'+row['mode'])
                controls=by_n[row['N']]
                if row['N']==2:
                    competent=all(controls.get(mode,{}).get(key,{}).get('termination')=='success'
                                  for mode in ('solo_even','solo_odd_remote'))
                else:
                    competent=all(controls.get(mode,{}).get(key,{}).get('termination')=='success'
                                  for mode in ('one_way','temporal_even_first','temporal_odd_first'))
                if competent and row['strict_tt_stall_after_encounter'] and row['active_agents_enter_gate']>=2:
                    category='A';reason='interaction then strict 2s-window/5s-hold stall; matched controls succeeded'
                else:
                    category='E';reason=('opposing gate encounter observed, but clean deadlock is unproven: '
                        'matched controls lack competence and/or no strict persistent stall')
        else:
            category='E';reason=f'unrecognized termination {row["termination"]}'
        row['failure_category']=category;row['failure_reason']=reason
    (root/'all_rollout_diagnostics.json').write_text(json.dumps(all_rows,indent=2)+'\n')
    for n in by_n:
        rows=[r for r in all_rows if r['N']==n]
        print(n,[(mode,{cat:sum(r['failure_category']==cat for r in rows if r['mode']==mode)
                    for cat in ('success','A','B','C','D','E')})
                 for mode in MODES if any(r['mode']==mode for r in rows)],flush=True)


if __name__=='__main__':main()
