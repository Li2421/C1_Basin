"""Per-rollout post-freeze Gap1 mechanism audit with explicit detector caveat."""
from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np

from diagnostics.gap_flow_safety_audit_v1.analyze import metrics


ROOT=Path('diagnostics/gap_flow_competence_v3_recovery/frozen_eval')


def analyze(root=ROOT):
    root=Path(root)
    controls={}
    for mode in ('solo_even','solo_odd_remote','one_way'):
        source=json.loads((root/mode/'summary.json').read_text())
        controls[mode]={row['rollout_id']:row for row in source['rollouts']}
    source=json.loads((root/'opposing/summary.json').read_text())
    rows=[]
    for entry in source['rollouts']:
        name=entry['rollout_id']
        path=root/'opposing/traces'/f'{name}.npz'
        row=metrics(path)
        with np.load(path,allow_pickle=False) as data:
            p=data['positions'];g=data['goals'];flow=data['u_flow'];safe=data['u_safe']
            pair=data['active_pair_count'];wall=data['active_wall_count']
            correction=data['projection_norm']
        dt=.05
        window=min(len(safe),int(round(20/dt)))
        errors=np.linalg.norm(p-g[None],axis=2)
        unit=(g[None]-p[:-1])/np.maximum(errors[:-1,:,None],1e-12)
        final_p=p[-window-1:]
        extent=np.ptp(final_p,axis=0)
        net_progress=float((errors[-window-1].sum()-errors[-1].sum())/len(g))
        final_pair=float(np.mean(pair[-window:]>0))
        final_wall=float(np.mean(wall[-window:]>0))
        flow_goal=float(np.mean(np.sum(flow[-window:]*unit[-window:],axis=2)))
        safe_goal=float(np.mean(np.sum(safe[-window:]*unit[-window:],axis=2)))
        matched={mode:bool(controls[mode][name]['success']) for mode in controls}
        row.update(matched_controls=matched,
            final_20s_mean_goal_progress_m=net_progress,
            final_20s_absolute_mean_goal_progress_m=abs(net_progress),
            final_20s_max_coordinate_extent_m=float(extent.max()),
            final_20s_agent_extents_m=extent.tolist(),
            final_20s_pair_active_fraction=final_pair,
            final_20s_wall_active_fraction=final_wall,
            final_20s_mean_flow_goal_component=flow_goal,
            final_20s_mean_safe_goal_component=safe_goal,
            final_20s_mean_projection_norm=float(np.mean(correction[-window:])),
            legacy_tt_strict_deadlock_detected=row['first_strict_tt_stall_step'] is not None,
            simulator_deadlock_event=False)
        # The simulator reports timeout.  This is an *offline scientific*
        # classification, not a replacement event or a changed TT detector.
        if entry['collision'] or entry['termination']=='collision':
            category='C';reason='swept simulator collision'
        elif entry['numerical_error'] or entry['termination']=='numerical_failure':
            category='D';reason='projection/solver numerical failure'
        elif entry['success']:
            category='success';reason='all goals reached safely'
        elif (row['first_opposing_gate_encounter_step'] is not None and
              row['active_agents_enter_gate']==2 and
              matched['solo_even'] and matched['solo_odd_remote'] and
              row['pre_encounter_mean_progress_rate']>.1 and
              row['post_encounter_mean_progress_rate']<.01 and
              abs(net_progress)<.05 and extent.max()<.25 and final_pair>.5):
            category='A';reason=('opposing gate encounter, matched solo competence, '
                'safe pair-constrained bounded gridlock with <0.05 m net '
                'mean-goal progress over the final 20 s; legacy TT rest '
                'criterion did not trigger because of bounded chattering')
        else:
            category='E';reason=('timeout did not satisfy persistent bounded '
                'no-progress evidence; inspect continued motion separately')
        row['failure_category']=category;row['failure_reason']=reason
        rows.append(row)
    result={'schema':'gap1_frozen_n2_mechanism_audit_v1',
            'checkpoint_sha256':source['checkpoint_sha256'],
            'offline_categories_are_not_simulator_events':True,
            'legacy_tt_detector_unchanged':True,'rollouts':rows}
    target=root/'all_rollout_diagnostics.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'counts':{name:sum(r['failure_category']==name for r in rows)
                                for name in ('success','A','B','C','D','E')},
                      'strict_tt_hits':sum(r['legacy_tt_strict_deadlock_detected'] for r in rows)},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    analyze(parser.parse_args().root)
