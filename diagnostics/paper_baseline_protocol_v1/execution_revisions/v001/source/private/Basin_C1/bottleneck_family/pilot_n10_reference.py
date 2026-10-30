"""Check whether public waypoint + accepted safety handles sparse N=10 traffic.

This is a diagnosis of expert-data feasibility, never a Gap1 opposing test.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import argparse
import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dev_closed_loop import load_nominal_cases
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity,uid
from shared_rollout_db.src.planner import preflight
from .collect_competence import rollout
from .scenario import Config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('diagnostics/gap_flow_competence_n10_pilot'))
    parser.add_argument('--full-one-way-only',action='store_true')
    args=parser.parse_args()
    output=args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest,cases,_=load_nominal_cases('diagnostics/gap_flow_v1/n10_wide_recovery/dataset',
                                         split='dev',final_evaluation=False)
    config=Config(**manifest['scenario_config'])
    modes=({'full10_LR':tuple(range(10))} if args.full_one_way_only else
           {'solo_LR':(0,), 'solo_RL':(1,), 'pair_LR':(0,2), 'pair_RL':(1,3)})
    planned=[];inputs=[]
    for case in cases[:4]:
        state=case.initial_state
        for mode,moving in modes.items():
            p=np.asarray(state['positions'],dtype=float)
            if mode=='full10_LR':
                p=p.copy();g=np.asarray(state['goals'],dtype=float).copy()
                p[1::2]=g[1::2]
                g[1::2]=np.asarray(state['positions'],dtype=float)[1::2]
            else:
                g=p.copy();g[list(moving)]=np.asarray(state['goals'],dtype=float)[list(moving)]
            inputs.append((case.rollout_id,state['episode_seed'],mode,moving,p,g))
            planned.append(dict(state_uid=uid('state',{'physical_fingerprint':config.physical_fingerprint,
                'positions':p.tolist(),'goals':g.tolist(),'episode_seed':state['episode_seed']}),
                eta_uid=eta_identity((0.,0.,0.))[0],
                controller_uid=uid('ctl',{'controller':'public_waypoint_v2_accepted_hard_safety_n10_pilot',
                    'mode':mode}),seed_keys=['0']))
    (output/'planned_rollouts.json').write_text(json.dumps({'requests':planned},indent=2)+'\n')
    cache=preflight(output/'planned_rollouts.json')
    (output/'cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous'] or cache['summary']['genuinely_missing'] != len(inputs):
        raise RuntimeError('preflight requires review/reuse')
    projector=CertifiedHardSafetyFilter(HardProjectionConfig())
    rows=[]
    for name,seed,mode,moving,p,g in inputs:
        result=rollout(replace(config,seed=seed,split='dev'),p,g,projector,
                       max_horizon=1400 if args.full_one_way_only else None)
        row=dict(case=name,mode=mode,moving=moving,success=result['success'],
            termination=result['termination'],steps=len(result['actions']),
            final_goal_distance=np.linalg.norm(result['states'][-1]-g,axis=1).tolist(),
            wall_active_fraction=result['wall_active_fraction'],
            mean_projection_norm=result['mean_projection_norm'])
        rows.append(row);print(json.dumps(row),flush=True)
    (output/'summary.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
