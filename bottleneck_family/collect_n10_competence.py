"""Balanced sparse-to-moderate N=10 navigation data, without opposing flow.

The reference is the public gate waypoint plus the unchanged safety filter.
Only solo, two-agent same-direction and five-agent same-direction tasks are
generated.  No release ordering or opposing priority appears in the data.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dataset import DatasetWriter, Trajectory
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .collect_competence import rollout
from .flow_dataset import GapFlowScenario
from .scenario import Config, build_instance


def transform(array, *, mirror, swap):
    value=np.asarray(array).copy()
    if mirror:
        value[...,0]*=-1
    if swap:
        # Swap each fixed even/odd row pair.  This is a relabeling, never a
        # scheduling rule; simulator pair constraints are permutation invariant.
        index=np.arange(value.shape[-2]) ^ 1
        value=value[...,index,:]
    return value


def collect(output, *, seed=314170, counts=None):
    counts=counts or {'train':24,'dev':6,'test':12}
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True)
    old=json.loads(Path('diagnostics/gap_flow_v1/n10_wide_recovery/dataset/manifest.json').read_text())
    config=Config(**old['scenario_config'])
    if config.num_agents!=10 or config.openings[0][0][1]!=.62:
        raise ValueError('N=10 exact reported Gap1 geometry required')
    rng=np.random.default_rng(seed)
    bases=[];requests=[]
    for split in ('train','dev','test'):
        for index in range(counts[split]):
            episode_seed=int(rng.integers(0,2**31-1))
            episode_config=replace(config,seed=episode_seed,split=split)
            instance=build_instance(episode_config)
            modes={
                'solo':(2*(index%5),),
                'pair':(2*(index%5),2*((index+1)%5)),
                'group5':(0,2,4,6,8),
            }
            for mode,moving in modes.items():
                positions=instance.positions.copy()
                goals=positions.copy();goals[list(moving)]=instance.goals[list(moving)]
                bases.append((split,index,episode_config,mode,moving,positions,goals))
                for mirror in (False,True):
                    for swap in (False,True):
                        p=transform(positions,mirror=mirror,swap=swap)
                        g=transform(goals,mirror=mirror,swap=swap)
                        requests.append(dict(state_uid=uid('state',{
                            'physical_fingerprint':config.physical_fingerprint,
                            'positions':p.tolist(),'goals':g.tolist(),
                            'episode_seed':episode_seed,'split':split}),
                            eta_uid=eta_identity((0.,0.,0.))[0],
                            controller_uid=uid('ctl',{
                                'controller':'public_gate_waypoint_v2_accepted_hard_safety_n10',
                                'mode':mode,'mirror':mirror,'swap':swap}),
                            seed_keys=[str(episode_seed)]))
    (output/'planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    cache=preflight(output/'planned_rollouts.json')
    (output/'cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous'] or cache['summary']['genuinely_missing']!=len(requests):
        raise RuntimeError('preflight requires review/reuse')
    projector=CertifiedHardSafetyFilter(HardProjectionConfig())
    writer=DatasetWriter(output/'dataset',GapFlowScenario(config),scenario_config=asdict(config))
    report={'schema':'gap1_n10_navigation_dataset_v1','seed':seed,'requested_base_counts':counts,
            'accepted':{},'teacher_failures':[],'preflight':cache['summary'],
            'teacher':'public_gate_waypoint_v2_plus_unchanged_hard_safety',
            'maximum_teacher_generation_steps':1400}
    for k,(split,index,episode_config,mode,moving,p,g) in enumerate(bases):
        base=rollout(episode_config,p,g,projector,max_horizon=1400)
        if not base['success']:
            report['teacher_failures'].append(dict(split=split,index=index,mode=mode,
                termination=base['termination'],steps=len(base['actions']),
                final_goal_distances=np.linalg.norm(base['states'][-1]-g,axis=1).tolist()))
            continue
        for mirror in (False,True):
            for swap in (False,True):
                pp=transform(p,mirror=mirror,swap=swap)
                gg=transform(g,mirror=mirror,swap=swap)
                commands=transform(base['actions'],mirror=mirror,swap=swap)
                replay=rollout(episode_config,pp,gg,None,actions=commands)
                if not replay['success'] or len(replay['actions'])!=len(commands):
                    raise RuntimeError('physical mirrored/permuted simulator replay failed')
                direction='RL' if mirror else 'LR'
                name=f'{split}_n10_nav_{index:04d}_{mode}_{direction}_swap{int(swap)}'
                writer.add(Trajectory(name,split,'nominal',
                    {'positions':pp,'goals':gg,'episode_seed':episode_config.seed,
                     'split':split,'mode':mode,'direction':direction,'active_agents':moving},
                    replay['states'],replay['observations'],commands,
                    {'success':True,'terminal_reason':'success','mode':mode,
                     'direction':direction,'mirror':mirror,'swap':swap,
                     'teacher':'public_gate_waypoint_v2_plus_accepted_hard_safety',
                     'min_swept_wall_clearance':replay['min_swept_wall_clearance'],
                     'original_projection_norm':base['mean_projection_norm']}))
                key=f'{split}_{mode}_{direction}'
                bucket=report['accepted'].setdefault(key,{'trajectories':0,'transitions':0})
                bucket['trajectories']+=1;bucket['transitions']+=len(commands)
        if (k+1)%12==0:
            print(json.dumps({'base_cases_done':k+1,'accepted':report['accepted'],
                              'teacher_failures':len(report['teacher_failures'])}),flush=True)
    for split in counts:
        for mode in ('solo','pair','group5'):
            if report['accepted'].get(f'{split}_{mode}_LR')!=report['accepted'].get(f'{split}_{mode}_RL'):
                raise RuntimeError(f'directional imbalance in {split} {mode}')
    writer.finalize(extra_report=report)
    (output/'collection_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train',type=int,default=24)
    parser.add_argument('--dev',type=int,default=6)
    parser.add_argument('--test',type=int,default=12)
    args=parser.parse_args()
    report=collect(args.output,counts={'train':args.train,'dev':args.dev,'test':args.test})
    print(json.dumps({k:v for k,v in report.items() if k!='teacher_failures'},indent=2),flush=True)


if __name__=='__main__':main()
