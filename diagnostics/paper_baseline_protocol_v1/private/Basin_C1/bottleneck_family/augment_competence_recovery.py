"""Add paired, non-opposing goal-local recovery continuations to Gap1 data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dataset import DatasetWriter, RecoveryAudit, Trajectory
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .collect_competence import rollout
from .environment import BottleneckEnv
from .flow_dataset import GapFlowScenario
from .scenario import Config


OFFSETS = ((+.60, +.18), (+.40, -.25), (-.45, +.24))


def augment(source, output):
    source,output=Path(source),Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    config=Config(**manifest['scenario_config'])
    writer=DatasetWriter(output/'dataset',GapFlowScenario(config),scenario_config=asdict(config))
    candidates=[]; rejected=[];modes_seen=set()
    for row in manifest['files']:
        with np.load(source/row['file'],allow_pickle=False) as data:
            states=data['states']; observations=data['observations'];actions=data['actions']
            initial=json.loads(str(data['initial_state_json'].item()))
            metadata=json.loads(str(data['metadata_json'].item()))
        writer.add(Trajectory(row['rollout_id'],row['split'],'nominal',initial,
                              states,observations,actions,metadata))
        if row['split']=='test':
            continue
        goals=np.asarray(initial['goals'],dtype=float)
        mode=initial['mode'];direction=1 if initial['direction']=='LR' else -1
        modes_seen.add(mode)
        swap=int(row['rollout_id'].endswith('swap1'))
        if 'active_agents' in initial:
            moving=tuple((int(agent)^swap) for agent in initial['active_agents'])
        else:
            moving=(swap,) if mode=='solo' else (0,1)
        for agent in moving:
            for offset_index,(forward,lateral) in enumerate(OFFSETS):
                p=states[-2].copy()
                p[agent]=goals[agent]+np.array((direction*forward,lateral))
                # If the local perturbation would place a disk outside free
                # space or overlap a parked peer, do not train on it.
                try:
                    env=BottleneckEnv(Config(**{**config.to_dict(),
                        'seed':initial['episode_seed'],'split':row['split']}))
                    env.reset(p,goals)
                except ValueError:
                    rejected.append(dict(parent=row['rollout_id'],agent=agent,
                                         offset_index=offset_index,reason='invalid_physical_state'))
                    continue
                name=f"{row['rollout_id']}_recovery_a{agent}_v{offset_index}"
                candidates.append((name,row,initial,p,goals,agent,offset_index))
    requests=[]
    for name,row,initial,p,g,agent,offset_index in candidates:
        requests.append(dict(state_uid=uid('state',{
            'physical_fingerprint':config.physical_fingerprint,
            'positions':p.tolist(),'goals':g.tolist(),
            'episode_seed':initial['episode_seed'],'parent':row['rollout_id']}),
            eta_uid=eta_identity((0.,0.,0.))[0],
            controller_uid=uid('ctl',{'controller':'gate_waypoint_v2_goal_recovery_teacher',
                'agent':agent,'offset_index':offset_index}),
            seed_keys=[str(initial['episode_seed'])]))
    (output/'planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    cache=preflight(output/'planned_rollouts.json')
    (output/'cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous'] or cache['summary']['genuinely_missing'] != len(candidates):
        raise RuntimeError('preflight requires review/reuse before recovery rollouts')
    projector=CertifiedHardSafetyFilter(HardProjectionConfig())
    counts={};failed=[]
    for n,(name,row,initial,p,g,agent,offset_index) in enumerate(candidates):
        episode_config=Config(**{**config.to_dict(),
            'seed':initial['episode_seed'],'split':row['split']})
        result=rollout(episode_config,p,g,projector)
        if not result['success']:
            failed.append(dict(name=name,termination=result['termination']))
            continue
        recovery=RecoveryAudit(row['rollout_id'],row['split'],int(row['length'])-1,
                               None,None,None,int(offset_index),True)
        writer.add(Trajectory(name,row['split'],'uniform_recovery',
            {'positions':p,'goals':g,'episode_seed':initial['episode_seed'],
             'split':row['split'],'mode':initial['mode'],'direction':initial['direction'],
             'source_rollout_id':row['rollout_id'],'offset_index':offset_index},
            result['states'],result['observations'],result['actions'],
            {'success':True,'terminal_reason':'success','mode':initial['mode'],
             'direction':initial['direction'],'expert':'public_gap_waypoint_v2_plus_accepted_hard_safety',
             'min_swept_wall_clearance':result['min_swept_wall_clearance'],
             'mean_projection_norm':result['mean_projection_norm']},recovery))
        key=f"{row['split']}_{initial['mode']}_{initial['direction']}"
        bucket=counts.setdefault(key,{'trajectories':0,'transitions':0})
        bucket['trajectories']+=1;bucket['transitions']+=len(result['actions'])
        if (n+1)%200==0:
            print(json.dumps({'recoveries_done':n+1,'counts':counts}),flush=True)
    for split in ('train','dev'):
        for mode in sorted(modes_seen):
            left=counts.get(f'{split}_{mode}_LR',{'trajectories':0,'transitions':0})
            right=counts.get(f'{split}_{mode}_RL',{'trajectories':0,'transitions':0})
            for field in ('trajectories','transitions'):
                if max(left[field],right[field]) and abs(left[field]-right[field])/max(left[field],right[field])>.03:
                    raise RuntimeError(f'unbalanced recovery data {split} {mode} {field}: {left} vs {right}')
    report={'schema':'gap1_goal_recovery_augmentation_v1','offsets':OFFSETS,
            'counts':counts,'invalid_candidates':rejected,'failed_rollouts':failed,
            'preflight':cache['summary'],'source_manifest':str(source/'manifest.json')}
    writer.finalize(extra_report=report)
    (output/'augmentation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(augment(args.source,args.output),indent=2),flush=True)


if __name__=='__main__':main()
