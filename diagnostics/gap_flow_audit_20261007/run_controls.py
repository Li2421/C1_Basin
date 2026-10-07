"""Frozen-checkpoint Gap1 competence controls with the published MC16 evaluator.

one_way: every existing start/goal is reused, but both groups travel left-to-right.
temporal_even_first / temporal_odd_first: original tasks, opposite group held at
its original start until the first group reaches all goals. No route commands.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import jax
import numpy as np

from bottleneck_family.environment import BottleneckEnv
from bottleneck_family.observation import policy_observation
from bottleneck_family.scenario import Config
from new_benchmark_common.dev_closed_loop import load_nominal_cases
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight


ROOT = Path('diagnostics/gap_flow_v1')
OUT = Path('diagnostics/gap_flow_audit_20261007/controls')
SPECS = ((2, 'n2_recovery_wide'), (10, 'n10_wide_recovery'),
         (50, 'n50_wide_recovery'))
MODES = ('one_way', 'temporal_even_first', 'temporal_odd_first',
         'park_opposite_even', 'park_opposite_odd')


def modified_state(state, mode):
    positions = np.asarray(state['positions'], dtype=float).copy()
    goals = np.asarray(state['goals'], dtype=float).copy()
    if mode == 'one_way':
        # Reuse the original point sets: odd agents reverse their trip so all
        # robots cross the same physical gap in the same direction.
        positions[1::2] = np.asarray(state['goals'])[1::2]
        goals[1::2] = np.asarray(state['positions'])[1::2]
    elif mode == 'park_opposite_even':
        goals[1::2] = positions[1::2]
    elif mode == 'park_opposite_odd':
        goals[0::2] = positions[0::2]
    return positions, goals


def main():
    requested = sys.argv[1:] or list(MODES)
    if any(mode not in MODES for mode in requested):
        raise ValueError(requested)
    for n, name in SPECS:
        root = ROOT/name
        manifest, cases, audit = load_nominal_cases(root/'dataset', split='test', final_evaluation=True)
        config = Config(**manifest['scenario_config'])
        ckpt = root/'train/best.pkl'
        agent, _ = load_checkpoint(ckpt, expected_environment_fingerprint=manifest['environment_fingerprint'])
        checkpoint_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        for mode in requested:
            folder = OUT/f'n{n}'/mode
            if folder.exists() and any(folder.iterdir()):
                print('Already populated:', folder, flush=True); continue
            folder.mkdir(parents=True, exist_ok=True)
            controller_uid = uid('ctl', {'flow_sha256':checkpoint_sha,
                'protocol':'gap_joint_macflow_stage1_v1', 'safety':'none',
                'samples_per_step':16, 'control':mode})
            requests = []
            for case in cases:
                p,g = modified_state(case.initial_state, mode)
                requests.append(dict(state_uid=uid('state', {'physical_fingerprint':config.physical_fingerprint,
                    'positions':p.tolist(), 'goals':g.tolist(), 'episode_seed':case.initial_state['episode_seed'],
                    'control':mode}), eta_uid=eta_identity((0.,0.,0.))[0],
                    controller_uid=controller_uid, seed_keys=['17']))
            plan=folder/'planned_rollouts.json'
            plan.write_text(json.dumps({'requests':requests},indent=2)+'\n')
            cached=preflight(plan)
            (folder/'cache_preflight.json').write_text(json.dumps(cached,indent=2)+'\n')
            if cached['summary']['ambiguous']:
                raise RuntimeError('ambiguous cache evidence')
            rows=[]
            for index,case in enumerate(cases):
                state=case.initial_state
                p,g=modified_state(state,mode)
                env=BottleneckEnv(replace(config,seed=state['episode_seed'],split='test'))
                env.reset(p,g)
                initial_hash=env.initial_state_sha256()
                episode_key=jax.random.fold_in(jax.random.PRNGKey(17),index)
                positions=[env.positions.copy()];clearances=[];terminal='timeout'
                phase=0
                first = np.arange(0 if mode=='temporal_even_first' else 1,n,2)
                second = np.arange(1 if mode=='temporal_even_first' else 0,n,2)
                phase_transition=None
                for step in range(config.max_steps):
                    obs=(policy_observation(env) if agent.config['obs_dim']==8 else
                         env.observation()['agents'].astype(np.float32))
                    sampled=np.asarray(sample_bounded_actions(agent,
                        np.repeat(obs[None],16,axis=0),
                        jax.random.fold_in(episode_key,step)),dtype=float)
                    action=sampled.mean(axis=0)
                    norms=np.linalg.norm(action,axis=1,keepdims=True)
                    action*=np.minimum(1.0,config.max_speed/np.maximum(norms,1e-30))
                    if mode.startswith('temporal_'):
                        if phase==0 and np.all(np.linalg.norm(env.positions[first]-env.goals[first],axis=1)
                                               <=config.goal_tolerance):
                            phase=1;phase_transition=step
                        action[second if phase==0 else first]=0
                    elif mode=='park_opposite_even':
                        action[1::2]=0
                    elif mode=='park_opposite_odd':
                        action[0::2]=0
                    _,done,info=env.step(action,diagnose_stalls=False)
                    positions.append(env.positions.copy());clearances.append(info['swept_clearance'])
                    terminal=info['termination']
                    if done:break
                if terminal=='running':terminal='timeout'
                remaining=np.linalg.norm(env.positions-env.goals,axis=1)
                row=dict(rollout_id=case.rollout_id,mode=mode,termination=terminal,
                    success=terminal=='success',steps=len(clearances),
                    collision=bool(env.collided),individual_goal_fraction=float(np.mean(remaining<=config.goal_tolerance)),
                    first_group_completed=phase==1 if mode.startswith('temporal_') else None,
                    phase_transition_step=phase_transition,
                    min_swept_clearance=float(min(clearances)))
                rows.append(row);print(n,mode,json.dumps(row),flush=True)
                trace_dir=folder/'traces';trace_dir.mkdir(exist_ok=True)
                video_meta=dict(batch_id='gap_flow_competence_audit_20261007',scenario='bottleneck_family',
                    rollout_id=f'{case.rollout_id}_{mode}',controller=f'joint_macflow_stage1_mc16_raw_{mode}',
                    checkpoint=f'{ckpt} sha256={checkpoint_sha}',seed=state['episode_seed'],
                    controller_rng_seed=17,source_record=str(case.archive_path),
                    initial_state_sha256=initial_hash,config=env.config.to_dict(),start_step=0,
                    episode_steps=len(clearances),complete=True,termination=terminal,
                    collision=bool(env.collided),safety_enabled=False,control_mode=mode,
                    phase_transition_step=phase_transition)
                np.savez_compressed(trace_dir/f'{case.rollout_id}.npz',positions=np.asarray(positions),
                    steps=np.arange(len(positions)),goals=env.goals,walls=env.walls,
                    swept_clearance=np.asarray(clearances),metadata_json=np.asarray(json.dumps(video_meta)))
            (folder/'summary.json').write_text(json.dumps(dict(mode=mode,N=n,
                checkpoint_sha256=checkpoint_sha,dataset_manifest_sha256=audit['manifest_sha256'],
                samples_per_step=16,safety_enabled=False,rollouts=rows),indent=2)+'\n')


if __name__=='__main__':main()
