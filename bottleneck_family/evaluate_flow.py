"""Frozen closed-loop evaluation of raw joint MACFlow policy."""
from __future__ import annotations
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
from new_benchmark_common.dev_closed_loop import load_nominal_cases
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .scenario import Config
from .environment import BottleneckEnv
from .observation import policy_observation


def run(dataset, checkpoint, output, *, seed=0, max_cases=None, max_steps=None,
        safety=False, samples_per_step=1, split='dev', final_evaluation=False):
    if samples_per_step < 1:
        raise ValueError("samples_per_step must be positive")
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest,cases,audit=load_nominal_cases(dataset,split=split,
                                            final_evaluation=final_evaluation)
    if max_cases is not None:
        cases=cases[:max_cases]
    config=Config(**manifest['scenario_config'])
    agent,metadata=load_checkpoint(checkpoint,expected_environment_fingerprint=manifest['environment_fingerprint'])
    if agent.config['num_agents'] != config.num_agents:
        raise ValueError('checkpoint/plant N mismatch')
    fingerprint=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    controller_uid=uid('ctl',{'flow_sha256':fingerprint,'protocol':'gap_joint_macflow_stage1_v1',
                              'safety':'hard_projection' if safety else 'none',
                              'samples_per_step':samples_per_step})
    projector = None
    if safety:
        from shared_control.hard_projection import HardSafetyFilter
        projector=HardSafetyFilter()
    requests=[]
    for case in cases:
        state=case.initial_state
        state_uid=uid('state',{'physical_fingerprint':config.physical_fingerprint,'state':state})
        requests.append(dict(state_uid=state_uid,eta_uid=eta_identity((0.,0.,0.))[0],
                             controller_uid=controller_uid,seed_keys=[str(seed)]))
    plan=output/'planned_rollouts.json'
    plan.write_text(json.dumps({'requests':requests},indent=2)+'\n')
    pre=preflight(plan)
    (output/'cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    if pre['summary']['ambiguous']:
        raise RuntimeError('ambiguous cache evidence; inspect before evaluation')
    # This stage is raw policy only; present cache data is not injected into the
    # rollout. A future DB writer can safely reuse exact compatible records.
    rows=[]
    for index,case in enumerate(cases):
        state=case.initial_state
        env=BottleneckEnv(replace(config,seed=state['episode_seed'],split=split))
        env.reset(np.asarray(state['positions']),np.asarray(state['goals']))
        initial_hash=env.initial_state_sha256()
        episode_key=jax.random.fold_in(jax.random.PRNGKey(seed),index)
        positions=[env.positions.copy()]; clearances=[]; terminal='timeout'
        limit=max_steps or config.max_steps
        for step in range(limit):
            observation=(policy_observation(env) if agent.config['obs_dim']==8 else
                         env.observation()['agents'].astype(np.float32))
            sampled=np.asarray(sample_bounded_actions(agent,
                np.repeat(observation[None],samples_per_step,axis=0),
                jax.random.fold_in(episode_key,step)),dtype=float)
            action=sampled.mean(axis=0)
            norms=np.linalg.norm(action,axis=1,keepdims=True)
            action *= np.minimum(1.0,config.max_speed/np.maximum(norms,1e-30))
            if projector is not None:
                action=projector(env.snapshot(),action).velocity
            _,done,info=env.step(action,diagnose_stalls=False)
            positions.append(env.positions.copy());clearances.append(info['swept_clearance'])
            terminal=info['termination']
            if done:break
        if terminal=='running':terminal='timeout'
        remaining=np.linalg.norm(env.positions-env.goals,axis=1)
        row=dict(rollout_id=case.rollout_id,termination=terminal,
                 full_task_success=terminal=='success',collision=env.collided,
                 individual_goal_fraction=float(np.mean(remaining<=config.goal_tolerance)),
                 steps=len(clearances),minimum_swept_clearance=float(min(clearances)))
        rows.append(row)
        print(json.dumps(row),flush=True)
        trace_dir=output/'traces';trace_dir.mkdir(exist_ok=True)
        video_meta=dict(batch_id=output.name,scenario='bottleneck_family',rollout_id=case.rollout_id,
            controller=f'joint_macflow_stage1_mc{samples_per_step}' + ('_hard_safety' if safety else '_raw'),
            checkpoint=f'{checkpoint} sha256={fingerprint}',
            seed=state['episode_seed'],controller_rng_seed=seed,
            source_record=str(case.archive_path),
            initial_state_sha256=initial_hash,config=env.config.to_dict(),
            start_step=0,episode_steps=len(clearances),complete=True,
            termination=terminal,collision=bool(env.collided),safety_enabled=bool(safety))
        np.savez_compressed(trace_dir/f'{case.rollout_id}.npz',positions=np.asarray(positions),
                            steps=np.arange(len(positions)),goals=env.goals,walls=env.walls,
                            swept_clearance=np.asarray(clearances),metadata_json=np.asarray(json.dumps(video_meta)))
    result=dict(schema='gap_joint_macflow_eval_v2',safety_enabled=bool(safety),
                samples_per_step=samples_per_step,checkpoint_sha256=fingerprint,
                dataset_manifest_sha256=audit['manifest_sha256'],split=split,
                opened_test_archives=audit['opened_test_archives'],rollouts=rows,
                aggregate=dict(count=len(rows),full_task_success=float(np.mean([r['full_task_success'] for r in rows])),
                               individual_goal_fraction=float(np.mean([r['individual_goal_fraction'] for r in rows])),
                               collision_rate=float(np.mean([r['collision'] for r in rows]))),
                cache_preflight=pre['summary'])
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--safety',action='store_true')
    parser.add_argument('--samples-per-step',type=int,default=1)
    parser.add_argument('--split',choices=('dev','test'),default='dev')
    parser.add_argument('--final-evaluation',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.checkpoint,args.output,seed=args.seed,
                         max_cases=args.max_cases,max_steps=args.max_steps,safety=args.safety,
                         samples_per_step=args.samples_per_step,split=args.split,
                         final_evaluation=args.final_evaluation)['aggregate'],indent=2))


if __name__=='__main__':main()
