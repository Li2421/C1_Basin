"""Development-policy full-trajectory uniform drift recovery acquisition.

This is intentionally distinct from targeted collision windows: every dev
policy rollout (including timeout episodes) contributes uniformly spaced valid
pre-terminal anchors.  Successful centralized re-queries are train data with
`source=uniform_recovery`; their RecoveryAudit preserves development rollout
and source time.  Test is neither loaded nor evaluated.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import jax
import numpy as np
from new_benchmark_common.dataset import RecoveryAudit, TRAJECTORY_SCHEMA, _digest, _jsonable
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import RingExchangeEnv
from .protocol import RingExchangeScenario


def _initial(root,row):
    with np.load(root/row['file'],allow_pickle=False) as d: return json.loads(str(d['initial_state_json'].item()))


def _append(path,continuation,initial,rollout_id,audit):
    states=np.asarray(continuation.states); observations=np.asarray(continuation.observations,dtype=np.float32); actions=np.asarray(continuation.actions,dtype=np.float32)
    digest=_digest(states,observations,actions)
    metadata=dict(continuation.metadata); metadata.update(success=True,terminal_reason=continuation.terminal_reason,
        rollout_id=rollout_id,split='train',source='uniform_recovery',trajectory_digest=digest)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,schema=np.asarray(TRAJECTORY_SCHEMA),states=states,observations=observations,actions=actions,
        initial_state_json=np.asarray(json.dumps(_jsonable(initial),sort_keys=True)),metadata_json=np.asarray(json.dumps(_jsonable(metadata),sort_keys=True)),
        recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__),sort_keys=True)))
    return {'file':str(path.relative_to(path.parents[2])),'rollout_id':rollout_id,'split':'train','source':'uniform_recovery',
            'trajectory_digest':digest,'length':int(len(actions))}


def collect(source,output,checkpoint,*,seed=31600,anchors=30,position_std=.035,velocity_std=.025):
    source,output,checkpoint=Path(source),Path(output),Path(checkpoint)
    if output.exists(): raise FileExistsError(output)
    if anchors<=0: raise ValueError('anchors must be positive')
    manifest=json.loads((source/'manifest.json').read_text()); shutil.copytree(source,output)
    agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=manifest['environment_fingerprint'])
    scenario=RingExchangeScenario(perturb_position_std=position_std,perturb_velocity_std=velocity_std)
    dev=[row for row in manifest['files'] if row['split']=='dev' and row['source']=='nominal']
    records=[]; rollout_rows=[]; attempted=accepted=invalid=failed=0; serial=0
    for index,row in enumerate(dev):
        initial=_initial(source,row); env=RingExchangeEnv();env.reset(initial['positions'],velocities=initial['velocities'],goals=initial['goals'])
        snapshots=[env.augmented_state()]; terminal='timeout'
        for step in range(env.config.max_steps):
            key=jax.random.fold_in(jax.random.PRNGKey(seed+index),step)
            action=np.asarray(sample_bounded_actions(agent,env.observation()[None],key)[0],dtype=np.float64)
            norm=np.linalg.norm(action,axis=-1,keepdims=True);action*=np.minimum(1.,(env.config.max_speed-1e-8)/np.maximum(norm,1e-12))
            _,_,done,info=env.step(action); terminal=str(info['termination'])
            if done: break
            snapshots.append(env.augmented_state())
        selected=np.unique(np.linspace(0,len(snapshots)-1,min(anchors,len(snapshots)),dtype=int))
        made=0
        for time in selected:
            snap=snapshots[int(time)]; p=np.asarray(snap['positions'],dtype=np.float64)
            physical={'positions':p,'velocities':np.asarray(snap['last_applied_velocity'],dtype=np.float64),'goals':np.asarray(snap['goals'],dtype=np.float64),
                      'split':'dev','recovery':True,'source_time':int(time)}
            attempted+=1; perturbation_seed=int(np.random.SeedSequence([seed,index,int(time)]).generate_state(1)[0])
            perturbed=scenario.perturb_state(physical,np.random.default_rng(perturbation_seed))
            if not scenario.valid_state(perturbed): invalid+=1;continue
            continuation=scenario.expert(perturbed,np.random.default_rng(perturbation_seed))
            if not continuation.success: failed+=1;continue
            audit=RecoveryAudit(source_rollout_id=f'dev_policy_v5_drift_{index:03d}',source_split='dev',source_time=int(time),
                collision_type=None,collision_identity=None,distance_to_collision=None,perturbation_seed=perturbation_seed,expert_success=True)
            roll_id=f'train_uniform_drift_v6_{serial:06d}'
            records.append(_append(output/'rollouts'/'train'/f'{roll_id}.npz',continuation,perturbed,roll_id,audit));serial+=1;accepted+=1;made+=1
        rollout_rows.append({'source_rollout_id':f'dev_policy_v5_drift_{index:03d}','nominal_rollout_id':row['rollout_id'],
            'terminal':terminal,'valid_policy_states':len(snapshots),'uniform_anchor_count':len(selected),'accepted':made})
    manifest=json.loads((output/'manifest.json').read_text());manifest['files'].extend(records)
    manifest['counts']['source']['uniform_recovery']+=accepted;manifest['counts']['split']['train']+=accepted
    report={'base_source':str(source),'policy_checkpoint':str(checkpoint),'source':'development_policy_full_trajectory_uniform_anchors_only',
        'anchors_per_rollout':anchors,'position_std':position_std,'velocity_std':velocity_std,'attempted':attempted,'accepted':accepted,
        'invalid_perturbation':invalid,'expert_failed_skipped':failed,'rollouts':rollout_rows,'test_opened':False}
    manifest['extra_report']['v6_policy_drift_uniform_recovery']=report
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    report={**report,'output':str(output),'output_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()}
    (output/'v6_policy_drift_uniform_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('checkpoint');p.add_argument('--anchors',type=int,default=30);p.add_argument('--seed',type=int,default=31600)
    a=p.parse_args();print(json.dumps(collect(a.source,a.output,a.checkpoint,seed=a.seed,anchors=a.anchors),indent=2))
