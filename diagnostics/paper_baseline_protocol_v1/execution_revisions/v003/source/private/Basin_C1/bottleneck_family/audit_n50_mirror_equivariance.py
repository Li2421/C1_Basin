"""Audit exact data reflection and learned Gap1 Flow reflection symmetry.

Uses paired, untouched DEV one-way expert trajectories. Model statistics
average independent standard-Gaussian Flow latents; no control is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions


OBS_SIGN=np.asarray([-1,1,-1,1,-1,1,-1,1],dtype=np.float32)
ACT_SIGN=np.asarray([-1,1],dtype=np.float32)


def run(dataset:Path,checkpoints:list[Path],output:Path,*,latents=64,seed=6137):
    dataset,output=Path(dataset),Path(output)
    manifest=json.loads((dataset/'manifest.json').read_text())
    n=int(manifest['scenario_config']['num_agents'])
    indexed={r['rollout_id']:r for r in manifest['files'] if r['split']=='dev'
             and r['source']=='nominal'}
    pairs=[]
    for name,row in indexed.items():
        if '_full_LR_perm0' not in name:
            continue
        mirror=name.replace('_full_LR_perm0','_full_RL_perm0')
        if mirror not in indexed:
            raise ValueError(f'missing exact mirrored DEV counterpart: {name}')
        pairs.append((row,indexed[mirror]))
    if not pairs:
        raise ValueError('no mirrored DEV full-density pairs')
    frames=[]
    for lr,rl in pairs:
        with np.load(dataset/lr['file'],allow_pickle=False) as left,\
             np.load(dataset/rl['file'],allow_pickle=False) as right:
            lo,ro=left['observations'],right['observations']
            la,ra=left['actions'],right['actions']
            for t in (0,10,100):
                if t>=min(len(la),len(ra)):
                    continue
                frames.append(dict(left=lr['rollout_id'],right=rl['rollout_id'],t=t,
                    obs_left=np.asarray(lo[t]),obs_right=np.asarray(ro[t]),
                    action_left=np.asarray(la[t]),action_right=np.asarray(ra[t])))
    obs_error=max(float(np.max(np.abs(f['obs_left']*OBS_SIGN-f['obs_right']))) for f in frames)
    action_error=max(float(np.max(np.abs(f['action_left']*ACT_SIGN-f['action_right']))) for f in frames)
    observations=[]
    for frame in frames:
        for side in ('left','right'):
            observations.extend([frame['obs_'+side]]*latents)
    batch=np.asarray(observations,dtype=np.float32)
    results={}
    for path in map(Path,checkpoints):
        agent,_=load_checkpoint(path,
            expected_environment_fingerprint=manifest['environment_fingerprint'])
        prediction=np.asarray(sample_bounded_actions(agent,batch,jax.random.PRNGKey(seed)))
        prediction=prediction.reshape(len(frames),2,latents,prediction.shape[-2],2)
        rows=[]
        for i,frame in enumerate(frames):
            mean_left=prediction[i,0].mean(axis=0)
            mean_right=prediction[i,1].mean(axis=0)
            reflected=mean_left*ACT_SIGN
            delta=reflected-mean_right
            rows.append(dict(left=frame['left'],right=frame['right'],step=frame['t'],
                mean_action_reflection_error=float(np.mean(np.linalg.norm(delta,axis=1))),
                maximum_action_reflection_error=float(np.max(np.linalg.norm(delta,axis=1))),
                mean_action_lr_norm=float(np.mean(np.linalg.norm(mean_left,axis=1))),
                mean_action_rl_norm=float(np.mean(np.linalg.norm(mean_right,axis=1)))))
        grouped={str(t):dict(frames=sum(r['step']==t for r in rows),
            mean_reflection_error=float(np.mean([r['mean_action_reflection_error'] for r in rows if r['step']==t])))
            for t in (0,10,100)}
        results[str(path)]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                grouped=grouped,rows=rows)
    report=dict(schema=f'gap1_n{n}_mirror_equivariance_audit_v1',N=n,
        dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
        physical_pairs=len(pairs),frames=len(frames),latents_per_direction=latents,
        exact_observation_reflection_max_error=obs_error,
        exact_expert_action_reflection_max_error=action_error,
        results=results,caveat='One-step model distribution means; complete rollout success assessed separately')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--latents',type=int,default=64)
    args=parser.parse_args()
    report=run(args.dataset,args.checkpoint,args.output,latents=args.latents)
    print(json.dumps({k:v['grouped'] for k,v in report['results'].items()},indent=2))


if __name__=='__main__':
    main()
