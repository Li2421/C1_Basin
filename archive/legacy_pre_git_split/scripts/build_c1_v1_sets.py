"""Create fixed validation starts/noise and an untouched Safety challenge set.

The legacy NPZ filename is retained; inputs do not encode a risk version.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import jax

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from single_integrator.c1.train import risk_start_pool, approved_checkpoint
from single_integrator.cbf import CBFConfig, CBFSafetyFilter
from single_integrator.environment import Config
from single_integrator.evaluate import load_policy, rollout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=850)
    p.add_argument('--validation-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--validation-only',action='store_true',
                   help='Build fixed validation inputs without running the Safety challenge search.')
    args=p.parse_args()
    if args.validation_size < 1 or args.horizon < 1:
        raise ValueError('validation-size and horizon must be positive')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if any(args.out_dir.iterdir()):raise FileExistsError('out-dir must be empty')
    checkpoint_hash=approved_checkpoint(args.checkpoint)
    baseline,meta=load_policy(args.checkpoint);plant=Config(**meta['evaluation_environment'])
    raw=ROOT/'datasets/give_way_si_short_v1/raw'
    pool=risk_start_pool(raw,plant,.05)
    starts=[]
    for i in range(args.validation_size):
        if i%2==0: starts.append(pool[(3*i)%len(pool)])
        else:
            with np.load(raw/f'episode_{2*((7*i)%200):04d}.npz') as z: starts.append(z['initial_positions'])
    starts=np.asarray(starts);noise=jax.random.normal(jax.random.PRNGKey(args.seed+1),(len(starts),args.horizon,4),dtype=np.float32)
    np.savez_compressed(args.out_dir/'fixed_validation_v1.npz',initial_positions=starts,noise=np.asarray(noise),
                        metadata_json=json.dumps(dict(seed=args.seed,horizon=args.horizon,kind='fixed_v1_validation',environment=plant.to_dict())))
    if args.validation_only:
        return
    # The test starts that already timed out under Safety are supplemented by
    # legal near-conflict offsets. The Safety policy is evaluated here only;
    # selected starts are never used by the trainer.
    wide=np.load(ROOT/'baseline_309_314/planning/wide_initial_states_200.npz')['test_initial_positions']
    candidates=[wide[i] for i in (15,19,20)]
    noise_ids=[15,19,20]
    for shift in (.18,.21,.24,.27):
        for y in (-.014,0.,.014):
            candidates.append(np.array([[-.18+shift,.008+y],[.18+shift,-.005-y]]))
            noise_ids.append(200+len(noise_ids)-3)
    cfg=replace(plant,max_steps=1200)
    challenge=[];challenge_noise_ids=[];selected_indices=[];records=[]
    for rid,start in enumerate(candidates):
        try:
            summary,_=rollout(baseline,start,cfg,42,noise_ids[rid],CBFSafetyFilter(CBFConfig()),cbf_config=CBFConfig())
            outcome=summary['outcome']
            records.append(dict(candidate_id=rid,noise_rollout_id=noise_ids[rid],initial_positions=np.asarray(start).tolist(),outcome=outcome,steps=summary['episode_steps']))
            if outcome in ('safe_deadlock','other_timeout') and not summary['wall_collision'] and not summary['agent_collision']:
                challenge.append(start)
                challenge_noise_ids.append(noise_ids[rid]);selected_indices.append(rid)
        except Exception as exc:
            records.append(dict(candidate_id=rid,initial_positions=np.asarray(start).tolist(),outcome='evaluation_error',error=str(exc)))
    (args.out_dir/'challenge_search.json').write_text(json.dumps(dict(candidates=records,selected_count=len(challenge),selected_indices=selected_indices),indent=2)+'\n')
    if len(challenge)<3:raise RuntimeError(f'only {len(challenge)} safe stalled challenge starts found; see challenge_search.json')
    np.savez_compressed(args.out_dir/'deadlock_challenge_v1.npz',initial_positions=np.asarray(challenge),
                        noise_rollout_ids=np.asarray(challenge_noise_ids),
                        metadata_json=json.dumps(dict(seed=42,max_steps=1200,baseline_checkpoint_sha256=checkpoint_hash,
                            selection='retrospective Safety safe_deadlock or timeout diagnostic; excluded from training')))

if __name__=='__main__':main()
