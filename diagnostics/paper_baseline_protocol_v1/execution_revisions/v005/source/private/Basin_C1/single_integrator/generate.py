"""Generate a fresh paired SI dataset; never relabel PID transitions as SI."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.expert import Expert
from giveway_initial_state import sample_initial_positions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out_dir',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=250)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--corridor_half_length',type=float,default=2.5)
    args=p.parse_args()
    if not 1<=args.pairs<=250:raise ValueError('pairs must be in [1,250]')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if any(args.out_dir.iterdir()):raise FileExistsError('Dataset output must be empty')
    raw=args.out_dir/'raw';raw.mkdir()
    # Keep recording beyond the evaluation success threshold to preserve expert
    # convergence tails. Detect deadlock but never truncate expert generation on it.
    config=Config(corridor_half_length=args.corridor_half_length,terminate_on_success=False,terminate_on_deadlock=False)
    env=GiveWayEnv(config)
    metadata=dict(schema='giveway_si_dataset_v1',environment=config.to_dict(),environment_fingerprint=config.fingerprint,evaluation_environment=Config(corridor_half_length=args.corridor_half_length).to_dict(),seed=args.seed,pairs=args.pairs,split_pairs={'train':[0,199],'val':[200,224],'test':[225,249]},expert_completion={'goal_error':.02,'speed':.03},reward='shared_goal_progress + 0.01*goal_success - per_agent_wall_count - agent_near_collision',complete=False)
    (args.out_dir/'environment.json').write_text(json.dumps(metadata,indent=2))
    rng=np.random.default_rng(args.seed);rows=[]
    for pair in range(args.pairs):
        initial=sample_initial_positions(rng)
        for mode in (0,1):
            env.reset(initial);expert=Expert(env,mode)
            buffers={k:[] for k in ['observations','actions','rewards','next_observations','dones','positions','velocities','stages','next_stages','wall_collision','agent_collision','deadlock','integration_residual']}
            complete=False
            for step in range(config.max_steps):
                observation=env.observation();stage=expert.stage
                action=expert.action();next_obs,reward,_,info=env.step(action)
                complete=bool((info['goal_errors']<=.02).all() and (info['speeds']<=.03).all())
                values=dict(observations=observation,actions=action,rewards=reward,next_observations=next_obs,dones=complete,positions=observation[:,:2].copy(),velocities=observation[:,2:4].copy(),stages=stage,next_stages=expert.stage,wall_collision=info['wall_collision'],agent_collision=info['agent_collision'],deadlock=info['deadlock'],integration_residual=info['integration_residual'])
                for k,v in values.items():buffers[k].append(v)
                if complete:break
            summary=env.summary();eid=pair*2+mode
            file=f'episode_{eid:04d}.npz'
            np.savez_compressed(raw/file,**{k:np.asarray(v) for k,v in buffers.items()},pair_id=pair,mode=mode,initial_positions=initial,goals=env.goals,success=summary['success'],expert_complete=complete,environment_fingerprint=config.fingerprint)
            rows.append(dict(episode_id=eid,pair_id=pair,mode='A_FIRST' if mode==0 else 'B_FIRST',steps=env.step_count,success=summary['success'],expert_complete=complete,agent_collision=summary['agent_collision'],wall_collision=summary['wall_collision'],deadlock=summary['deadlock'],file=file))
        if (pair+1)%25==0:print(f'pairs={pair+1}, episodes={len(rows)}',flush=True)
    with (args.out_dir/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata['complete']=True
    metadata['summary']={k:sum(r[k] for r in rows) for k in ['success','expert_complete','agent_collision','wall_collision','deadlock']}
    metadata['summary']['episodes']=len(rows)
    metadata['summary']['transitions']=sum(r['steps'] for r in rows)
    (args.out_dir/'environment.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata['summary']),flush=True)
    if not all(r['expert_complete'] and not r['agent_collision'] and not r['wall_collision'] for r in rows):
        raise RuntimeError('Expert dataset failed quality gate; inspect before training')


if __name__=='__main__':main()
