"""Development-only goal-arrival/overshoot audit for v7 lane-local policy."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import jax,numpy as np
from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
from .environment import Config
from .lane_local import FourWayLaneLocalEnv, local_to_world, representation_fingerprint, world_to_local

def run(dataset,checkpoint,output,*,seed=3125):
    dataset,checkpoint,output=Path(dataset),Path(checkpoint),Path(output)
    if output.exists():raise FileExistsError(output)
    m=json.loads((dataset/'manifest.json').read_text());cfg=Config(**{k:v for k,v in m['scenario_config'].items() if k in Config.__dataclass_fields__});fp=representation_fingerprint(cfg)
    agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=fp);rows=[]
    for case,row in enumerate(r for r in m['files'] if r['split']=='dev' and r['source']=='nominal'):
        with np.load(dataset/row['file'],allow_pickle=False) as d:initial=json.loads(str(d['initial_state_json'].item()))
        env=FourWayLaneLocalEnv(cfg);env.reset(np.asarray(initial['positions']),np.asarray(initial['velocities']));positions=[env.positions.copy()];local_actions=[];info={'termination':'timeout'};key=jax.random.PRNGKey(seed+case)
        for t in range(cfg.max_steps):
            u=np.asarray(sample_bounded_actions(agent,env.observation()[None],jax.random.fold_in(key,t))[0],dtype=np.float64);n=np.linalg.norm(u,axis=-1,keepdims=True);u*=np.minimum(1.,(cfg.max_speed-1e-8)/np.maximum(n,1e-12))
            _,_,done,info=env.step(local_to_world(u));positions.append(env.positions.copy());local_actions.append(u)
            if done:break
        p=np.asarray(positions);acts=np.asarray(local_actions);errors=np.linalg.norm(env.goals[None]-p,axis=-1);entered=[];departed=[]
        for a in range(4):
            hit=np.flatnonzero(errors[:,a]<=cfg.goal_tolerance);first=int(hit[0]) if len(hit) else None
            leave=None if first is None else next((int(q) for q in range(first+1,len(errors)) if errors[q,a]>cfg.goal_tolerance),None)
            entered.append(first);departed.append(leave)
        rel_local=world_to_local(env.goals-env.positions)
        # Negative forward remaining means the disk has passed its goal along
        # its own approach direction; this is a physical overshoot predicate.
        wall_agent=int(np.argmin(env.distances()[0].reshape(-1)))
        rows.append({'rollout_id':row['rollout_id'],'termination':info['termination'],'episode_steps':len(acts),'min_goal_errors':errors.min(axis=0).tolist(),'first_goal_tolerance_steps':entered,'first_departure_after_goal_steps':departed,'agents_ever_goal_reached':int(sum(x is not None for x in entered)),'agents_left_after_reaching':int(sum(x is not None for x in departed)),'termination_relative_goal_local':rel_local.tolist(),'termination_local_action':acts[-1].tolist(),'wall_nearest_agent':wall_agent,'wall_collision':bool(info['wall_collision']),'agent_collision':bool(info['agent_collision'])})
    total_agents=4*len(rows);summary={'rollouts':len(rows),'agents':total_agents,'agents_ever_goal_reached':sum(r['agents_ever_goal_reached'] for r in rows),'agents_left_after_reaching':sum(r['agents_left_after_reaching'] for r in rows),'rollouts_with_goal_departure':sum(r['agents_left_after_reaching']>0 for r in rows),'min_goal_error_quantiles':np.quantile(np.asarray([x for r in rows for x in r['min_goal_errors']]),[0,.25,.5,.75,1]).tolist(),'test_opened':False}
    output.mkdir(parents=True);(output/'audit.json').write_text(json.dumps({'scope':'development only','summary':summary,'rollouts':rows},indent=2,sort_keys=True)+'\n');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--seed',type=int,default=3125);a=p.parse_args();print(json.dumps(run(a.dataset,a.checkpoint,a.output,seed=a.seed),indent=2))
