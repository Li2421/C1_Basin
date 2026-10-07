"""Development-only goal/forward-action audit for a world-frame Four policy."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import jax,numpy as np
from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
from .environment import Config,FourWayIntersectionEnv
from .lane_local import world_to_local

def run(dataset,checkpoint,output,*,seed=3123):
 dataset,checkpoint,output=Path(dataset),Path(checkpoint),Path(output)
 if output.exists():raise FileExistsError(output)
 m=json.loads((dataset/'manifest.json').read_text());cfg=Config(**{k:v for k,v in m['scenario_config'].items() if k in Config.__dataclass_fields__});agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=cfg.fingerprint);rows=[]
 for case,row in enumerate(r for r in m['files'] if r['split']=='dev' and r['source']=='nominal'):
  with np.load(dataset/row['file'],allow_pickle=False) as d:initial=json.loads(str(d['initial_state_json'].item()))
  env=FourWayIntersectionEnv(cfg);env.reset(np.asarray(initial['positions']),np.asarray(initial['velocities']));p=[env.positions.copy()];acts=[];key=jax.random.PRNGKey(seed+case);info={'termination':'timeout'}
  for t in range(cfg.max_steps):
   u=np.asarray(sample_bounded_actions(agent,env.observation()[None],jax.random.fold_in(key,t))[0],dtype=np.float64);n=np.linalg.norm(u,axis=-1,keepdims=True);u*=np.minimum(1.,(cfg.max_speed-1e-8)/np.maximum(n,1e-12));_,_,done,info=env.step(u);p.append(env.positions.copy());acts.append(u)
   if done:break
  p=np.asarray(p);acts=np.asarray(acts);err=np.linalg.norm(env.goals[None]-p,axis=-1);entered=[];left=[]
  for a in range(4):
   h=np.flatnonzero(err[:,a]<=cfg.goal_tolerance);f=int(h[0]) if len(h) else None;entered.append(f);left.append(None if f is None else next((int(q) for q in range(f+1,len(err)) if err[q,a]>cfg.goal_tolerance),None))
  rem_local=world_to_local(env.goals-env.positions);act_local=world_to_local(acts[-1]);overshot=rem_local[:,0]<-cfg.goal_tolerance;outward=(act_local[:,0]>.05)&overshot
  rows.append({'rollout_id':row['rollout_id'],'termination':info['termination'],'steps':len(acts),'min_goal_errors':err.min(0).tolist(),'entered':entered,'left':left,'termination_relative_goal_local':rem_local.tolist(),'termination_local_action':act_local.tolist(),'overshot_agents':np.flatnonzero(overshot).tolist(),'overshot_still_forward_agents':np.flatnonzero(outward).tolist()})
 flat=[x for r in rows for x in r['min_goal_errors']];summary={'rollouts':len(rows),'agents_ever_goal_reached':sum(sum(x is not None for x in r['entered']) for r in rows),'agents_left_after_reaching':sum(sum(x is not None for x in r['left']) for r in rows),'rollouts_with_goal_departure':sum(any(x is not None for x in r['left']) for r in rows),'overshot_agents_at_terminal':sum(len(r['overshot_agents']) for r in rows),'overshot_still_forward_at_terminal':sum(len(r['overshot_still_forward_agents']) for r in rows),'min_goal_error_median':float(np.median(flat)),'test_opened':False}
 output.mkdir(parents=True);(output/'audit.json').write_text(json.dumps({'scope':'development only','summary':summary,'rollouts':rows},indent=2,sort_keys=True)+'\n');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--seed',type=int,default=3123);a=p.parse_args();print(json.dumps(run(a.dataset,a.checkpoint,a.output,seed=a.seed),indent=2))
