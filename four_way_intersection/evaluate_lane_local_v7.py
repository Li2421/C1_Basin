"""Development-only closed-loop classification for the v7 lane-local policy."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import jax,numpy as np
from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
from .environment import Config
from .lane_local import FourWayLaneLocalEnv, local_to_world, representation_fingerprint

def run(dataset,checkpoint,output,*,seed=3125):
    dataset,checkpoint,output=Path(dataset),Path(checkpoint),Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((dataset/'manifest.json').read_text());cfg=Config(**{k:v for k,v in manifest['scenario_config'].items() if k in Config.__dataclass_fields__})
    expected=representation_fingerprint(cfg)
    if manifest['environment_fingerprint']!=expected:raise ValueError('dataset does not have v7 lane-local fingerprint')
    agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=expected)
    rows=[]
    for case,row in enumerate(r for r in manifest['files'] if r['split']=='dev' and r['source']=='nominal'):
        with np.load(dataset/row['file'],allow_pickle=False) as d:initial=json.loads(str(d['initial_state_json'].item()))
        env=FourWayLaneLocalEnv(cfg);env.reset(np.asarray(initial['positions']),np.asarray(initial['velocities']));key=jax.random.PRNGKey(seed+case);info={'termination':'timeout'}
        for step in range(cfg.max_steps):
            local=np.asarray(sample_bounded_actions(agent,env.observation()[None],jax.random.fold_in(key,step))[0],dtype=np.float64);norm=np.linalg.norm(local,axis=-1,keepdims=True);local*=np.minimum(1.,(cfg.max_speed-1e-8)/np.maximum(norm,1e-12))
            _,_,done,info=env.step(local_to_world(local))
            if done:break
        rows.append({'rollout_id':row['rollout_id'],'termination':info['termination'],'episode_steps':int(info['step']),'wall_collision':bool(info['wall_collision']),'agent_collision':bool(info['agent_collision']),'nearest_wall_identity':info.get('nearest_wall_identity')})
    n=max(len(rows),1);aggregate={'rollouts':len(rows),'full_task_success':sum(x['termination']=='success' for x in rows)/n,'wall_collision':sum(x['wall_collision'] for x in rows)/n,'agent_collision':sum(x['agent_collision'] for x in rows)/n,'timeout':sum(x['termination']=='timeout' for x in rows)/n,'mean_episode_length':float(np.mean([x['episode_steps'] for x in rows])),'test_opened':False,'representation':'four_way_lane_local_physical_v7'}
    output.mkdir(parents=True);(output/'metrics.json').write_text(json.dumps({'scope':'development only','aggregate':aggregate,'rollouts':rows},indent=2,sort_keys=True)+'\n');return aggregate
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('checkpoint');p.add_argument('output');p.add_argument('--seed',type=int,default=3125);a=p.parse_args();print(json.dumps(run(a.dataset,a.checkpoint,a.output,seed=a.seed),indent=2))
