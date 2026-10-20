"""Navigation expert gate: native single-agent physical semantics, append-only evidence."""
from common import *
from geometry import GeneratedGeometry
from single_agent import SingleAgentGeneratedEnv
from bottleneck_family.scenario import Config
from navigation_field import NavigationGrid,expert_action
import fcntl,time

def worker(stage,index):
 assert read(HERE/'REPRODUCTION_GATE.json')['passed'];assert read(HERE/'NAVIGATION_UNIT_TESTS.json')['passed'];prereg()
 cases=read(HERE/f'{stage}_MISSING_CASES.json');folder=HERE/stage;folder.mkdir(exist_ok=True);cached=None;grid=None
 while True:
  with (folder/'cursor').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);i=int(f.read().strip() or 0);f.seek(0);f.truncate();f.write(str(i+1));f.flush()
  if i>=len(cases):break
  c=cases[i];s=c['state'];path=folder/'raw'/f"{c['rollout_uid']}.json"
  if path.exists():continue
  begin=time.monotonic();error=None;actions=[];pos=[np.asarray(s['positions'])];clear=[];term='numerical_failure'
  try:
   if cached!=s['geometry_seed']:
    cfg=dict(c['config']);cfg['num_agents']=4;geo=GeneratedGeometry(Config(**cfg),s['rectangles']);grid=NavigationGrid(geo);cached=s['geometry_seed']
   cfg=dict(c['config']);cfg['num_agents']=4;env=SingleAgentGeneratedEnv(grid.geometry,s['positions'],s['goals'],Config(**cfg));field=grid.field(env.goals[0])
   for step in range(env.config.max_steps):
    u=expert_action(env,[field]);_,done,info=env.step(u,diagnose_stalls=False);actions.append(u);pos.append(env.positions.copy());clear.append(info['min_swept_wall_clearance']);term=info['termination']
    if done:break
  except Exception as e:error=dict(type=type(e).__name__,message=str(e));term='numerical_failure'
  positions=np.asarray(pos);poly=np.asarray(s['paths'][0]);a=poly[:-1];d=np.diff(poly,axis=0);v=positions[:,0,None]-a[None];t=np.clip(np.sum(v*d[None],-1)/np.maximum(np.sum(d*d,-1),1e-30),0,1);deviation=np.linalg.norm(v-t[...,None]*d[None],axis=-1).min(-1)
  trace=folder/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(exist_ok=True);np.savez_compressed(trace,positions=positions,actions=np.asarray(actions).reshape(-1,1,2),wall_clearance=np.asarray(clear),path_deviation=deviation,goals=s['goals'],walls=grid.geometry.walls if grid else [])
  result=dict(cache_identity_schema='registered_exact_uid_v1',experiment_uid=EXP,scenario='Generated '+s['family'],**{k:c[k] for k in ['scenario_uid','state_uid','controller_uid','eta_uid','rollout_uid','seed_key']},eta=c['eta'],future_index=c['seed'],future_root=2026101740,initial_positions=s['positions'],initial_velocities=s['velocities'],goals=s['goals'],flow_checkpoint_sha256=None,success=int(term=='success' and error is None),collision=int(term=='collision'),timeout=int(term=='timeout'),deadlock=0,numerical_failure=int(error is not None),outcome=term,error=error,episode_length=len(actions),first_action=actions[0] if len(actions) else None,trace_path=str(trace),trace_sha256=sha(trace),case=c,elapsed_seconds=time.monotonic()-begin,minimum_wall_clearance=min(clear) if clear else None,route_deviation_state_fraction=float(np.mean(deviation>.5)),route_deviation_episode=bool(np.any(deviation>.5)),maximum_path_deviation=float(deviation.max()),final_position=positions[-1].tolist())
  write(path,result);j=cache_writer.append_journal([clean(result)],EXP,f'expert_{index}');assert j.is_relative_to(ROOT/'shared_rollout_db/journals'/EXP)
  print(i,s['family'],s['map_index'],s['index'],term,flush=True)
if __name__=='__main__':worker(sys.argv[1],int(sys.argv[2]))
