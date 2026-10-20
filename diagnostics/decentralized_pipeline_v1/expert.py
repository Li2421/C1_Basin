from core import *
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.planner import preflight
EXP='exp_decentralized_pipeline_v1_expert'
def create():
 prereg();states=read(HERE/'EVAL_STATES.json');config=read(HERE/'base_protocol.json')['environments']['ring_exchange']['config']['config'];v=dict(scenario_uid=states[0]['scenario_uid'],controller_class='CentralizedExpert',planner_sha256=sha(ROOT/'ring_exchange/expert.py'),parameters=dict(route_radius=1.85,route_speed=.46,arc_increment=.10,direct_goal_distance=1.10,require_success=False),horizon=700,dt=.05,conditioning='privileged deterministic offline joint two-circulation search then native replay',success_semantics='native collision_free_success and native success termination',rng=dict(deterministic=True),physical_config=config);ctl=uid('ctl',v);eid,eta,hexes=eta_identity([0.,0.,0.]);rows=[]
 for s in states:
  seed=canonical(dict(expert_index=0,deterministic=True,rng_namespace=s['rng_namespace']));rows.append(dict(rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=seed)),state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=seed,state_index=s['study_index'],seed=0))
 write(HERE/'expert_protocol.json',dict(experiment_uid=EXP,states=states,controller_uid=ctl,payload=v,cases=rows,expert_code_sha256=sha(__file__)));manifest=HERE/'expert_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]))
 with connect() as db,transaction(db):
  db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round28 expert upper reference',str(HERE/'expert'),sha(HERE/'expert_protocol.json'),sha(__file__),canonical(dict(privileged=True))))
  db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl,v['scenario_uid'],None,None,uid('safe',dict(physical=config,planner=v['planner_sha256'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'));db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*eta,hexes))
 pf=preflight(manifest);write(HERE/'expert_CACHE_PREFLIGHT.json',pf);assert pf['summary']['ambiguous']==0
 with connect(True) as db:missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
 assert len(missing)==32;write(HERE/'expert_MISSING_CASES.json',missing);seal('expert_PREFLIGHT.receipt.json',[manifest,HERE/'expert_CACHE_PREFLIGHT.json',HERE/'expert_protocol.json'],before_submission=True);print('Expert preflight PASS32')
def run():
 prereg();from ring_exchange.environment import RingExchangeEnv,LocalFrameConfig
 from ring_exchange.expert import CentralizedExpert
 from shared_rollout_db.src import cache_writer
 import jax,time
 jax.config.update('jax_enable_x64',False);cache_writer.ROOT=HERE;p=read(HERE/'expert_protocol.json');assert sha(__file__)==p['expert_code_sha256'];assert sha(ROOT/'ring_exchange/expert.py')==p['payload']['planner_sha256']
 for c in read(HERE/'expert_MISSING_CASES.json'):
  out=HERE/'expert/raw'/f"{c['rollout_uid']}.json"
  if out.exists():continue
  s=p['states'][c['state_index']];cfg=LocalFrameConfig(**p['payload']['physical_config']);env=RingExchangeEnv(cfg);physical=s['physical'];env.reset(np.asarray(physical['positions']),velocities=np.asarray(physical['velocities']),goals=np.asarray(physical['goals']));initial=clean(env.augmented_state());pos=[env.positions.copy()];actions=[];swept=[];term='timeout';error=None;start=time.perf_counter();planning=None;plan=None
  try:
   plan=CentralizedExpert().plan(env,require_success=False);planning=time.perf_counter()-start
   for action in plan.actions:
    _,_,done,info=env.step(action);pos.append(env.positions.copy());actions.append(action.copy());term=info['termination'];swept.append(min(float(np.min(info['swept_obstacle_distance']))-cfg.collision_margin,float(np.min(info['swept_outer_distance']))-cfg.collision_margin,float(np.min(info['swept_agent_distances']))-cfg.agent_collision_margin))
    if done:break
   assert np.array_equal(np.asarray(pos),plan.positions),'expert replay must match validated planned trajectory'
  except Exception as e:error=dict(type=type(e).__name__,message=str(e)[:600],step=len(actions));term='numerical_failure';planning=time.perf_counter()-start if planning is None else planning
  summary=env.summary();collision=any(summary.get(k,False) for k in ['wall_collision','outer_collision','obstacle_collision','agent_collision']);trace=out.with_suffix('.npz');trace.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(trace,positions=np.asarray(pos),actions=np.asarray(actions),steps=np.arange(len(pos)),goals=env.goals,walls=np.empty((0,2,2)),swept_clearance=np.asarray(swept),initial_augmented_state_json=np.array(json.dumps(initial)))
  r=dict(schema='round28_expert_seed_v1',experiment_uid=EXP,case=c,**{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']},success=int(summary['collision_free_success'] and term=='success' and error is None),collision=int(collision),timeout=int(term=='timeout'),deadlock=int(term=='deadlock'),numerical_failure=int(error is not None),episode_length=len(actions),outcome=term,error=error,planning_seconds=planning,candidate_count=plan.candidate_count if plan else None,plan_metadata=plan.metadata if plan else None,trace_path=str(trace),trace_sha256=sha(trace),protocol_sha256=sha(HERE/'expert_protocol.json'),execution_code_sha256=sha(__file__),jax_enable_x64=bool(jax.config.x64_enabled));write(out,r);cache_writer.append_journal([r],EXP,'expert');print('expert',c['state_index'],term,'planning',planning,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
