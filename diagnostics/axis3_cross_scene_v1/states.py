from common import *
from registry import normalized_state
from shared_rollout_db.src.rollout_db import uid,canonical

def signature(v):
 try:
  values=[]
  for k in ['positions','velocities','goals']:
   a=np.asarray(v[k],np.float64);assert a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all();a=a.copy();a[a==0]=0;values.append([a.shape,a.tobytes().hex()])
  return hashlib.sha256(canonical(dict(values=values,timestep=float(v.get('timestep',0)))).encode()).hexdigest()
 except (KeyError,TypeError,ValueError,AssertionError):return None

def main():
 prereg();gate=read(HERE/'CONSISTENCY_GATE.json')['scenes'];inv=read(HERE/'HISTORICAL_INVENTORY.json');cfg=read(HERE/'DESIGN_CONFIG.json');generated=[];errors=[];by={}
 # Copy is byte-identical because original sampler sets environment variables on import.
 from db_sampler import _sample_spec
 from double_bottleneck.scenario import INITIAL_REGIMES,build_layout as db_layout
 from double_bottleneck.environment import Config as DBConfig,DoubleBottleneckEnv
 from single_integrator.environment import Config as ToyConfig,GiveWayEnv
 from four_way_intersection.environment import Config as FourConfig,FourWayIntersectionEnv
 from four_way_intersection.scenario import sample_initial_state,build_layout as four_layout
 from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv,sample_instance
 for sc in SCENES:
  if not gate[sc]['passed']:continue
  cf=inv['environments'][sc]['config']['config'];cc={'toy_give_way':ToyConfig,'double_bottleneck':DBConfig,'four_way_intersection':FourConfig,'ring_exchange':LocalFrameConfig}[sc](**cf);by[sc]={}
  for split,num in [('pilot',16),('probe',16),('eval',32)]:
   if split=='probe' and sc=='four_way_intersection':continue
   out=[]
   for i in range(num):
    seed=cfg[split+'_start']+cfg['scene_offsets'][sc]+i;rng=np.random.default_rng(seed);metadata=dict(draw_seed=seed,split=split,outcome_blind=True)
    if sc=='toy_give_way':
     x=rng.uniform(.55,1.05,2);y=rng.uniform(-.025,.025,2);positions=np.asarray([[-x[0],y[0]],[x[1],y[1]]],np.float32).astype(float);vel=np.zeros((2,2));goals=np.asarray([[1.09,0],[-1.09,0]]);env=GiveWayEnv(cc);env.reset(positions)
    elif sc=='double_bottleneck':
     regime=INITIAL_REGIMES[i%3];spec=_sample_spec('test',regime,i,rng);positions=spec.positions;vel=spec.initial_velocities;goals=db_layout(cc).goals;metadata['regime']=regime;env=DoubleBottleneckEnv(cc);env.reset(positions,regime=regime);env.velocities=np.asarray(vel,np.float64).copy()
    elif sc=='four_way_intersection':
     positions,vel,m=sample_initial_state(cc,'development',i,base_seed=cfg[split+'_start']+cfg['scene_offsets'][sc]);goals=four_layout(cc).goals;env=FourWayIntersectionEnv(cc);env.reset(positions,velocities=vel)
    else:
     ins=sample_instance(seed,'development',cc);positions=ins.positions;vel=ins.velocities;goals=ins.goals;env=RingExchangeEnv(cc);env.reset(positions,velocities=vel,goals=goals)
    physical=clean(dict(positions=positions,velocities=vel,goals=goals,timestep=0));alias=f'round34_{sc}_{split}_{i}_{seed}';sid=inv['environments'][sc]['scenario_uid'];content=uid('content',dict(physical=physical,reset='native true t0'));s=dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),scenario=sc,scenario_uid=sid,content_hash=content,physical=physical,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),family=alias,study_index=i,study_split=split,metadata=metadata);out.append(s);generated.append(s)
   by[sc][split]=out
 with rollout_db.connect(True) as db:old=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
 ids={r['state_uid'] for r in old};contents={r['content_hash'] for r in old};signatures=set();unresolved=0
 for r in old:
  try:sig=signature(json.loads(r['physical_state_json']) if r['physical_state_json'] else None)
  except (TypeError,ValueError):sig=None
  if sig is None:unresolved+=1
  else:signatures.add(sig)
 collisions=[]
 for s in generated:
  sig=signature(s['physical']);assert sig is not None
  if s['state_uid'] in ids or s['content_hash'] in contents or sig in signatures:collisions.append(s['uid'])
  signatures.add(sig)
 write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=not collisions,new_states=len(generated),old_DB_states=len(old),unrecoverable_physical=unresolved,collisions=collisions,failed_runtime_scenes=[s for s in SCENES if not gate[s]['passed']],expert_acceptance_not_run=True));assert not collisions,'Stop without redraw'
 for sc,splits in by.items():
  for split,states in splits.items():by[sc][split]=[normalized_state(s) for s in states]
 write(HERE/'STATES.json',by);print('State reset/dedup PASS',len(generated),flush=True)
if __name__=='__main__':main()
