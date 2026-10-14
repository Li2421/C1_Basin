from core import *
import copy
from collections import Counter,defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.planner import preflight
from adapter import FourFlowAdapter
from four_way_intersection.environment import Config
from four_way_intersection.scenario import sample_initial_state,build_layout
prereg();gate();old=read(E8/'protocol.json');cfg=Config(**old['environments']['four_way_intersection']['config']['config']);sc='four_way_intersection';sid=old['environments'][sc]['scenario_uid']
def sig(p):
 if not isinstance(p,dict):return None
 arr=[]
 for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
  v=next((p[k] for k in keys if k in p),None)
  if v is None:return None
  a=np.asarray(v,np.float64);a=a.copy();a[a==0]=0;arr.append((a.shape,a.tobytes().hex()))
 return hashlib.sha256(canonical(dict(arr=arr,timestep=float(p.get('timestep',p.get('step',0))))).encode()).hexdigest()
states=[]
for i in range(32):
 pos,vel,meta=sample_initial_state(cfg,'development',i,base_seed=2026141400);physical=dict(positions=pos.tolist(),velocities=vel.tolist(),goals=build_layout(cfg).goals.tolist(),timestep=0);alias=f'round14b_four_census_{i}';content=uid('content',dict(physical=physical,reset='empty monitor true-t0'));states.append(dict(uid=alias,state_uid=uid('state',dict(scenario=sid,content=content)),content_hash=content,scenario=sc,scenario_uid=sid,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),physical=physical,family='round14b_four_census',study_split='prospective_round14b',study_index=i,metadata=meta))
sigs={sig(s['physical']) for s in states};assert len(sigs)==32;collisions=[];unresolved=Counter();recovered=0
with connect(True) as db:
 historical=db.execute('SELECT state_uid,content_hash,physical_state_json,identity_quality FROM state').fetchall()
 for row in historical:
  try:p=json.loads(row['physical_state_json']) if row['physical_state_json'] else None
  except ValueError:p=None
  h=sig(p)
  if h is None:unresolved[row['identity_quality']]+=1
  else:recovered+=1
  if h in sigs or row['state_uid'] in {s['state_uid'] for s in states} or row['content_hash'] in {s['content_hash'] for s in states}:collisions.append(row['state_uid'])
assert not collisions
freeze(HERE/'FRESH_STATES.json',states);write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=True,fresh=32,all_DB_rows=len(historical),recoverable=recovered,unresolved=dict(unresolved),collisions=collisions,draw_base_seed=2026141400,no_target_labels=True))
meta=read(HERE/'COMPATIBILITY_METADATA_AUDIT.json');profiles={sc:{}};cohort=[];rows=[]
for row in meta['rows']:
 if not row['compatible_metadata'] or row['kind']=='final' and not row['param_different_from_best']:continue
 name=row['run']+'__'+row['kind'];v=copy.deepcopy(old['controllers'][sc]['flow_isotropic']['payload']);v.update(flow_checkpoint_path=row['path'],flow_checkpoint_sha256=row['file_sha256'],round14_adapter_sha256=sha(HERE/'adapter.py'));ctl=uid('ctl',v);profiles[sc][name]=dict(controller_uid=ctl,payload=v);cohort.append(dict(name=name,group=row['kind'],run=row['run'],param_sha256=row['param_sha256']))
 for arm in ['F10','F13','F15']:
  a=int(arm[1:]);eta=old['eta'][a];e=eta_identity(eta)[0]
  for s in states:
   for seed in range(2):
    sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));rid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk));rows.append(dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=rid,scene=sc,controller_name=name,base_controller_name=name,split=s['study_split'],state_index=s['study_index'],eta_index=a,eta=eta,eta_a=eta,eta_b=eta,switch_step=1100,seed=seed,arm=arm))
p=dict(schema='round14b_four_controller_census_v1',experiment_uid=EXP,states=states,controllers=profiles,base_profiles=profiles,environments=old['environments'],eta=old['eta'],source_hashes=read(HERE/'gate_protocol.json')['source_hashes'],run_sha256=sha(HERE/'run.py'),core_sha256=sha(HERE/'core.py'),cohort=cohort,controller_order=[x['name'] for x in cohort],arms=['F10','F13','F15'],prereg_sha256=sha(HERE/'PREREGISTRATION.md'))
freeze(HERE/'protocol.json',p);freeze(HERE/'main_CASES.json',rows)
with connect() as db,transaction(db):
 db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round14B Four compatible-controller census',str(HERE/'main'),sha(HERE/'protocol.json'),sha(HERE/'run.py'),canonical(dict(requests=len(rows)))))
 for s in states:
  db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],sid,s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments'][sc]['config']),canonical(dict(alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'));db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(sid,s['uid'],s['state_uid'],EXP))
 for ctl in profiles[sc].values():
  v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],sid,v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
groups=defaultdict(list)
for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
freeze(HERE/'main_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=ks) for (s,e,c),ks in groups.items()]));pre=preflight(HERE/'main_PLANNED_ROLLOUTS.json');write(HERE/'main_CACHE_PREFLIGHT.json',pre);assert pre['summary']['ambiguous']==0 and pre['summary']['genuinely_missing']==len(rows);freeze(HERE/'main_MISSING_CASES.json',rows);print(len(rows),'genuinely missing')
