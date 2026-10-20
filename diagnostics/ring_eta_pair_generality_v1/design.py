from core import *
import copy
from collections import defaultdict,Counter
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight
prereg();gate();old=read(R10/'protocol.json');draw=read(HERE/'ETA_PAIR_DRAW.json');eta=read(R4/'protocol.json')['eta'];assert eta==old['eta']
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json');states=generate('round22_eta_pairs',32,2026252200,native)
def signature(v):
 if not isinstance(v,dict):return None
 values=[]
 for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
  val=next((v[k] for k in keys if k in v),None)
  if val is None:return None
  a=np.asarray(val,np.float64).copy()
  if a.ndim!=2 or a.shape[1]!=2:return None
  a[a==0]=0;values.append((a.shape,a.tobytes().hex()))
 return uid('physical',dict(values=values,timestep=float(v.get('timestep',v.get('step',0)))))
sigs={signature(s['physical']) for s in states};assert len(sigs)==32;collisions=[];unresolved=0
with connect(True) as db:history=db.execute('SELECT state_uid,content_hash,physical_state_json FROM state').fetchall()
ids={s['state_uid'] for s in states};contents={s['content_hash'] for s in states}
for h in history:
 try:v=json.loads(h['physical_state_json']) if h['physical_state_json'] else None
 except ValueError:v=None
 sig=signature(v);unresolved+=sig is None
 if h['state_uid'] in ids or h['content_hash'] in contents or sig in sigs:collisions.append(h['state_uid'])
write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=not collisions,new=32,DB_rows=len(history),unresolved_physical=unresolved,collisions=collisions));assert not collisions
profiles=copy.deepcopy(old['base_profiles']);arms=[f'F{a}' for a in draw['fixed_eta_indices']]+[f'S{a}to{b}' for pair in draw['pairs'] for a,b in (pair,pair[::-1])];rows=[]
for base in old['controller_order']:
 for arm in arms:
  if arm.startswith('F'):a=b=int(arm[1:]);t=700;name=base
  else:
   a,b=map(int,arm[1:].split('to'));t=500;name=f'{base}__eta{a}to{b}__t{t}';v=copy.deepcopy(old['base_profiles'][base]['payload']);v.update(parent_controller_uid=old['base_profiles'][base]['controller_uid'],eta_schedule=dict(eta_a=eta[a],eta_b=eta[b],eta_a_hex=[float_hex(x) for x in eta[a]],eta_b_hex=[float_hex(x) for x in eta[b]],switch_step=t,action_index_rule='a for index<t; b for index>=t; preserve native termination'),schedule_runner_sha256=sha(R9/'run.py'),schedule_core_sha256=sha(R9/'core.py'),execution_semantics='scene_precision_guard_eta_schedule_v1');profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_name=base)
  for s in states:
   for seed in (0,1):
    sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(eta[a])[0];ctl=profiles[name]['controller_uid'];rows.append(dict(state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk)),scene='ring_exchange',controller_name=name,base_controller_name=base,split='round22_prospective',state_index=s['study_index'],eta_index=a,eta=eta[a],eta_a=eta[a],eta_b=eta[b],eta_a_index=a,eta_b_index=b,switch_step=t,seed=seed,arm=arm))
assert len(rows)==draw['main_rollouts'];p=dict(old,experiment_uid=EXP,states=states,fresh_states=states,arms=arms,controllers={'ring_exchange':profiles},core_sha256=sha(HERE/'core.py'),runner_sha256=sha(R9/'run.py'),io_binding_sha256=sha(HERE/'run.py'),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'));write(HERE/'protocol.json',p);write(HERE/'FRESH_STATES.json',states);write(HERE/'main_CASES.json',rows)
with connect() as db,transaction(db):
 db.execute('UPDATE experiment SET protocol_hash=? WHERE experiment_uid=?',(sha(HERE/'protocol.json'),EXP))
 for s in states:db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments']['ring_exchange']['config']),canonical(s['metadata']),'CONTENT_EXACT'))
 for ctl in profiles.values():
  v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
groups=defaultdict(list)
for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
write(HERE/'main_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]));q=preflight(HERE/'main_PLANNED_ROLLOUTS.json');write(HERE/'main_CACHE_PREFLIGHT.json',q);assert q['summary']['ambiguous']==0
with connect(True) as db:missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
assert len(missing)==len(rows);write(HERE/'main_MISSING_CASES.json',missing);write(HERE/'main_PREFLIGHT.receipt.json',dict(utc=utc(),planned=len(rows),missing=len(missing),preflight_sha256=sha(HERE/'main_CACHE_PREFLIGHT.json'),manifest_sha256=sha(HERE/'main_PLANNED_ROLLOUTS.json'),before_submission=True));print('Main preflight',len(rows))
