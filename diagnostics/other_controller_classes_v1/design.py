from core import *
import copy
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity,float_hex
from shared_rollout_db.src.planner import preflight
EXP='exp_other_controller_classes_v1'
REFS=['base_u_macflow','base_u_v10_local_macflow','base_u_v9_local_macflow','base_u_plus_w_v4_wide_macflow']
def signature(v):
 try:
  values=[]
  for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
   a=np.asarray(next(v[k] for k in keys if k in v),np.float64).copy();assert a.shape==(4,2);a[a==0]=0;values.append(a.tobytes().hex())
  return uid('physical',dict(values=values,timestep=float(v.get('timestep',v.get('step',0)))))
 except (TypeError,KeyError,StopIteration,AssertionError):return None
def dedup(states,label):
 with connect(True) as db:old=db.execute('SELECT state_uid,content_hash,physical_state_json FROM state').fetchall()
 sig={signature(s['physical']) for s in states};ids={s['state_uid'] for s in states};contents={s['content_hash'] for s in states};assert len(sig)==len(states);collisions=[];unresolved=0
 for o in old:
  v=json.loads(o['physical_state_json']) if o['physical_state_json'] else None;x=signature(v);unresolved+=x is None
  if o['state_uid'] in ids or o['content_hash'] in contents or (x is not None and x in sig):collisions.append(o['state_uid'])
 write(HERE/f'{label}_DEDUP_AUDIT.json',dict(utc=utc(),passed=not collisions,new=len(states),DB_rows=len(old),unresolved_physical=unresolved,collisions=collisions));assert not collisions
def register(p,states,profiles):
 with connect() as db,transaction(db):
  db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round27 non-Flow TT',str(HERE),sha(HERE/'base_protocol.json'),sha(HERE/'controllers.py'),canonical(dict(pure_Q=True))))
  for s in states:db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments']['ring_exchange']['config']),canonical(s['metadata']),'CONTENT_EXACT'))
  for ctl in profiles.values():
   v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v.get('flow_checkpoint_sha256'),v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
def cases(p,states,names,arms):
 out=[]
 for n in names:
  for arm in arms:
   t=int(arm[1:]) if arm.startswith('S') else 700;a=13 if arm.startswith('S') else int(arm[1:]);b=10 if arm.startswith('S') else a;profile=copy.deepcopy(p['base_profiles'][n]);v=profile['payload']
   if arm.startswith('S'):v.update(eta_schedule=dict(eta_a=p['eta'][a],eta_b=p['eta'][b],switch_step=t,action_index_rule='a for index<t else b'));profile['controller_uid']=uid('ctl',v)
   p['controllers'][n+'__'+arm]=profile
   for s in states:
    for seed in ([0,1] if p['phase']=='main' else [0]):
     sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));e=eta_identity(p['eta'][a])[0];ctl=profile['controller_uid'];out.append(dict(rollout_uid=uid('roll',dict(state=s['state_uid'],eta=e,controller=ctl,seed=sk)),state_uid=s['state_uid'],eta_uid=e,controller_uid=ctl,seed_key=sk,controller_name=n+'__'+arm,base_controller_name=n,state_index=s['study_index'],seed=seed,eta_a=p['eta'][a],eta_b=p['eta'][b],eta_index=a,switch_step=t,arm=arm))
 return out
def stage(phase,states,names,arms):
 prereg();p=read(HERE/'base_protocol.json');p.update(phase=phase,states=states,controller_order=names,arms=arms,controllers={},runner_sha256=sha(HERE/'run.py'));rows=cases(p,states,names,arms);write(HERE/f'{phase}_protocol.json',p);write(HERE/f'{phase}_CASES.json',rows);register(p,states,p['controllers'])
 groups={}
 for c in rows:groups.setdefault((c['state_uid'],c['eta_uid'],c['controller_uid']),[]).append(c['seed_key'])
 manifest=HERE/f'{phase}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=seeds) for (s,e,c),seeds in groups.items()]))
 q=preflight(manifest);write(HERE/f'{phase}_CACHE_PREFLIGHT.json',q);assert q['summary']['ambiguous']==0
 with connect(True) as db:missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
 assert len(missing)==len(rows),'Unexpected existing execution: exact reuse adapter required; stop, never rerun unknown'
 write(HERE/f'{phase}_MISSING_CASES.json',missing);seal(f'{phase}_PREFLIGHT.receipt.json',[manifest,HERE/f'{phase}_CACHE_PREFLIGHT.json',HERE/f'{phase}_protocol.json'],before_execution=True,planned=len(rows),missing=len(missing));print(phase,len(rows),flush=True)
def setup():
 prereg();from controllers import M_GRID,P_GRID
 q=read(D/'controller_universal_eta_schedule_v1/protocol.json');profiles={};base=copy.deepcopy(q['base_profiles'][REFS[0]]['payload'])
 for kind,grid in [('M',M_GRID),('P',P_GRID)]:
  for i,params in enumerate(grid):
   v=copy.deepcopy(base);v.update(controller_class=kind,nominal_parameters=list(params),nominal_implementation_sha256=sha(HERE/'controllers.py'),flow_checkpoint_path=None,flow_checkpoint_sha256=None,conditioning='nominal_TT_project_correct_reproject',execution_semantics='nonflow_TT_native_v1',jax_enable_x64=False,placement='TT');profiles[f'{kind}{i}']=dict(controller_uid=uid('ctl',v),payload=v)
 for n in REFS:
  v=copy.deepcopy(q['base_profiles'][n]['payload']);v.update(controller_class='Flow',conditioning='nominal_TT_project_correct_reproject',execution_semantics='nonflow_TT_native_v1',jax_enable_x64=False,placement='TT');profiles[n]=dict(controller_uid=uid('ctl',v),payload=v)
 p=dict(eta=q['eta'],base_profiles=profiles,controller_order=list(profiles),native_manifest=read(base['adapter_manifest_path']),environments=q['environments'],experiment_uid=EXP,controller_code_sha256=sha(HERE/'controllers.py'),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'));write(HERE/'base_protocol.json',p)
 from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
 native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');pilot=generate('round27_pilot',16,202610152710,native);dedup(pilot,'pilot');write(HERE/'PILOT_STATES.json',pilot)
 # Search historical TT identity/precision metadata without reading FF test labels.
 with connect(True) as db:hist=[dict(r) for r in db.execute("SELECT controller_uid,config_json FROM controller_config WHERE config_json LIKE '%TT%' OR config_json LIKE '%matched_TT%'")]
 exact=[]
 for h in hist:
  v=json.loads(h['config_json'])
  if v.get('flow_checkpoint_sha256') in [profiles[n]['payload']['flow_checkpoint_sha256'] for n in REFS] and v.get('jax_enable_x64') is False and v.get('execution_semantics')=='nonflow_TT_native_v1':exact.append(h['controller_uid'])
 write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),TT_profiles_scanned=len(hist),exact_full_semantic_matches=exact,why_no_old_exact_gate='Old TT profiles lack this complete nominal+precision+TT semantic fingerprint. Use prompt-authorized independent reference repeats; R10 FF labels are not TT labels.',compatible_FF_reference='R10 original immutable checkpoints/results; use after choices frozen only'))
 assert not exact,'Existing exact TT evidence found; audit it before running'
 gate_states=read(D/'controller_loco_real_flow_v1/protocol.json')['states'][:2];stage('gate',gate_states,REFS,['F10','F13'])
 seal('CONTROLLER_IMPLEMENTATION_FROZEN.receipt.json',[HERE/'controllers.py',HERE/'base_protocol.json',HERE/'test_controllers.py',HERE/'UNIT_TESTS.json',HERE/'PREREGISTRATION.md'],before_pilot_labels=True)
if __name__=='__main__':
 if sys.argv[1]=='setup':setup()
 elif sys.argv[1]=='pilot':
  assert read(HERE/'CONSISTENCY_GATE.json')['passed'];stage('pilot',read(HERE/'PILOT_STATES.json'),[f'{k}{i}' for k in ['M','P'] for i in range(6)],['F10','F13'])
 elif sys.argv[1]=='main':
  assert (HERE/'CHOICES_FROZEN.receipt.json').exists();p=read(HERE/'CAPABILITY_GATE.json');stage('main',read(HERE/'EVAL_STATES.json'),p['eligible']+REFS,['F10','F13','F2','S200','S300','S500'])
