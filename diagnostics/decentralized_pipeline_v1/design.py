from core import *
import copy
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical,eta_identity
from shared_rollout_db.src.planner import preflight
EXP='exp_decentralized_pipeline_v1'
def signature(v):
 try:
  arrays=[]
  for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
   a=np.asarray(next(v[k] for k in keys if k in v),np.float64).copy();assert a.ndim==2 and a.shape[1]==2;a[a==0]=0;arrays.append((a.shape,a.tobytes().hex()))
  return uid('physical',dict(arrays=arrays,timestep=float(v.get('timestep',v.get('step',0)))))
 except (TypeError,KeyError,StopIteration,AssertionError):return None
def setup():
 prereg();assert read(HERE/'UNIT_TESTS.json')['status']=='PASS';q=read(D/'controller_universal_eta_schedule_v1/protocol.json');old=read(D/'other_controller_classes_v1/gate_protocol.json');refs=read(D/'other_controller_classes_v1/gate_CASES.json');p=dict(experiment_uid=EXP,base_profiles=q['base_profiles'],controller_order=q['controller_order'],eta=q['eta'],environments=q['environments'],pipeline_sha256=sha(HERE/'pipeline.py'),runner_sha256=sha(HERE/'run.py'),preregistration_sha256=sha(HERE/'PREREGISTRATION.md'))
 with connect(True) as db:
  allstates=db.execute('SELECT state_uid,content_hash,physical_state_json FROM state').fetchall();metadata=[dict(r) for r in db.execute("SELECT controller_uid,config_json FROM controller_config WHERE config_json LIKE '%decentralized%' OR config_json LIKE '%D-shared%'")]
 write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),sources=['R10 original14 immutable profiles/checkpoints','R27 central TT16 gate raw+traces','native CBF/basis/expert/video implementation'],decentralized_controller_metadata_candidates=metadata,why_new='No existing per-robot TT safety/correction evidence under this pipeline; FF/central labels not interchangeable. New main states require new labels; exact preflight still mandatory.'))
 from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
 native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');pilot=generate('round28_pilot_reserved',8,202610280000,native);pilotsigs={signature(s['physical']) for s in pilot};ids={r['state_uid'] for r in allstates};contents={r['content_hash'] for r in allstates};sigs=set();unresolved=0
 for row in allstates:
  v=json.loads(row['physical_state_json']) if row['physical_state_json'] else None;sig=signature(v)
  if sig is None:unresolved+=1
  else:sigs.add(sig)
 states=[];rejected=[]
 for seed in range(202610281000,202610282000):
  s=generate('round28_eval',1,seed,native)[0];sig=signature(s['physical'])
  if s['state_uid'] in ids or s['content_hash'] in contents or sig in sigs or sig in pilotsigs:rejected.append(dict(seed=seed,state_uid=s['state_uid'],reason='DB/pilot duplicate'));continue
  s['study_index']=len(states);states.append(s);sigs.add(sig)
  if len(states)==32:break
 assert len(states)==32 and 202610280007<202610281000;write(HERE/'PILOT_RESERVED_STATES.json',pilot);write(HERE/'EVAL_STATES.json',states);write(HERE/'STATE_DEDUP_AUDIT.json',dict(utc=utc(),passed=True,DB_rows=len(allstates),unresolved_physical=unresolved,selected=32,rejected=rejected,pilot_seed_interval=[202610280000,202610280007],eval_candidate_interval=[202610281000,202610281999],selected_seeds=[s['metadata']['draw_seed'] for s in states],pilot_eval_disjoint=True));write(HERE/'base_protocol.json',p)
 gate=dict(p,states=old['states'],cases=refs,reference_folder=str(D/'other_controller_classes_v1'));write(HERE/'gate_protocol.json',gate);manifest=HERE/'gate_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in refs]));pf=preflight(manifest);write(HERE/'gate_CACHE_PREFLIGHT.json',pf);assert pf['summary']['genuinely_missing']==0 and pf['summary']['ambiguous']==0
 seal('IMPLEMENTATION_FROZEN.receipt.json',[HERE/'pipeline.py',HERE/'run.py',HERE/'test_components.py',HERE/'UNIT_TESTS.json',HERE/'base_protocol.json',HERE/'EVAL_STATES.json',HERE/'STATE_DEDUP_AUDIT.json',HERE/'PREREGISTRATION.md'],before_new_main_labels=True);print('Design frozen,16 cached central consistency cases,32 fresh states')
def main_design():
 prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed'];p=read(HERE/'base_protocol.json');p.update(states=read(HERE/'EVAL_STATES.json'),controllers={},arms=['F10','F13','S300','S500'],pipelines=['shared','local','central'],runner_sha256=sha(HERE/'run.py'));rows=[];r27=read(D/'other_controller_classes_v1/base_protocol.json')['base_profiles']
 for n in p['controller_order']:
  for mode in ['shared','local','central','rhr','rhr_f10']:
   for arm in (['B-RHR'] if mode=='rhr' else ['B-RHR+F10'] if mode=='rhr_f10' else p['arms']):
    v=copy.deepcopy(p['base_profiles'][n]['payload']);v.update(controller_class='Flow',conditioning='nominal_TT_project_correct_reproject',execution_semantics='nonflow_TT_native_v1',placement='TT',jax_enable_x64=False)
    if mode!='central':v.update(conditioning='per_robot_TT_project_local_basis_reproject',execution_semantics='round28_decentralized_TT_v1',decentralized_mode=mode,decentralized_pipeline_sha256=p['pipeline_sha256'],neighbor_velocity_assumption='zero, half responsibility',sense_radius=1.5 if mode=='local' else 'all',right_hand_rule=dict(speed_threshold=.01,consecutive_steps=40,amplitude=.15,tangent='clockwise central obstacle') if mode.startswith('rhr') else None)
    if arm.startswith('S'):a=13;b=10;t=int(arm[1:]);v['eta_schedule']=dict(eta_a=p['eta'][a],eta_b=p['eta'][b],switch_step=t,action_index_rule='a for index<t else b')
    else:a=10 if arm in ['F10','B-RHR+F10'] else 13 if arm=='F13' else None;b=a;t=700
    eta_a=p['eta'][a] if a is not None else [0.,0.,0.];eta_b=p['eta'][b] if b is not None else [0.,0.,0.];ctl=uid('ctl',v);cname=n+'__'+mode+'__'+arm;p['controllers'][cname]=dict(controller_uid=ctl,payload=v)
    for s in p['states']:
     for seed in [0,1]:
      sk=canonical(dict(future_index=seed,future_root=2026100403,rng_namespace=s['rng_namespace']));eid=eta_identity(eta_a)[0];rows.append(dict(rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk)),state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,controller_name=cname,base_controller_name=n,pipeline=mode,arm=arm,state_index=s['study_index'],seed=seed,eta_a=eta_a,eta_b=eta_b,switch_step=t))
 assert len(rows)==12544 and len({c['rollout_uid'] for c in rows})==12544;write(HERE/'main_protocol.json',p);write(HERE/'main_CASES.json',rows)
 with connect() as db,transaction(db):
  db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round28 decentralized pipelines',str(HERE),sha(HERE/'main_protocol.json'),sha(HERE/'pipeline.py'),canonical(dict(preregistered=True,requires_video=True))))
  for s in p['states']:db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',(s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments']['ring_exchange']['config']),canonical(s['metadata']),'CONTENT_EXACT'))
  for c in p['controllers'].values():
   v=c['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(c['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(native=v['safety_projection_sha256'],mode=v.get('decentralized_mode','central'),local=p['pipeline_sha256'] if v.get('decentralized_mode') else None)),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
  for eta in [[0.,0.,0.],p['eta'][10],p['eta'][13]]:
   eid,values,hexes=eta_identity(eta);db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*values,hexes))
 groups={}
 for c in rows:groups.setdefault((c['state_uid'],c['eta_uid'],c['controller_uid']),[]).append(c['seed_key'])
 manifest=HERE/'main_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=seeds) for (s,e,c),seeds in groups.items()]));pf=preflight(manifest);write(HERE/'main_CACHE_PREFLIGHT.json',pf);assert pf['summary']['ambiguous']==0
 with connect(True) as db:missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
 assert len(missing)==len(rows),'Exact existing evidence found: audit/reuse, never rerun unknown';write(HERE/'main_MISSING_CASES.json',missing);seal('main_PREFLIGHT.receipt.json',[manifest,HERE/'main_CACHE_PREFLIGHT.json',HERE/'main_protocol.json'],before_submission=True,planned=len(rows),missing=len(missing));print('main preflight',len(rows))
if __name__=='__main__':globals()[sys.argv[1]]()
