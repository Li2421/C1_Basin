from core import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical
from shared_rollout_db.src.planner import preflight
prereg();old=read(E8/'protocol.json');allcases=read(E8/'cases.json');cases=[c for c in allcases if c['controller_name']=='flow_isotropic' and c['split']=='train' and c['state_index']==0 and c['eta_index'] in [10,15] and c['seed']==0];assert len(cases)==8
assets={str(E8/'protocol.json'):sha(E8/'protocol.json'),str(E8/'raw_labels.npz'):sha(E8/'raw_labels.npz'),str(E8/'dataset.npz'):sha(E8/'dataset.npz'),str(ROOT/'docs/controller_generalization_audit_20261011.md'):sha(ROOT/'docs/controller_generalization_audit_20261011.md'),str(R9/'run.py'):sha(R9/'run.py'),str(R10/'core.py'):sha(R10/'core.py'),str(ROOT/'diagnostics/controller_phase_decomposition_v1/results.json'):sha(ROOT/'diagnostics/controller_phase_decomposition_v1/results.json')};assets.update(old['source_code_hashes'])
for sc in SCENES:
 for n in NAMES:
  v=old['controllers'][sc][n]['payload'];assets[v['flow_checkpoint_path']]=v['flow_checkpoint_sha256']
  for f,h in v['action_code_hashes'].items():assets[str(FIELD/f)]=h
for f,h in assets.items():assert sha(f)==h,f
expected=[];rows=[];profiles=copy.deepcopy(old['controllers'])
with connect(True) as db:
 for c in cases:
  r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r and not r['conflict_quarantined'];candidates=[E8/'raw'/f"{c['rollout_uid']}.json",ROOT/'diagnostics/controller_flow_endpoint_probe_v1/raw'/f"{c['rollout_uid']}.json"]
  path=next((q for q in candidates if q.exists()),None)
  if path is None:
   paths=list(ROOT.glob('diagnostics/*/raw/'+c['rollout_uid']+'.json'));assert paths;path=paths[0]
  raw=read(path);assert raw['case']==c and raw['outcome']==r['outcome'] and raw['episode_length']==r['episode_length'];expected.append(dict(case=c,path=str(path),sha256=sha(path),DB_source_file=r['original_source_file']))
  new=copy.deepcopy(c);v=copy.deepcopy(old['controllers'][c['scene']][c['controller_name']]['payload']);v['round13_io_validation_sha256']=sha(HERE/'run.py');name=c['controller_name']+'__gate';ctl=uid('ctl',v);profiles[c['scene']][name]=dict(controller_uid=ctl,payload=v);new.update(controller_name=name,base_controller_name=c['controller_name'],controller_uid=ctl,arm=f"F{c['eta_index']}",eta_a=c['eta'],eta_b=c['eta'],switch_step=v['horizon'],expected_rollout_uid=c['rollout_uid']);new['rollout_uid']=uid('roll',dict(state=new['state_uid'],eta=new['eta_uid'],controller=ctl,seed=new['seed_key']));rows.append(new)
p=dict(schema='round13_cross_scene_schedule_gate',experiment_uid=EXP,states=old['states'],controllers=profiles,base_profiles=old['controllers'],environments=old['environments'],eta=old['eta'],source_hashes=assets,run_sha256=sha(HERE/'run.py'),core_sha256=sha(HERE/'core.py'),horizons={sc:old['environments'][sc]['config']['config']['max_steps'] for sc in SCENES});freeze(HERE/'gate_protocol.json',p);freeze(HERE/'gate_CASES.json',rows);freeze(HERE/'GATE_EXPECTED.json',expected);freeze(HERE/'HISTORICAL_ASSET_INDEX.json',dict(utc=utc(),assets=assets,historical_gate_records=expected,reusable='E8 native frozen FF controllers and fixed-eta evidence; R9/10 schedule semantics; phase effects',missing='160 genuinely new states and13440 seven-arm Q4 target episodes',no_training=True))
with connect() as db,transaction(db):
 db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP+'__gate','Round13 historical exact32 gate',str(HERE/'gate'),sha(HERE/'gate_protocol.json'),sha(HERE/'run.py'),canonical(dict(validation_only=True))))
 for sc in SCENES:
  for n,ctl in profiles[sc].items():
   if not n.endswith('__gate'):continue
   v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
groups=defaultdict(list)
for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
freeze(HERE/'gate_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys) for (s,e,c),keys in groups.items()]));pre=preflight(HERE/'gate_PLANNED_ROLLOUTS.json');write(HERE/'gate_CACHE_PREFLIGHT.json',pre);assert pre['summary']['ambiguous']==0;assert pre['summary']['genuinely_missing']==8;freeze(HERE/'gate_MISSING_CASES.json',rows)
print('Historical evidence and gate preflight PASS')
