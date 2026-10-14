from core import *
import copy
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical
from shared_rollout_db.src.planner import preflight
from adapter import FourFlowAdapter
import jax
prereg();old=read(E8/'protocol.json');cases=[c for c in read(E8/'cases.json') if c['scene']=='four_way_intersection' and c['controller_name']=='flow_isotropic' and c['split']=='train' and c['state_index'] in [0,1] and c['eta_index'] in [10,15] and c['seed'] in [0,1]];assert len(cases)==8
meta=read(HERE/'COMPATIBILITY_METADATA_AUDIT.json');forward=[]
for row in meta['rows']:
 if not row['compatible_metadata'] or row['kind']=='final' and not row['param_different_from_best']:continue
 rt=FourFlowAdapter(row['path']);jax.config.update('jax_enable_x64',False);state=next(s for s in old['states'] if s['state_uid']==cases[0]['state_uid']);env=rt.make_env(state);key=jax.random.PRNGKey(2026101401);a=np.asarray(rt.native_fn(rt.observation(env),key));b=np.asarray(rt.native_fn(rt.observation(env),key));assert a.tobytes()==b.tobytes() and np.isfinite(a).all();forward.append(dict(run=row['run'],kind=row['kind'],forward_bytes_identical=True,first_action=a.tolist(),fp32=not jax.config.x64_enabled))
write(HERE/'COMPATIBILITY_GATE.json',dict(utc=utc(),passed=bool(forward),compatible=forward,metadata_audit_sha256=sha(HERE/'COMPATIBILITY_METADATA_AUDIT.json')))
assets=copy.deepcopy(old['source_code_hashes']);assets[str(HERE/'adapter.py')]=sha(HERE/'adapter.py');profiles=copy.deepcopy(old['controllers']);expected=[];rows=[]
for c in cases:
 path=ROOT/'diagnostics/controller_flow_endpoint_probe_v1/raw'/f"{c['rollout_uid']}.json";assert path.exists();expected.append(dict(case=c,path=str(path),sha256=sha(path)))
 v=copy.deepcopy(old['controllers'][c['scene']]['flow_isotropic']['payload']);v.update(round14_adapter_sha256=sha(HERE/'adapter.py'),round14_gate_run_sha256=sha(HERE/'run.py'));name='v13_best__gate';ctl=uid('ctl',v);profiles[c['scene']][name]=dict(controller_uid=ctl,payload=v);new=copy.deepcopy(c);new.update(controller_name=name,base_controller_name=name,controller_uid=ctl,arm=f"F{c['eta_index']}",eta_a=c['eta'],eta_b=c['eta'],switch_step=1100,expected_rollout_uid=c['rollout_uid']);new['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=ctl,seed=c['seed_key']));rows.append(new)
p=dict(experiment_uid=EXP,states=old['states'],controllers=profiles,base_profiles=profiles,environments=old['environments'],eta=old['eta'],source_hashes=assets,run_sha256=sha(HERE/'run.py'),core_sha256=sha(HERE/'core.py'));freeze(HERE/'gate_protocol.json',p);freeze(HERE/'gate_CASES.json',rows);freeze(HERE/'GATE_EXPECTED.json',expected)
with connect() as db,transaction(db):
 db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP+'__gate','Round14B adapter exact8 gate',str(HERE/'gate'),sha(HERE/'gate_protocol.json'),sha(HERE/'run.py'),canonical(dict(validation_only=True))))
 for name,ctl in profiles['four_way_intersection'].items():
  if name!='v13_best__gate':continue
  v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
freeze(HERE/'gate_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=c['state_uid'],eta_uid=c['eta_uid'],controller_uid=c['controller_uid'],seed_keys=[c['seed_key']]) for c in rows]));pre=preflight(HERE/'gate_PLANNED_ROLLOUTS.json');write(HERE/'gate_CACHE_PREFLIGHT.json',pre);assert pre['summary']['genuinely_missing']==8;freeze(HERE/'gate_MISSING_CASES.json',rows)
print('Compatibility forward and gate preflight PASS')
