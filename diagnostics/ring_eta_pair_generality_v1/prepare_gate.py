from core import *
import copy
from collections import defaultdict
from shared_rollout_db.src.rollout_db import connect,transaction,uid,canonical
from shared_rollout_db.src.planner import preflight
prereg();old=read(R10/'protocol.json');rows=copy.deepcopy(read(R10/'gate_CASES.json'));profiles=copy.deepcopy(old['controllers']);expected=[]
for c in rows:
 olduid=c['rollout_uid'];historical=R10/'gate/raw'/f'{olduid}.json';assert historical.exists();expected.append(dict(path=str(historical),sha256=sha(historical),old_uid=olduid));v=profiles['ring_exchange'][c['controller_name']]['payload'];v['round22_validation_binding_sha256']=sha(HERE/'run.py');ctl=uid('ctl',v);profiles['ring_exchange'][c['controller_name']]['controller_uid']=ctl;c['controller_uid']=ctl;c['expected_uid']=olduid;c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=ctl,seed=c['seed_key']))
p=dict(old,experiment_uid=EXP,controllers=profiles,core_sha256=sha(HERE/'core.py'),runner_sha256=sha(R9/'run.py'),io_binding_sha256=sha(HERE/'run.py'));write(HERE/'gate_protocol.json',p);write(HERE/'gate_CASES.json',rows);write(HERE/'GATE_EXPECTED.json',expected);(HERE/'STAGE0_SUMMARY.md').write_text('Historical Ring runner and profiles reused; new random pair/state evidence gap. No target labels opened.\n')
with connect() as db,transaction(db):
 db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',(EXP,'Round22 eta pair generality',str(HERE),sha(HERE/'gate_protocol.json'),sha(R9/'run.py'),canonical(dict(gate32=True))))
 for c in rows:
  ctl=profiles['ring_exchange'][c['controller_name']];v=ctl['payload'];db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
groups=defaultdict(list)
for c in rows:groups[c['state_uid'],c['eta_uid'],c['controller_uid']].append(c['seed_key'])
write(HERE/'gate_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=k) for (s,e,c),k in groups.items()]));q=preflight(HERE/'gate_PLANNED_ROLLOUTS.json');write(HERE/'gate_CACHE_PREFLIGHT.json',q);assert q['summary']['ambiguous']==0 and q['summary']['genuinely_missing']==32;write(HERE/'gate_MISSING_CASES.json',rows);write(HERE/'gate_PREFLIGHT.receipt.json',dict(utc=utc(),manifest_sha256=sha(HERE/'gate_PLANNED_ROLLOUTS.json'),preflight_sha256=sha(HERE/'gate_CACHE_PREFLIGHT.json')));print('Gate32 preflight PASS')
