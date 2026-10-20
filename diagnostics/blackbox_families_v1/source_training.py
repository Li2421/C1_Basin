from common import *
from design import case
from registry import register,normalized_state
from shared_rollout_db.src.rollout_db import uid,eta_identity,canonical
import copy

def flow_execute(c,p,cache,phase):
 if c['base_controller_name']=='base_u_v10_local_macflow' and c['base_controller_name'] not in cache:
  from validation_v2.adapters import FourRingRuntime
  import jax
  cache[c['base_controller_name']]=FourRingRuntime('ring_exchange');jax.config.update('jax_enable_x64',False);assert sha(cache[c['base_controller_name']].checkpoint)==p['profiles'][c['base_controller_name']]['payload']['flow_checkpoint_sha256']
 from episode import execute
 return execute(c,p,cache,phase)

def gate():
 p=dict(states=[],profiles=read(D/'blackbox_addendum_v1/ORIGINAL_PROFILES.json'),eta=read(D/'controller_universal_eta_schedule_v1/protocol.json')['eta'],native_manifest=read(D/'other_controller_classes_v1/base_protocol.json')['native_manifest'],types={});oldp=read(D/'axis4a_geometry_horizon_v1/gate_protocol.json');oldstates={s['state_uid']:s for s in oldp['states']};rows=[];profiles={}
 for ref in read(D/'axis4a_geometry_horizon_v1/gate_CASES.json'):
  s=normalized_state(oldstates[ref['state_uid']],validation='round38_source_FF_reproduction');name=ref['base_controller_name'];p['states'].append(s) if s['state_uid'] not in {x['state_uid'] for x in p['states']} else None;c,cp=case(s,name,p['profiles'][name],p,ref['arm'],ref['seed']);c.update(placement='FF',cohort='historical_flow',reference_path=ref['reference_path'],reference_sha256=ref['reference_sha256']);rows.append(c);profiles[cp['controller_uid']]=cp
 assert len(rows)==16;register('gate_flow',p,rows,profiles)

def design():
 assert read(HERE/'FLOW_REPRODUCTION_GATE.json')['passed'];p=dict(states=read(HERE/'TUNING_STATES.json'),profiles=read(D/'blackbox_addendum_v1/ORIGINAL_PROFILES.json'),eta=read(D/'controller_universal_eta_schedule_v1/protocol.json')['eta'],native_manifest=read(D/'other_controller_classes_v1/base_protocol.json')['native_manifest'],types={});source=read(D/'axis2_controller_v1/main_protocol.json');p['profiles'].update({n:source['profiles'][n] for n in [f'G{i}' for i in range(6)]+[f'DM{i}' for i in range(6)]});rows=[];profiles={};names=list(p['profiles']);assert len(names)==26
 for name in names:
  for s in p['states']:
   for arm in ['F10','F13','S500']:
    c,cp=case(s,name,p['profiles'][name],p,arm);c['placement']='TT' if name.startswith(('G','DM')) else 'FF';rows.append(c);profiles[cp['controller_uid']]=cp
 register('source_training',p,rows,profiles)
if __name__=='__main__':globals()[sys.argv[1]]()
