"""Correct scenario transport for already generated traces, without any rollout."""
from common import *
from registry import register,normalized_state
from design import case
from shared_rollout_db.src.rollout_db import connect
import copy
oldp=read(HERE/'gate_flow_protocol.json');p=copy.deepcopy(oldp);p['states']=[];cases=[];profiles={};mapping=[]
for c in read(HERE/'gate_flow_CASES.json'):
 r=read(HERE/'gate_flow/raw'/f"{c['rollout_uid']}.json");s=copy.deepcopy(next(s for s in oldp['states'] if s['state_uid']==c['state_uid']));s['scenario_uid']=p['profiles'][c['base_controller_name']]['payload']['scenario_uid'];s=normalized_state(s,validation='round38_source_FF_reproduction_correct_transport')
 if s['state_uid'] not in {x['state_uid'] for x in p['states']}:p['states'].append(s)
 new,cp=case(s,c['base_controller_name'],p['profiles'][c['base_controller_name']],p,c['arm'],c['seed']);new.update(placement='FF',cohort='historical_flow',reference_path=c['reference_path'],reference_sha256=c['reference_sha256']);cases.append(new);profiles[cp['controller_uid']]=cp;mapping.append(dict(old_rollout_uid=c['rollout_uid'],new_rollout_uid=new['rollout_uid'],source_raw=str(HERE/'gate_flow/raw'/f"{c['rollout_uid']}.json"),source_sha256=sha(HERE/'gate_flow/raw'/f"{c['rollout_uid']}.json")))
register('gate_flow_corrected',p,cases,profiles)
for c,m in zip(cases,mapping):
 r=read(m['source_raw']);r.update(case=c,**{k:c[k] for k in ['rollout_uid','state_uid','eta_uid','controller_uid','seed_key']},scenario_uid=next(s for s in p['states'] if s['state_uid']==c['state_uid'])['scenario_uid'],new_physical_steps=0,transport_correction=m);path=HERE/'gate_flow_corrected/raw'/f"{c['rollout_uid']}.json";write(path,r);cache_writer.append_journal([r],EXP,'flow_transport_correction')
write(HERE/'FLOW_TRANSPORT_CORRECTION.json',dict(utc=utc(),reason='Round35 geometry-alias scenario identity differs from original frozen Flow controller scenario UID despite identical H700 radius0.9 physical config. First journal classified SUMMARY_ONLY, no rollout rows inserted. Re-register state in original controller scenario; keep original uid/random stream/positions/goals/velocities and original traces exactly. No controller identity change; no existing row changed; zero repeated physical steps.',mapping=mapping,new_physical_steps=0))
