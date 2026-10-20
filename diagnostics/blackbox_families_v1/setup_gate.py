from common import *
import copy
from registry import register,normalized_state
from shared_rollout_db.src.rollout_db import uid,eta_identity,canonical
p={'states':[],'profiles':{},'eta':read(D/'controller_universal_eta_schedule_v1/protocol.json')['eta'],'native_manifest':read(D/'other_controller_classes_v1/main_protocol.json')['native_manifest']};rows=[];audit={}
for group,folder,field,names in [('historical_new_types','axis2_controller_v1','profiles',['G0','G1','DM0','DM1']),('historical_centralized_sampling','other_controller_classes_v1','base_profiles',['M0','M1','M2','M3'])]:
 h=D/folder;oldp=read(h/'main_protocol.json');allrows=read(h/'main_CASES.json');states={s['state_uid']:s for s in oldp['states']};audit[group]={'protocol':str(h/'main_protocol.json'),'protocol_sha256':sha(h/'main_protocol.json'),'controllers':names,'rows':[]}
 for name in names:
  p['profiles'][name]=oldp[field][name]
  refs=sorted([c for c in allrows if c['base_controller_name']==name and c['arm']=='F10'],key=lambda c:(c['state_uid'],c['seed']))[:4];assert len(refs)==4
  for ref in refs:
   s=normalized_state(states[ref['state_uid']],validation='round38_reproduction_'+group)
   if s['state_uid'] not in {s['state_uid'] for s in p['states']}:p['states'].append(s)
   cp=p['profiles'][name];eid=eta_identity(p['eta'][10])[0];sk=canonical({'future_index':ref['seed']});rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cp['controller_uid'],seed=sk));path=h/'main/raw'/f"{ref['rollout_uid']}.json";old=read(path);assert Path(old['trace_path']).exists()
   rows.append(dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cp['controller_uid'],seed_key=sk,base_controller_name=name,arm='F10',seed=ref['seed'],eta_a=p['eta'][10],eta_b=p['eta'][10],switch_step=700,placement='TT',cohort=group,reference_path=str(path),reference_sha256=sha(path)))
   audit[group]['rows'].append({'reference_path':str(path),'trace_path':old['trace_path']})
write(HERE/'HISTORY_GATE_AUDIT.json',audit)
register('gate',p,rows,{x['controller_uid']:x for x in p['profiles'].values()})
