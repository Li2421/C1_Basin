from common import *
from registry import register
from design import case
from shared_rollout_db.src.rollout_db import uid
import copy

def design(stage):
 prereg();gate=read(HERE/'CONSISTENCY_GATE.json')['scenes'];p=read(HERE/'gate_v2_protocol.json');p['states']=[];states=read(HERE/'STATES.json');profiles={};rows=[];cfg=read(HERE/'DESIGN_CONFIG.json');eligible=read(HERE/'CAPABILITY_GATE.json')['eligible'] if stage=='main' else {sc:list(read(HERE/'HISTORICAL_INVENTORY.json')['profiles'][sc]) for sc in states}
 for sc,names in eligible.items():
  if not names:continue
  assert gate[sc]['passed'];ss=states[sc]['pilot' if stage=='pilot' else 'eval'];p['states']+=ss
  for name in names:
   key=sc+'::'+name;cp=p['profiles'][key]
   for arm in list(dict.fromkeys(['F10',cfg['preferred'][sc]])) if stage=='pilot' else ARMS:
    profile=copy.deepcopy(cp)
    if arm=='nominal_only':profile['payload'].update(conditioning='nominal_only_without_safety_or_correction',execution_semantics='axis3_native_nominal_only',runtime_sha256=sha(HERE/'runtime.py'))
    elif arm.startswith('S'):
     H=int(profile['payload']['horizon']);t=round((3 if arm.endswith('t1') else 5)*H/7);profile['payload'].update(eta_schedule=dict(early=p['eta'][int(arm[1:3])],late=p['eta'][10],switch_step=t,action_index_rule='early if index<t else late'),execution_semantics='axis3_native_FF_absolute_schedule')
    profile['controller_uid']=uid('ctl',profile['payload']);profiles[profile['controller_uid']]=profile
    for s in ss:
     for seed in [0] if stage=='pilot' else [0,1]:rows.append(case(s,key,profile,arm,seed))
 p['eligible']=eligible;register(stage,p,rows,profiles)

def capability():
 cfg=read(HERE/'DESIGN_CONFIG.json');gate=read(HERE/'CONSISTENCY_GATE.json')['scenes'];inv=read(HERE/'HISTORICAL_INVENTORY.json');rows=read(HERE/'pilot_CASES.json');rates={};eligible={};excluded={};ambiguous={};status={}
 for sc in SCENES:
  eligible[sc]=[];excluded[sc]=[];ambiguous[sc]=[]
  if not gate[sc]['passed']:status[sc]='GAP: full-trajectory reproduction failed under mandatedFP32';continue
  for name in inv['scenes'][sc]['controllers']:
   key=sc+'::'+name;rates[key]={}
   for arm in list(dict.fromkeys(['F10',cfg['preferred'][sc]])):
    values=[read(HERE/'pilot/raw'/f"{c['rollout_uid']}.json") for c in rows if c['base_controller_name']==key and c['arm']==arm];assert len(values)==16;lo=sum(x['success'] for x in values)/16;hi=lo+sum(x['numerical_failure'] for x in values)/16;rates[key][arm]=[lo,hi]
   pairs=[rates[key][a] for a in ['F10',cfg['preferred'][sc]]];reject=all(x[1]<.15 for x in pairs) or all(x[0]>.98 for x in pairs);possible=all(x[0]<.15 for x in pairs) or all(x[1]>.98 for x in pairs)
   (excluded[sc] if reject else ambiguous[sc] if possible else eligible[sc]).append(name)
  status[sc]='PASS' if len(eligible[sc])>=2 else 'GAP: fewer than2 eligiblecontrollers'
  if len(eligible[sc])<2:eligible[sc]=[]
 write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),eligible=eligible,excluded=excluded,ambiguous=ambiguous,rates=rates,scene_status=status,no_tuning_or_replacement=True));print('Capability',{s:len(v) for s,v in eligible.items()},flush=True)
def pilot():design('pilot')
def evaluation():
 assert (HERE/'CHOICES_FROZEN.receipt.json').exists();design('main')
if __name__=='__main__':globals()[sys.argv[1]]()
