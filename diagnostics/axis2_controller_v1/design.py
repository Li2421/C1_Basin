from common import *
from registry import normalized_state,register
from shared_rollout_db.src.rollout_db import connect,uid,canonical,eta_identity
import copy
SOURCES=[('2a','ring_independent_flow_confirmation_v2','main',14),('2b','controller_new_checkpoint_confirmation_v1/fp32_n8','main',8),('2cN2','ring_critic_new_recipe_confirmation_v1','wave1',13),('2cTEST','ring_eta2_gap_v1','test',18)]
def case(s,key,profile,arm,seed,placement,cohort,old=None):
 q=read(D/'controller_universal_eta_schedule_v1/protocol.json');a=[0.,0.,0.] if arm in ['nominal_only','safety_baseline'] else q['eta'][13 if arm.startswith('S') else int(arm[1:])];b=q['eta'][10] if arm.startswith('S') else a;t=int(arm[1:]) if arm.startswith('S') else 700;sk=canonical({'future_index':seed});eid=eta_identity(a)[0];cid=profile['controller_uid'];c=dict(rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),state_uid=s['state_uid'],original_state_uid=s['original_state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=key,arm=arm,seed=seed,eta_a=a,eta_b=b,switch_step=t,placement=placement,cohort=cohort,state_index=s.get('study_index',0))
 if old:c['reference_rollout_uid']=old['rollout_uid'];c['original_seed_key']=old['seed_key']
 return c

def history_and_gate():
 prereg();q=read(D/'controller_universal_eta_schedule_v1/protocol.json');native=read(D/'other_controller_classes_v1/base_protocol.json')['native_manifest'];p=dict(states=[],profiles={},native_manifest=native,eta=q['eta']);rows=[];inventory={};baselines=[];actual_states={};sha_set=set();allstateuids={}
 for cohort,folder,phase,count in SOURCES:
  h=D/folder;source=read(h/'protocol.json');refs=read(h/f'{phase}_CASES.json');order=source['controller_order']
  if cohort=='2cN2':order=[r['name'] for r in read(h/'TRAINING_PLAN.json') if r.get('group')=='N2']
  elif cohort=='2cTEST':order=[r['name'] for r in read(h/'TRAINING_PLAN.json') if r.get('split')=='TEST']
  assert len(order)==count,(cohort,len(order));selected=[c for c in refs if c['base_controller_name'] in order];wanted=sorted({c['state_uid'] for c in selected})[:32];assert len(wanted)==32
  states={s['state_uid']:s for s in source['states']};assert set(wanted)<=set(states);lookup={(c['base_controller_name'],c['state_uid'],c['seed'],c['arm']):c for c in selected};gateorder=sorted(order)[:4];normstates={sid:normalized_state(states[sid]) for sid in wanted};actual_states[cohort]=list(normstates.values());allstateuids[cohort]=wanted
  for name in order:
   profile=source['base_profiles'][name];key=cohort+'::'+name;p['profiles'][key]=profile;sha_set.add(profile['payload']['flow_checkpoint_sha256'])
   for sid in wanted:
    for seed in [0,1]:
     original=lookup[(name,sid,seed,'F10')]
     for arm in ['nominal_only','safety_baseline']:
      cp=copy.deepcopy(profile)
      if arm=='nominal_only':cp['payload'].update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='axis2_original_Flow_nominal_only_v1',nominal_implementation_sha256=sha(HERE/'runtime.py'));cp['controller_uid']=uid('ctl',cp['payload'])
      baselines.append(dict(cohort=cohort,key=key,profile=cp,state=normstates[sid],arm=arm,seed=seed,original=original))
  for sid in wanted[:2]:
   gs=normalized_state(states[sid],validation='round33_gate_'+cohort);p['states'].append(gs)
   for name in gateorder:
    key=cohort+'::'+name
    for seed in [0,1]:
     original=lookup[(name,sid,seed,'F10')];c=case(gs,key,p['profiles'][key],'F10',seed,'FF',cohort,original);c['reference_path']=str(h/phase/'raw'/f"{original['rollout_uid']}.json");assert Path(c['reference_path']).exists();rows.append(c)
  inventory[cohort]=dict(source=str(h),controllers=order,state_uids=wanted,original_method_sources=[str(x) for x in h.glob('*CHOICE*') if x.is_file()],original_gate_reference_count=16,new_baselines=count*32*4)
 h=D/'other_controller_classes_v1';source=read(h/'main_protocol.json');refs=read(h/'main_CASES.json');names=sorted(source['controller_order'])[:4];wanted=sorted({c['state_uid'] for c in refs})[:2];states={s['state_uid']:s for s in source['states']};lookup={(c['base_controller_name'],c['state_uid'],c['seed'],c['arm']):c for c in refs}
 for name in names:p['profiles']['TT::'+name]=source['base_profiles'][name]
 for sid in wanted:
  gs=normalized_state(states[sid],validation='round33_gate_TT');p['states'].append(gs)
  for name in names:
   for seed in [0,1]:
    original=lookup[(name,sid,seed,'F10')];c=case(gs,'TT::'+name,p['profiles']['TT::'+name],'F10',seed,'TT','TT',original);c['reference_path']=str(h/'main/raw'/f"{original['rollout_uid']}.json");rows.append(c)
 assert len(rows)==80 and len(baselines)==6784
 # Audit all historical eta0 identities for these exact checkpoints, before any new baseline.
 candidates=[]
 with connect(True) as db:
  for r in db.execute('SELECT controller_uid,config_json FROM controller_config'):
   try:v=json.loads(r['config_json'])
   except (ValueError,TypeError):continue
   if v.get('flow_checkpoint_sha256') in sha_set:candidates.append(r['controller_uid'])
  zero=eta_identity([0.,0.,0.])[0];oldzero=[]
  for cid in candidates:
   oldzero.extend(dict(r) for r in db.execute('SELECT rollout_uid,state_uid,controller_uid,seed_key,original_source_file,numerical_failure FROM rollout WHERE controller_uid=? AND eta_uid=?',(cid,zero)))
 write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),cohorts=inventory,exact_checkpoint_profiles_scanned=len(candidates),historical_eta0_records=oldzero,new_gap='Only nominal/safety labels missing. Original correction labels/choices remain reusable. No checkpoint retraining or old R5 probing.',training_history_file='TRAINING_HISTORY_AUDIT.json',R29_private_DB_provenance=True,R30_completed_with_source_and_chart_qualifications=True,normalization='Canonical transport identity only; physical state and original uid/random stream unchanged. Original full key stored per case/mapping.'))
 assert not oldzero,'Historical eta0 found: audit/reuse before submitting; no silent reruns'
 write(HERE/'BASELINE_DESIGN.json',baselines);write(HERE/'BASELINE_STATES.json',actual_states);write(HERE/'SOURCE_INVENTORY.json',inventory)
 profiles={c['controller_uid']:p['profiles'][c['base_controller_name']] for c in rows};register('gate',p,rows,profiles)

def baselines():
 checks=read(HERE/'CONSISTENCY_GATE.json');assert checks['passed'];source=read(HERE/'gate_protocol.json');p=dict(source,states=[],profiles=copy.deepcopy(source['profiles']));rows=[];profiles={};seen=set()
 for v in read(HERE/'BASELINE_DESIGN.json'):
  s=v['state'];key=v['key'];cp=v['profile'];
  if s['state_uid'] not in seen:p['states'].append(s);seen.add(s['state_uid'])
  rows.append(case(s,key,cp,v['arm'],v['seed'],'FF',v['cohort'],v['original']));profiles[cp['controller_uid']]=cp
 assert len(rows)==6784;register('baseline',p,rows,profiles)
if __name__=='__main__':globals()[sys.argv[1]]()
