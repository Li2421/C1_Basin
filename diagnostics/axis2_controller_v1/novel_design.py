from common import *
from registry import normalized_state,register
from design import case
from shared_rollout_db.src.rollout_db import connect,uid
import copy

def physical_signature(v):
 try:
  arrays=[]
  for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
   a=np.asarray(next(v[k] for k in keys if k in v),np.float64).copy();assert a.shape==(4,2);a[a==0]=0;arrays.append(a.tobytes().hex())
  return uid('physical',dict(arrays=arrays,timestep=float(v.get('timestep',v.get('step',0)))))
 except (TypeError,KeyError,StopIteration,AssertionError):return None

def setup():
 prereg();assert read(HERE/'UNIT_TESTS.json')['passed'];assert read(HERE/'CONSISTENCY_GATE.json')['groups']['TT']['passed'];assert (HERE/'MIXED_FROZEN.receipt.json').exists();cfg=read(HERE/'DESIGN_CONFIG.json');q=read(D/'controller_universal_eta_schedule_v1/protocol.json');r27=read(D/'other_controller_classes_v1/base_protocol.json');p=dict(profiles={},states=[],eta=q['eta'],native_manifest=r27['native_manifest'],controller_order=[],types={})
 for g in cfg['G']:
  folder=HERE/'gaussian'/g['name'];assert (folder/'COMPLETE.receipt.json').exists();v=copy.deepcopy(q['base_profiles'][g['recipe']]['payload']);v.update(controller_class='G',flow_checkpoint_path=str(folder/'best.msgpack'),flow_checkpoint_sha256=sha(folder/'best.msgpack'),normalization_sha256=sha(folder/'NORMALIZATION.json'),Gaussian_config_sha256=sha(folder/'CONFIG.json'),source_Flow_recipe=g['recipe'],conditioning='nominal_TT_project_correct_reproject',execution_semantics='nonflow_TT_native_v1',placement='TT',chain='TT',nominal_implementation_sha256=sha(HERE/'controllers.py'),jax_enable_x64=False);p['profiles'][g['name']]=dict(controller_uid=uid('ctl',v),payload=v)
 for kind in ['DM','VO','PT']:
  for i,params in enumerate(cfg[kind]):
   name=kind+str(i);v=copy.deepcopy(r27['base_profiles']['M0']['payload']);v.update(controller_class=kind,nominal_parameters=params+cfg['PT_repulsion'] if kind=='PT' else params,nominal_implementation_sha256=sha(HERE/'controllers.py'),nominal_observation_chart='physical world positions/velocities and own goal',placement='TT',chain='TT',jax_enable_x64=False);p['profiles'][name]=dict(controller_uid=uid('ctl',v),payload=v)
 p['controller_order']=list(p['profiles']);p['types']={n:p['profiles'][n]['payload']['controller_class'] for n in p['profiles']};assert len(p['profiles'])==24
 from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
 from ring_exchange.environment import RingExchangeEnv,LocalFrameConfig
 native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');envcfg=LocalFrameConfig(**native['environment']['config']['config']);pilot=generate(cfg['pilot_split'],16,cfg['pilot_seed_interval'][0],native);evals=generate(cfg['eval_split'],32,cfg['eval_seed_interval'][0],native)
 with connect(True) as db:old=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
 oldids={r['state_uid'] for r in old};contents={r['content_hash'] for r in old};signatures=set();unresolved=0
 for row in old:
  sig=physical_signature(json.loads(row['physical_state_json']) if row['physical_state_json'] else None)
  if sig is None:unresolved+=1
  else:signatures.add(sig)
 hits=[]
 for s in pilot+evals:
  RingExchangeEnv(envcfg).reset(**{k:np.asarray(v) for k,v in s['physical'].items() if k!='timestep'});sig=physical_signature(s['physical'])
  if s['state_uid'] in oldids or s['content_hash'] in contents or sig in signatures:hits.append(s['state_uid'])
  signatures.add(sig)
 write(HERE/'NEW_STATES_DEDUP.json',dict(utc=utc(),passed=not hits,pilot=16,evaluation=32,DB_rows=len(old),unrecoverable_physical=unresolved,duplicates=hits));assert not hits
 pilot=[normalized_state(s) for s in pilot];evals=[normalized_state(s) for s in evals];write(HERE/'PILOT_STATES.json',pilot);write(HERE/'EVAL_STATES.json',evals);write(HERE/'NEW_BASE_PROTOCOL.json',p);p.update(states=pilot,phase='pilot');rows=[]
 for name in p['controller_order']:
  for s in pilot:
   for arm in ['F10','F13']:rows.append(case(s,name,p['profiles'][name],arm,0,'TT',p['types'][name]))
 assert len(rows)==768;register('pilot',p,rows,p['profiles']);seal('NEW_CONTROLLERS_FROZEN.receipt.json',[HERE/'NEW_BASE_PROTOCOL.json',HERE/'controllers.py',HERE/'runtime.py',HERE/'UNIT_TESTS.json',HERE/'PILOT_STATES.json',HERE/'EVAL_STATES.json',HERE/'NEW_STATES_DEDUP.json',HERE/'MIXED_FROZEN.receipt.json'],before_new_type_pilot_labels=True)

def capability():
 p=read(HERE/'pilot_protocol.json');rates={};eligible=[];uncertain=[];excluded=[]
 for name in p['controller_order']:
  rates[name]={}
  for arm in ['F10','F13']:
   rows=[c for c in read(HERE/'pilot_CASES.json') if c['base_controller_name']==name and c['arm']==arm];r=[read(HERE/'pilot/raw'/f"{c['rollout_uid']}.json") for c in rows];assert len(r)==16;low=sum(x['success'] for x in r)/16;high=low+sum(x['numerical_failure'] for x in r)/16;rates[name][arm]=[low,high]
  a,b=rates[name]['F10'],rates[name]['F13'];reject=(a[1]<.15 and b[1]<.15) or (a[0]>.98 and b[0]>.98);ambiguous=((a[0]<.15 and b[0]<.15) or (a[1]>.98 and b[1]>.98)) and not reject
  if reject:excluded.append(name)
  elif ambiguous:uncertain.append(name)
  else:eligible.append(name)
 bytype={t:[n for n in eligible if p['types'][n]==t] for t in ['G','DM','VO','PT']};write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),eligible=eligible,excluded=excluded,ambiguous=uncertain,rates=rates,eligible_by_type=bytype,type_status={t:'PASS' if len(v)>=4 else 'INCONCLUSIVE' for t,v in bytype.items()},no_replacement_or_tuning=True));print('Eligible',len(eligible),{t:len(v) for t,v in bytype.items()},flush=True)

def main_design():
 assert (HERE/'CHOICES_FROZEN.receipt.json').exists();cap=read(HERE/'CAPABILITY_GATE.json');p=read(HERE/'NEW_BASE_PROTOCOL.json');p.update(states=read(HERE/'EVAL_STATES.json'),controller_order=cap['eligible'],phase='main');profiles={};rows=[]
 for name in p['controller_order']:
  for arm in ['nominal_only','safety_baseline']+MENU:
   cp=copy.deepcopy(p['profiles'][name]);v=cp['payload']
   if arm=='nominal_only':v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='axis2_novel_nominal_only_v1',nominal_runtime_sha256=sha(HERE/'runtime.py'))
   elif arm.startswith('S'):v['eta_schedule']=dict(eta_a=p['eta'][13],eta_b=p['eta'][10],switch_step=int(arm[1:]),action_index_rule='a if index<t else b')
   cp['controller_uid']=uid('ctl',v);profiles[cp['controller_uid']]=cp
   for s in p['states']:
    for seed in [0,1]:rows.append(case(s,name,cp,arm,seed,'TT',p['types'][name]))
 assert len(rows)==len(cap['eligible'])*32*2*8;register('main',p,rows,profiles)
if __name__=='__main__':globals()[sys.argv[1]]()
