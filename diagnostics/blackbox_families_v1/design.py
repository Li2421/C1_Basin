from common import *
from registry import register,normalized_state
from shared_rollout_db.src.rollout_db import connect,uid,canonical,eta_identity
import copy
FAMILIES=['centralized_sampling','cross_entropy','prioritized_astar','deterministic','mixture','diffusion']
def signature(v):
 try:
  arrays=[]
  for keys in [('positions','initial_positions'),('velocities','initial_velocities'),('goals',)]:
   a=np.asarray(next(v[k] for k in keys if k in v),np.float64).copy();assert a.shape==(4,2);a[a==0]=0;arrays.append(a.tobytes().hex())
  return uid('physical',dict(arrays=arrays,timestep=float(v.get('timestep',v.get('step',0)))))
 except (TypeError,KeyError,StopIteration,AssertionError):return None

def states():
 prereg();cfg=read(HERE/'DESIGN_CONFIG.json')
 from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
 from ring_exchange.environment import RingExchangeEnv,LocalFrameConfig
 native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');pilot=generate(cfg['pilot_split'],16,cfg['pilot_seed_start'],native);evaluation=generate(cfg['eval_split'],32,cfg['eval_seed_start'],native)
 with connect(True) as db:old=[dict(r) for r in db.execute('SELECT state_uid,content_hash,physical_state_json FROM state')]
 sigs={signature(json.loads(r['physical_state_json']) if r['physical_state_json'] else None) for r in old};oldids={r['state_uid'] for r in old};contents={r['content_hash'] for r in old};hits=[]
 unresolved=sum(signature(json.loads(r['physical_state_json']) if r['physical_state_json'] else None) is None for r in old)
 for s in pilot+evaluation:
  RingExchangeEnv(LocalFrameConfig(**native['environment']['config']['config'])).reset(**{k:np.asarray(v) for k,v in s['physical'].items() if k!='timestep'});sig=signature(s['physical'])
  if s['state_uid'] in oldids or s['content_hash'] in contents or sig in sigs:hits.append(s['state_uid'])
  sigs.add(sig)
 write(HERE/'STATE_DEDUP.json',dict(utc=utc(),passed=not hits,DB_rows=len(old),unrecoverable_physical=unresolved,duplicates=hits,all_known_DB_states_and_within_round=True));assert not hits
 write(HERE/'PILOT_STATES.json',[normalized_state(s) for s in pilot]);write(HERE/'EVAL_STATES.json',[normalized_state(s) for s in evaluation]);write(HERE/'TUNING_STATES.json',[normalized_state(s) for s in read(D/'blackbox_addendum_v1/TUNING_STATES.json')])

def profiles(families):
 cfg=read(HERE/'DESIGN_CONFIG.json');old=read(D/'other_controller_classes_v1/base_protocol.json');q=read(D/'controller_universal_eta_schedule_v1/protocol.json');p=dict(states=[],profiles={},eta=q['eta'],native_manifest=old['native_manifest'],types={},controller_order=[])
 for family in families:
  for i in range(6):
   name=family+str(i);v=copy.deepcopy(old['base_profiles']['M'+str(i) if family=='centralized_sampling' else 'M0']['payload'])
   if family=='centralized_sampling':cp=copy.deepcopy(old['base_profiles']['M'+str(i)])
   elif family in ['cross_entropy','prioritized_astar']:
    v.update(controller_class='CEM' if family=='cross_entropy' else 'ASTAR',nominal_parameters=cfg['CEM' if family=='cross_entropy' else 'ASTAR'][i],nominal_implementation_sha256=sha(HERE/('controllers.py' if family=='cross_entropy' else 'spacetime_astar.py')),nominal_observation_chart='physical world positions/velocities and own goal',placement='TT',chain='TT',jax_enable_x64=False);cp=dict(controller_uid=uid('ctl',v),payload=v)
   else:
    g=next(x for x in read(HERE/'TRAINING_CONFIGS.json') if x['name']==name);folder=HERE/'policies'/name
    assert (folder/'COMPLETE.receipt.json').exists(),name
    for path,digest in read(folder/'COMPLETE.receipt.json')['files'].items():assert sha(path)==digest
    v=copy.deepcopy(q['base_profiles'][g['recipe']]['payload']);v.update(controller_class=family,flow_checkpoint_path=str(folder/'best.msgpack'),flow_checkpoint_sha256=sha(folder/'best.msgpack'),normalization_sha256=sha(folder/'NORMALIZATION.json'),policy_config_sha256=sha(folder/'CONFIG.json'),source_Flow_recipe=g['recipe'],conditioning='nominal_TT_project_correct_reproject',execution_semantics='nonflow_TT_native_v1',placement='TT',chain='TT',nominal_implementation_sha256=sha(HERE/'learned.py'),jax_enable_x64=False);cp=dict(controller_uid=uid('ctl',v),payload=v)
   p['profiles'][name]=cp;p['types'][name]=family;p['controller_order'].append(name)
 return p

def case(s,name,cp,p,arm,seed=0,rule=None):
 v=copy.deepcopy(cp['payload']);a=[0.,0.,0.] if arm in ['nominal_only','safety_baseline','RHR','RHR_fair'] else p['eta'][10 if arm=='F10' else 13];b=p['eta'][10] if arm in ['S500','Sprog'] else a;t=500 if arm=='S500' else 700
 if arm=='nominal_only':v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='round38_nominal_only_v1')
 elif arm=='S500':v['eta_schedule']=dict(eta_a=a,eta_b=b,switch_step=t,action_index_rule='a if index<t else b')
 elif arm=='Sprog':v['progress_schedule']=dict(pstar=.9681620884830543,fallback=600,implementation_sha256=sha(HERE/'progress.py'))
 if rule:v['continuing_right_hand']=dict(rule=rule,implementation_sha256=sha(HERE/'right_hand.py'),trigger_sha256=sha(HERE/'trigger.py'))
 cid=uid('ctl',v);profile=dict(controller_uid=cid,payload=v);eid=eta_identity(a)[0];sk=canonical({'future_index':seed});c=dict(rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,family=p['types'].get(name,'training'),arm=arm,seed=seed,eta_a=a,eta_b=b,switch_step=t,placement='TT',state_index=s.get('study_index',0),rhr_rule=rule)
 return c,profile

def pilot(families):
 assert read(HERE/'REPRODUCTION_GATE.json')['passed'];assert read(HERE/'UNIT_TESTS.json')['passed'];p=profiles(families);p['states']=read(HERE/'PILOT_STATES.json');rows=[];cps={};rule=read(HERE/'FIXED_RIGHT_HAND_RECOMPUTATION.json')['selected']['rule'];stage='pilot_'+families[0]
 for name in p['controller_order']:
  for s in p['states']:
   for arm in ['F10','S500','RHR']:
    c,cp=case(s,name,p['profiles'][name],p,arm,rule=rule if arm=='RHR' else None);rows.append(c);cps[cp['controller_uid']]=cp
 register(stage,p,rows,cps);seal(stage+'_CONTROLLERS_FROZEN.receipt.json',[HERE/f'{stage}_protocol.json',HERE/'controllers.py',HERE/'learned.py',HERE/'spacetime_astar.py',HERE/'episode.py',HERE/'UNIT_TESTS.json'],before_pilot_labels=True)
if __name__=='__main__':
 if sys.argv[1]=='states':states()
 else:pilot(sys.argv[1:])
