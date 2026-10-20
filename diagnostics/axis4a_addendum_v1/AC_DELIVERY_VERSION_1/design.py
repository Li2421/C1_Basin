from common import *
from registry import normalized_state,register
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy

def config(H,r):
 v=copy.deepcopy(read(HERE/'HISTORICAL_INVENTORY.json')['config']);v.update(max_steps=H,obstacle_radius=r);return v

def protocol():return dict(states=[],profiles={},scenarios={},eta=read(R10/'protocol.json')['eta'])
def state(p,s,H,r,validation=None):
 cfg=config(H,r);sid=uid('scenario',dict(round35_physical_config=cfg))
 p['scenarios'][sid]=dict(scenario_uid=sid,name='ring_exchange',code_config_fingerprint=uid('cfg',cfg),metadata_json=canonical(dict(config=cfg,physical_source_sha256=sha(ROOT/'ring_exchange/environment.py'))))
 s=copy.deepcopy(s);s['scenario_uid']=sid;s=normalized_state(s,validation);s.update(evaluation_config=cfg,scenario='ring_exchange')
 if s['state_uid'] not in {x['state_uid'] for x in p['states']}:p['states'].append(s)
 return s

def profile(p,name,H,r,arm,validation=None):
 base=copy.deepcopy(read(HERE/'ORIGINAL_PROFILES.json')[name]);v=base['payload'];cfg=config(H,r);a=[0.,0.,0.] if arm in ['nominal_only','safety_baseline','expert'] else p['eta'][10 if arm=='F10' else 13];b=p['eta'][10] if arm in ['Sabs','Srel','S500','Sprog'] else a;t={'Sabs':500,'S500':500,'Srel':round(5*H/7)}.get(arm,H)
 v.update(scenario_uid=uid('scenario',dict(round35_physical_config=cfg)),horizon=H,evaluation_config=cfg,jax_enable_x64=False,round35_execution=dict(arm=arm,eta_a=a,eta_b=b,switch_step=t,progress_threshold=None,validation=validation),execution_source_sha256={k:sha(HERE/k) for k in ['runtime.py','factor_runtime.py','prefix.py','progress.py']})
 key=uid('profile',v);p['profiles'][key]=dict(controller_uid=uid('ctl',v),payload=v)
 return key,a,b,t

def case(p,s,name,H,r,arm,seed,validation=None):
 key,a,b,t=profile(p,name,H,r,arm,validation);eid=eta_identity(a)[0];sk=canonical(dict(future_index=seed));cid=p['profiles'][key]['controller_uid'];return dict(H=H,r=r,profile_key=key,base_controller_name=name,state_uid=s['state_uid'],original_state_uid=s['original_state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),arm=arm,seed=seed,eta_a=a,eta_b=b,switch_step=t)

def gate():
 p=protocol();ss=read(HERE/'EVAL_STATES_ORIGINAL.json');refs=read(HERE/'SOURCE_CORRECTIONS.json');look={(q['case']['base_controller_name'],q['case']['state_uid'],q['case']['arm'],q['case']['seed']):q for q in refs};names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];spec=[(n,0,'F10',0) for n in names]+[(n,1,'F13',1) for n in ['base_u_macflow','base_u_v10_local_macflow']];rows=[]
 for n,i,arm,seed in spec:
  s=state(p,ss[i],700,.9,'round35_actor_gate');c=case(p,s,n,700,.9,arm,seed,'round35_actor_gate');q=look[n,ss[i]['state_uid'],arm,seed];c.update(reference_path=q['raw_path'],reference_sha256=q['raw_sha256']);rows.append(c)
 register('gate',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
if __name__=='__main__':globals()[sys.argv[1]]()
