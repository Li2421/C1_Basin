from design import *

def actors(stage,levels,arms,prefixes=False):
 p=protocol();rows=[];names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names'];ss=read(HERE/'EVAL_STATES_ORIGINAL.json');old={(q['case']['base_controller_name'],q['case']['state_uid'],q['case']['arm'],q['case']['seed']):q for q in read(HERE/'SOURCE_CORRECTIONS.json')}
 baseline={}
 if prefixes:
  for c in read(HERE/'baseline_CASES.json'):
   f=HERE/'baseline/raw'/f"{c['rollout_uid']}.json";assert f.exists();baseline[c['base_controller_name'],c['original_state_uid'],c['arm'],c['seed']]=dict(raw_path=str(f),raw_sha256=sha(f))
 for H,r in levels:
  for i,original in enumerate(ss):
   s=state(p,original,H,r);s['study_index']=i
   for name in names:
    for arm in arms(H):
     for seed in [0,1]:
      c=case(p,s,name,H,r,arm,seed);c['state_index']=i
      if prefixes:
       src=baseline[name,original['state_uid'],arm,seed] if arm in ['nominal_only','safety_baseline'] else old[name,original['state_uid'],'F10' if arm=='F10' else 'S500' if arm=='Sabs' else 'F13',seed]
       c['prefix_source']={k:src[k] for k in ['raw_path','raw_sha256']}
      rows.append(c)
 register(stage,p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
def baseline():actors('baseline',[(700,.9)],lambda H:['nominal_only','safety_baseline'])
def horizon():actors('horizon',[(500,.9),(1000,.9)],lambda H:['nominal_only','safety_baseline','F10','F13','Srel']+(['Sabs'] if H==1000 else []),True)

def expert_case(p,original,H,r,validation=None):
 s=state(p,original,H,r,validation);cfg=config(H,r);v=dict(scenario_uid=s['scenario_uid'],controller_class='CentralizedExpert',planner_sha256=sha(ROOT/'ring_exchange/expert.py'),parameters=dict(route_radius=1.85,route_speed=.46,arc_increment=.10,direct_goal_distance=1.10,require_success=False),horizon=H,dt=cfg['dt'],physical_config=cfg,conditioning='privileged deterministic offline joint two-circulation search then native replay',success_semantics='native collision_free_success and native success termination',rng=dict(deterministic=True),flow_checkpoint_sha256=None,orthoflow3_sha256=None,safety_projection_sha256=None,cbf=cfg,validation=validation,execution_source_sha256={k:sha(HERE/k) for k in ['runtime.py','factor_runtime.py','prefix.py','progress.py']});key=uid('profile',v);cid=uid('ctl',v);p['profiles'][key]=dict(controller_uid=cid,payload=v);eid=eta_identity([0.,0.,0.])[0];sk=canonical(dict(future_index=0));return dict(H=H,r=r,kind='expert',evaluation_config=cfg,profile_key=key,base_controller_name='expert',arm='expert',state_uid=s['state_uid'],original_state_uid=s['original_state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),seed=0,eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_step=H)
def expert_gate():
 p=protocol();old=D/'decentralized_pipeline_v1';source=read(old/'expert_protocol.json');rows=[];states={s['state_uid']:s for s in source['states']}
 for ref in sorted(source['cases'],key=lambda x:x['state_uid'])[:16]:
  original=states[ref['state_uid']];c=expert_case(p,original,700,.9,'round35_expert_gate');f=old/'expert/raw'/f"{ref['rollout_uid']}.json";assert f.exists();c.update(reference_path=str(f),reference_sha256=sha(f));rows.append(c)
 register('expert_gate',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
def experts(levels,stage):
 p=protocol();rows=[expert_case(p,s,H,r) for H,r in levels for s in read(HERE/'EVAL_STATES_ORIGINAL.json')];register(stage,p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
if __name__=='__main__':globals()[sys.argv[1]]()
