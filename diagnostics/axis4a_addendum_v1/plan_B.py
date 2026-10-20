"""Part B cannot even register requests until the explicit decision receipt exists."""
from design import *
OLD=D/'axis4a_geometry_horizon_v1'
def main():
 assert (HERE/'PSTAR_DECISION.receipt.json').exists(),'B userdecision required; no default'
 decision=read(HERE/'PSTAR_DECISION.receipt.json');assert decision['authorized_rollouts'] and decision['pstar'] is not None;threshold=decision['pstar'];assert (HERE/'SELECTED_PROGRESS_TEST.receipt.json').exists()
 p=protocol();rows=[];sources={}
 for c in read(OLD/'horizon_CASES.json'):
  if c['arm']=='F13' and c['r']==.9:
   f=OLD/'horizon/raw'/f"{c['rollout_uid']}.json";sources[c['H'],c['base_controller_name'],c['original_state_uid'],c['seed']]=dict(raw_path=str(f),raw_sha256=sha(f))
 for ref in read(HERE/'SOURCE_CORRECTIONS.json'):
  c=ref['case']
  if c['arm']=='F13':sources[700,c['base_controller_name'],c['state_uid'],c['seed']]={k:ref[k] for k in ['raw_path','raw_sha256']}
 for H in [500,700,1000]:
  for i,original in enumerate(read(HERE/'EVAL_STATES_ORIGINAL.json')):
   s=state(p,original,H,.9);s['study_index']=i
   for name in read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']:
    for seed in [0,1]:
     c=case(p,s,name,H,.9,'Sprog',seed);old=p['profiles'].pop(c['profile_key']);v=old['payload'];v['round35_execution']['progress_threshold']=threshold;v['progress_definition_receipt_sha256']=sha(HERE/'PSTAR_DECISION.receipt.json');key=uid('profile',v);cid=uid('ctl',v);p['profiles'][key]=dict(controller_uid=cid,payload=v);c.update(profile_key=key,controller_uid=cid,rollout_uid=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key'])),progress_threshold=threshold,state_index=i,prefix_source=sources[H,name,original['state_uid'],seed]);rows.append(c)
 register('B',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
if __name__=='__main__':main()
