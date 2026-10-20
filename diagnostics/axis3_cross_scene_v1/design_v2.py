from common import *
from registry import normalized_state,register
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy

def case(s,key,cp,arm,seed):
 H=int(cp['payload']['horizon']);eta=read(HERE/'HISTORICAL_INVENTORY.json')['eta'];a=[0.,0.,0.] if arm in ['nominal_only','safety_baseline'] else eta[arm[1:3] if arm.startswith('S') else arm[1:]]['values'];b=eta['10']['values'] if arm.startswith('S') else a;t=round((3 if arm.endswith('t1') else 5)*H/7) if arm.startswith('S') else H;eid=eta_identity(a)[0];sk=canonical({'future_index':seed});return dict(rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cp['controller_uid'],seed=sk)),state_uid=s['state_uid'],original_state_uid=s['original_state_uid'],eta_uid=eid,controller_uid=cp['controller_uid'],seed_key=sk,base_controller_name=key,scene=key.split('::',1)[0],arm=arm,seed=seed,eta_a=a,eta_b=b,switch_step=t,state_index=s.get('study_index',0))

def gate():
 prereg();inv=read(HERE/'HISTORICAL_INVENTORY.json');old=read(R13/'protocol.json');p=dict(states=[],profiles={},environments=inv['environments'],eta=old['eta']);rows=[]
 for sc in SCENES:
  h=R19 if sc=='four_way_intersection' else R13;refs=read(h/('FIXED_CASES.json' if sc=='four_way_intersection' else 'main_CASES.json'));source=read(h/('FIXED_PROTOCOL.json' if sc=='four_way_intersection' else 'protocol.json'));names=inv['scenes'][sc]['controllers'];ss=sorted({c['state_uid'] for c in refs if c['base_controller_name'] in names and c['scene']==sc})[:2];states={s['state_uid']:s for s in source['states']};lookup={(c['base_controller_name'],c['state_uid'],c['arm'],c['seed']):c for c in refs};
  for n in names:
   cp=copy.deepcopy(inv['profiles'][sc][n]);cp['payload'].update(jax_enable_x64=False,precision_protocol='shared_axes_FP32',native_source_precision=inv['scenes'][sc]['historical_x64']);cp['controller_uid']=uid('ctl',cp['payload']);p['profiles'][sc+'::'+n]=cp
  for sid in ss:
   s=normalized_state(states[sid],validation='round34_gate_'+sc);p['states'].append(s)
   for n in names[:2]:
    for arm in ['F10','F15']:
     for seed in [0,1]:
      original=lookup[n,sid,arm,seed];key=sc+'::'+n;c=case(s,key,p['profiles'][key],arm,seed);c['reference_path']=str(h/('stage2/raw' if sc=='four_way_intersection' else 'main/raw')/f"{original['rollout_uid']}.json");assert Path(c['reference_path']).exists(),c['reference_path'];c['reference_rollout_uid']=original['rollout_uid'];c['original_seed_key']=original['seed_key'];rows.append(c)
 assert len(rows)==64;register('gate_v2',p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
if __name__=='__main__':globals()[sys.argv[1]]()
