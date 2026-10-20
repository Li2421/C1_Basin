"""Given actions retain exact cache identity; requery is a separate strategy."""
from common import *
from given_actions import action as given_action,case as given_case,fixed_source_uid,TIMES
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy

def action(a,b,mode='fixed',step=500,name='two_eta',late_source='given'):
    assert late_source in ('given','requery')
    if late_source=='given':return dict(given_action(a,b,mode,step,name),late_source='given')
    assert mode in ('fixed','progress')
    # b is an encoding placeholder; equality does not make requery diagonal.
    return dict(name=name,eta_a=np.asarray(a,np.float64).tolist(),eta_b=np.asarray(b,np.float64).tolist(),switch_mode=mode,switch_step=int(step) if mode=='fixed' else 700,late_source='requery')

def case(base,name,s,a,seed,state_index=0,candidate_index=-1):
    if a.get('late_source','given')=='given':
        c,cp=given_case(base,name,s,a,seed,state_index,candidate_index);c['late_source']='given';return c,cp
    a=action(a['eta_a'],a['eta_b'],a['switch_mode'],a['switch_step'],a.get('name','requery'),'requery');v=copy.deepcopy(base['profiles'][name]['payload'])
    from requery_generator import CHECKPOINT,NORMALIZATION
    v['late_eta_requery']=dict(switch_mode=a['switch_mode'],switch_step=a['switch_step'],early_eta='request_eta_uid',progress_threshold=.9681620884830543,fallback=600,generator_checkpoint_sha256=sha(CHECKPOINT),normalization_sha256=sha(NORMALIZATION),conditioning_key='initial_normalized_state_uid',conditioning_implementation_sha256=sha(HERE/'legacy_conditioning.py'),generator_wrapper_sha256=sha(HERE/'requery_generator.py'),execution_implementation_sha256=sha(HERE/'requery_execution.py'),progress_sha256=sha(HERE/'progress.py'))
    cid=uid('ctl',v);eid=eta_identity(a['eta_a'])[0];sk=canonical(dict(future_index=int(seed)));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk))
    c=dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,family=base['types'][name],arm=a['name'],candidate_index=int(candidate_index),seed=int(seed),eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode=a['switch_mode'],switch_step=a['switch_step'],late_source='requery',state_index=int(state_index),placement='FF' if base['types'][name]=='Flow' else 'TT',rhr_rule=None)
    return c,dict(controller_uid=cid,payload=v)
