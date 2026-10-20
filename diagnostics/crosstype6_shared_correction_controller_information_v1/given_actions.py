"""Canonical physical actions; critic encodings never change nominal cache IDs."""
from common import *
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy
TIMES=(200,300,400,500,600)
def action(a,b,mode='fixed',step=500,name='two_eta'):
    a=np.asarray(a,np.float64).tolist();b=np.asarray(b,np.float64).tolist();assert mode in ('fixed','progress')
    if np.array_equal(a,b):mode='fixed';step=500
    return dict(name=name,eta_a=a,eta_b=b,switch_mode=mode,switch_step=int(step) if mode=='fixed' else 700)
def case(base,name,s,a,seed,state_index=0,candidate_index=-1):
    a=action(a['eta_a'],a['eta_b'],a['switch_mode'],a['switch_step'],a.get('name','two_eta'));v=copy.deepcopy(base['profiles'][name]['payload']);diagonal=np.array_equal(a['eta_a'],a['eta_b'])
    if a['switch_mode']=='progress':v['generator_progress_schedule']=dict(early_eta='request_eta_uid',late_eta=a['eta_b'],pstar=.9681620884830543,fallback=600,progress_implementation_sha256=sha(HERE/'progress.py'),wrapper_sha256=sha(HERE/'generated_execution.py'))
    elif not diagonal:v['eta_schedule']=dict(eta_a=a['eta_a'],eta_b=a['eta_b'],switch_step=a['switch_step'],action_index_rule='a if index<t else b')
    cid=uid('ctl',v);eid=eta_identity(a['eta_a'])[0];sk=canonical(dict(future_index=int(seed)));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk))
    c=dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,family=base['types'][name],arm=a['name'],candidate_index=int(candidate_index),seed=int(seed),eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode=a['switch_mode'],switch_step=a['switch_step'],state_index=int(state_index),placement='FF' if base['types'][name]=='Flow' else 'TT',rhr_rule=None)
    return c,dict(controller_uid=cid,payload=v)
def fixed_source_uid(base,name,state_uid,eta,seed):
    # The native Flow profile retains a historical alias. Physical action cases
    # in crosstype1c/3 use the canonical payload digest, including for eta=0.
    cid=uid('ctl',base['profiles'][name]['payload'])
    return uid('roll',dict(state=state_uid,eta=eta_identity(eta)[0],controller=cid,seed=canonical(dict(future_index=int(seed)))))
