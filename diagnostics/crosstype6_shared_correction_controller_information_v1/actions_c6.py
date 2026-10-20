"""Physical identity for new schedules; legacy constant identities retained."""
from common import *
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
from given_actions import case as given_case
from schedule_c6 import THRESHOLDS,TRIGGERS
import copy

def action(a,b,threshold,trigger='none',late_source='given',name='learned_schedule'):
    a=np.asarray(a,np.float64).tolist();b=np.asarray(b,np.float64).tolist()
    assert trigger in TRIGGERS and late_source in ('given','requery')
    diagonal=late_source=='given' and np.array_equal(a,b)
    if diagonal:threshold=0.
    if diagonal and not np.any(a):trigger='none'
    return dict(name=name,eta_a=a,eta_b=b,switch_mode='fixed' if diagonal and trigger=='none' else 'progress',switch_step=500 if diagonal and trigger=='none' else 700,progress_threshold=float(threshold),trigger=trigger,late_source=late_source)

def case(base,name,s,a,seed,state_index=0,candidate_index=-1,force_new=False,probe=False):
    if a['switch_mode']=='fixed' and a.get('trigger','none')=='none' and a.get('late_source','given')=='given' and not (force_new or probe):
        c,cp=given_case(base,name,s,a,seed,state_index,candidate_index);c.update(progress_threshold=0.,trigger='none',late_source='given');return c,cp
    v=copy.deepcopy(base['profiles'][name]['payload'])
    v['crosstype6_schedule']=dict(schema='before_action_progress_stall_v1',early_eta='request_eta_uid',late_eta=a['eta_b'] if a.get('late_source','given')=='given' else 'isolated_frozen_generator_at_actual_switch',late_source=a.get('late_source','given'),progress_threshold=a['progress_threshold'],trigger=a['trigger'],fallback=600,schedule_sha256=sha(HERE/'schedule_c6.py'),execution_sha256=sha(HERE/'execution_c6.py'),controller_process_XLA_FLAGS='unset')
    if a.get('late_source')=='requery':
        v['crosstype6_schedule']['frozen_reference']={n:sha(HERE/n) for n in ('requery_client.py','requery_helper.py','requery_generator_v2.py','snapshot_conditioning.py','frozen_networks.py')}
        v['crosstype6_schedule']['frozen_checkpoint_sha256']=sha(D/'orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack')
    if probe:v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=700,terminate_on_success=False,terminate_on_collision=False)
    cid=uid('ctl',v);eid=eta_identity(a['eta_a'])[0];sk=canonical(dict(future_index=int(seed)));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk))
    c=dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,family=base['types'][name],arm=a['name'],candidate_index=int(candidate_index),seed=int(seed),eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode=a['switch_mode'],switch_step=a['switch_step'],progress_threshold=a['progress_threshold'],trigger=a['trigger'],late_source=a.get('late_source','given'),state_index=int(state_index),placement='FF' if base['types'][name]=='Flow' else 'TT',rhr_rule=None,c6_schedule=True)
    return c,dict(controller_uid=cid,payload=v)

def execute(c,p,cache,phase,probe=False):
    if c.get('c6_schedule'):
        from execution_c6 import execute as run
    else:
        from execution_v3 import execute as run
    return run(c,p,cache,phase,probe=probe)
