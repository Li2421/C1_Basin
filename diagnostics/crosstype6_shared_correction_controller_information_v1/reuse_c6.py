"""Only actual physical trace sources certify new schedule aliases; never journal."""
from common import *
from given_actions import fixed_source_uid
from records import actual
from schedule_c6 import offline_trigger
from progress import offline_switch,progress

def infer(c,base,db):
    validation=read(HERE/'EARLY_TERMINATION_EQUIVALENCE.json')
    if not validation['passed'] or c['family'] not in validation['validated_types']:return None
    trigger=c.get('trigger','none');source_eta=[0.,0.,0.] if trigger!='none' else c['eta_a']
    rid=fixed_source_uid(base,c['base_controller_name'],c['state_uid'],source_eta,c['seed'])
    row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
    if row is None or row['conflict_quarantined']:return None
    e=actual(dict(row))
    if e is None:return None
    r=e['record'];rule=None
    with np.load(e['trace_path'],allow_pickle=False) as z:
        pos=z['positions'];g=z['goals']
        if trigger!='none':
            if offline_trigger(pos,g,trigger) is not None:return None
            if r.get('error'):
                from schedule_c6 import TRIGGERS
                w,eps=TRIGGERS[trigger];t=len(pos)-1;d0=np.linalg.norm(g-pos[0],axis=-1)
                if t>=w and progress(pos[t],g,d0)-progress(pos[t-w],g,d0)<eps:return None
            rule=dict(kind='no_trigger_before_termination',trigger=trigger,offline_trigger=None)
        else:
            if r['outcome'] not in ('success','collision','numerical_failure'):return None
            if offline_switch(pos,g,700,c.get('progress_threshold',.968)) is not None:return None
            if r.get('error'):
                if len(pos)-1>=600 or progress(pos[-1],g,np.linalg.norm(g-pos[0],axis=-1))>=c['progress_threshold']:return None
            rule=dict(kind='constant_early_terminated_before_switch',threshold=c.get('progress_threshold',.968),offline_switch=None)
    return dict(target_rollout_uid=c['rollout_uid'],source_rollout_uid=rid,source_record_file=e['record_source_path'],source_trace_path=e['trace_path'],source_trace_sha256=e['trace_sha256'],proof=rule,outcome={k:r[k] for k in ('success','collision','timeout','deadlock','numerical_failure','outcome','episode_length','physical_steps')},written_to_shared_DB=False)
