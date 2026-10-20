"""Conservative, trace-certified local equivalence from actually executed sources."""
from common import *
from actions import fixed_source_uid
from records import actual
from progress import offline_switch,progress
THRESHOLD=.9681620884830543

def proof(c,source):
    r=source['record']
    if r['outcome'] not in ('success','collision','numerical_failure'):return None
    steps=int(r['episode_length']);physical=int(r['physical_steps'])
    if c['switch_mode']=='fixed':
        if steps>=int(c['switch_step']):return None
        return dict(rule='termination_strictly_before_fixed_switch',source_steps=steps,switch_step=c['switch_step'])
    if steps>=600 or physical>=600:return None
    with np.load(source['trace_path'],allow_pickle=False) as z:
        pos=z['positions'];goals=z['goals'];switch=offline_switch(pos,goals,700,THRESHOLD)
        # Guard the unrecorded failed action: it may have evaluated the rule at
        # the final stored frame before failing, even though no step was saved.
        last=progress(pos[-1],goals,np.linalg.norm(goals-pos[0],axis=-1))
    if switch is not None or last>=THRESHOLD:return None
    return dict(rule='terminated_before_any_progress_switch',source_steps=steps,source_physical_steps=physical,offline_switch=None,last_frame_progress=last,fallback=600,threshold=THRESHOLD)

def infer(c,base,db,validated_types):
    if c['family'] not in validated_types:return None
    if np.array_equal(c['eta_a'],c['eta_b']):return None
    rid=fixed_source_uid(base,c['base_controller_name'],c['state_uid'],c['eta_a'],c['seed'])
    if rid==c['rollout_uid']:return None
    row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
    if row is None or row['conflict_quarantined']:return None
    source=actual(dict(row))
    if source is None:return None
    evidence=proof(c,source)
    if evidence is None:return None
    r=source['record']
    return dict(target_rollout_uid=c['rollout_uid'],source_rollout_uid=rid,kind='提前结束等价复用',source_record_file=source['record_source_path'],source_trace_path=source['trace_path'],source_trace_sha256=source['trace_sha256'],proof=evidence,outcome={k:r[k] for k in ('success','collision','timeout','deadlock','numerical_failure','outcome','episode_length','physical_steps')},written_to_shared_DB=False)

def validated():
    p=HERE/'EARLY_TERMINATION_EQUIVALENCE.json'
    if not p.exists():return set()
    r=read(p)
    # Any discrepancy globally disables this optimization; ordinary execution
    # remains valid. Stateful types additionally require direct coverage.
    return set(r['validated_types']) if r['passed'] else set()
