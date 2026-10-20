"""Validated actual-source equivalence, including equal requery placeholders."""
from common import *
from actions_v3 import fixed_source_uid
from early_reuse import proof
from records import actual

def validated():
    p=HERE/'EARLY_TERMINATION_EQUIVALENCE_v3.json'
    if not p.exists():return set()
    r=read(p);return set(r['validated_types']) if r['passed'] else set()
def infer(c,base,db):
    if c['family'] not in validated():return None
    if c.get('late_source','given')=='given' and np.array_equal(c['eta_a'],c['eta_b']):return None
    rid=fixed_source_uid(base,c['base_controller_name'],c['state_uid'],c['eta_a'],c['seed']);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
    if row is None or row['conflict_quarantined'] or rid==c['rollout_uid']:return None
    source=actual(dict(row))
    if source is None:return None
    pr=proof(c,source)
    if pr is None:return None
    r=source['record'];return dict(target_rollout_uid=c['rollout_uid'],source_rollout_uid=rid,kind='提前结束等价复用',source_record_file=source['record_source_path'],source_trace_path=source['trace_path'],source_trace_sha256=source['trace_sha256'],proof=pr,outcome={k:r[k] for k in ('success','collision','timeout','deadlock','numerical_failure','outcome','episode_length','physical_steps')},written_to_shared_DB=False)
