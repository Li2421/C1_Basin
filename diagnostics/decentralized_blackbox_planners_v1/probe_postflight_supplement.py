from common import *
from shared_rollout_db.src.rollout_db import connect
G=guard()
phase='probe';p=read(HERE/'probe_protocol.json');pf=planner.preflight(HERE/'probe_PLANNED_ROLLOUTS.json');write(HERE/'probe_CACHE_POSTFLIGHT.json',pf)
with connect(True) as db:
 rows=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid=?',(p['experiment_uid'],))];n=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(p['experiment_uid'],)).fetchone()[0]
assert n==0
write(HERE/'PROBE_DB_POSTFLIGHT.json',dict(utc=utc(),rollout_rows=n,source_rows=rows,expected_records=576,expected_classification='SUMMARY_ONLY',explanation='Probes have no success/termination labels and must not create task rollout rows. Task-cache preflight remains missing by design; immutable probe raw/trace and merged SUMMARY_ONLY sources are authoritative.',guard=guard()))
seal('PROBE_DB_POSTFLIGHT.receipt.json',[HERE/'probe_CACHE_POSTFLIGHT.json',HERE/'PROBE_DB_POSTFLIGHT.json'],task_rollout_rows=0,guard=guard())
