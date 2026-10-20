"""Read-only actual evidence resolution; inferred local aliases are never sources."""
from common import *
from functools import lru_cache
ROOTS=[D/'crosstype4_no_repair_schedule_diagnosis_v1',D/'crosstype3_unseen_controller_schedule_v1',D/'crosstype1c_two_eta_schedule_v1',D/'crosstype1b_generator_critic_pipeline_v1',D/'crosstype2_mixed_zoo_selector_v1']
PHASES=('A','B_training','B_evaluation','new_evaluation','training','pilot','remaining','diagnostic','training_legacy','training_mixed','training_ppo','capability','gate_locked','gate_development','gate_ppo')
def permitted(path):
    p=Path(path).resolve();assert not any(part.startswith(('decentralized_','teacher_student_','multiscene_')) for part in p.parts)
    if p.is_relative_to(ROOT/'shared_rollout_db/journals'):assert not any('decentralized' in part or 'teacher_student' in part or 'multiscene' in part for part in p.parts)
    return p
@lru_cache(16)
def journal_index(path):
    p=permitted(path);result={}
    with p.open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'rollout_uid' in r and 'trace_path' in r:result[r['rollout_uid']]=r
    return result
def actual(row,verify=True):
    rid=row['rollout_uid'];record=None;path=None
    for root in [HERE,*ROOTS]:
        for phase in PHASES:
            p=root/phase/'raw'/f'{rid}.json'
            if p.exists():path=p;record=read(p);break
        if record is not None:break
    if record is None and row.get('original_source_file'):
        p=Path(row['original_source_file'])
        if not p.exists() and p.with_suffix('.merged').exists():p=p.with_suffix('.merged')
        if p.exists() and p.suffix in ('.jsonl','.merged'):
            record=journal_index(str(p)).get(rid);path=p
    if record is None or record.get('rollout_uid')!=rid or not record.get('trace_path'):return None
    trace=permitted(record['trace_path'])
    if not trace.exists():return None
    if verify:assert sha(trace)==record['trace_sha256']
    return dict(record=record,record_source_path=str(path),trace_path=str(trace),trace_sha256=record['trace_sha256'],source_rollout_uid=rid)
