"""Unchanged scientific runtime, optional source-verified operational snapshots."""
from common import *
from execution import execute
import numpy as np

def one(stage,c,pause):
    raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json";marker=HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json"
    if raw.exists():
        r=read(raw);assert sha(r['trace_path'])==r['trace_sha256']
        if not marker.exists():
            j=cache_writer.append_journal([r],EXP,worker_id=f'{stage}_repair_{os.getpid()}');write(marker,dict(journal=str(j),recovered_existing_raw=True))
        return r
    kind=c['controller_payload']['nominal']['type_code']
    if c.get('cross_scene_new')and kind in ('prioritized_astar','conflict_based_search'):
        proof=read(HERE/'CHECKPOINT_COMPATIBILITY.json');assert proof['passed']
        from runtime_checkpointed import run
        from checkpoint_store import CheckpointStore
        r,a=run(c,CheckpointStore(c,pause=pause))
        r['operational_resume'].update(validation_receipt=str(HERE/'CHECKPOINT_COMPATIBILITY.json'),validation_receipt_sha256=sha(HERE/'CHECKPOINT_COMPATIBILITY.json'))
    else:r,a=execute(c)
    trace=HERE/stage/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True);tmp=trace.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**a)
    with tmp.open('rb')as f:os.fsync(f.fileno())
    tmp.replace(trace);r=record(c,dict(r,trace_path=str(trace),trace_sha256=sha(trace)));write(raw,r)
    j=cache_writer.append_journal([r],EXP,worker_id=f'{stage}_{os.getpid()}');write(marker,dict(journal=str(j),raw_sha256=sha(raw)))
    print(stage,c['pipeline_scene'],c['controller_payload']['nominal']['controller'],c['state']['index'],
        c['eta_uid'],r['outcome'],round(r['elapsed_seconds'],3),flush=True)
    return r
