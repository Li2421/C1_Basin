"""Serial post-batch lifecycle; no simulation and no unknown rerun."""
from core import *
import subprocess
from shared_rollout_db.src.cache_writer import append_journal

def step(script,*args):
    write(HERE/'FINALIZE_STAGE.json',dict(utc=utc(),stage=script,args=args,status='RUNNING'))
    result=subprocess.run([PY,'-B',str(HERE/script),*args],cwd=HERE,text=True,capture_output=True)
    with (HERE/'FINALIZE.log').open('a') as f:f.write(f'\n{utc()} {script} {args}\n{result.stdout}\n{result.stderr}\n')
    result.check_returncode()
    write(HERE/'FINALIZE_STAGE.json',dict(utc=utc(),stage=script,args=args,status='PASS'))

def main():
    prereg();gate();rows=read(HERE/'main_CASES.json');missing=[c['rollout_uid'] for c in rows if not(HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()]
    if missing:
        write(HERE/'INCOMPLETE.json',dict(utc=utc(),missing=missing,reason='Batch incomplete: preserve all completed and unknown evidence; main report cannot be marked complete'));raise RuntimeError(f'Missing {len(missing)} raw records; no automatic rerun')
    # Recover completed records that were interrupted between raw and buffered journal flush.
    # Duplicate journal lines are identical and handled idempotently by the single merger.
    for start in range(0,len(rows),128):
        records=[read(HERE/'main/raw'/f"{c['rollout_uid']}.json") for c in rows[start:start+128]];append_journal(clean(records),EXP,'complete_raw_recovery')
    step('asset_audit.py');step('merge.py');step('postflight.py','main');step('analysis.py');step('classifier.py');step('report.py')
    write(HERE/'COMPLETE.json',dict(utc=utc(),status='COMPLETE',main=10240,consistency_gate=32,DEVELOPMENT=True,results_sha256=sha(HERE/'results.json'),report_sha256=sha(HERE/'REPORT.md'),handoff_sha256=sha(HERE/'HANDOFF.md')))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(HERE/'FINALIZE_ERROR.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),complete=False));raise
