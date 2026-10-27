"""Recover an actually stalled pilot queue without repeating completed evidence."""
from common import *
from schedule import jobs, cancel, merge, log, run
from queueing import build
from pilot_tail import present, PYTHON
import subprocess
import signal


def main():
    progress=read(HERE/'PROGRESS.json')
    assert progress['stage']=='pilot'
    assert progress['last_new_result_age_seconds']>=1800,progress
    cases=read(HERE/'pilot_CASES.json')
    if all(present(c)for c in cases):
        return
    # Stop the coordinator first, then wait until all writers are gone before
    # invoking the sole official merger. Completed raw/journals remain intact.
    for entry in (str(HERE/'pilot_tail.py'),'diagnostics/cross_scene_correction_v1/pilot_tail.py'):
        pattern='^'+PYTHON+' -B '+entry+'$'
        found=subprocess.run(['pgrep','-f',pattern],capture_output=True,text=True)
        for pid in found.stdout.split():
            os.kill(int(pid),signal.SIGTERM)
    stopped=[r for r in jobs()if r['name']in ('xsc_pilot','xsc_pilot_tail')]
    log('pilot_no_new_result_30min_cancel_resubmit_shorter',
        previous_progress=progress,stopped_jobs=stopped,seconds=5400,
        incomplete_is_not_an_outcome=True)
    for row in stopped:
        cancel(row['id'])
    while any(r['name']in ('xsc_pilot','xsc_pilot_tail')for r in jobs()):
        time.sleep(1)
    merge('pilot_partial_recovery')
    archive=HERE/'pilot_tail_history'/str(time.time_ns())
    archive.mkdir(parents=True)
    if(HERE/'PILOT_TAIL_ERROR.json').exists():
        (HERE/'PILOT_TAIL_ERROR.json').rename(archive/'PILOT_TAIL_ERROR.json')
    for name in ('pilot_CACHE_PREFLIGHT.json','pilot_MISSING_CASES.json',
                 'pilot_REUSED_ROWS.json','pilot_PREFLIGHT.receipt.json'):
        p=HERE/name
        if p.exists():
            p.rename(archive/name)
    (HERE/'QUEUES/pilot').rename(archive/'original_queue')
    todo,reused=register('pilot',cases)
    write(HERE/'PILOT_STALL_RECOVERY.json',dict(utc=utc(),
        original_queue_archive=str(archive),previous_progress=progress,
        stopped_jobs=stopped,missing_requests=[c['rollout_uid']for c in todo],
        reused_completed_requests=len(reused),retry_walltime_seconds=5400,
        cancelled_incomplete_attempt_has_no_scientific_label=True))
    build('pilot')
    run('pilot',initial_seconds=5400)
    assert all(present(c)for c in cases)
    write(HERE/'PILOT_TAIL_COMPLETED.json',dict(utc=utc(),requests=len(cases),
        original_queue_archive=str(archive),no_completed_request_reexecuted=True,
        incomplete_attempt_restarted_per_30min_rule=True,
        recovery_receipt=str(HERE/'PILOT_STALL_RECOVERY.json')))
    subprocess.run([PYTHON,'-B',str(HERE/'advance.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        write(HERE/'PILOT_RECOVERY_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise
