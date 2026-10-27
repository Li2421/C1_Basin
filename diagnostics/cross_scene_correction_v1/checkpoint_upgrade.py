"""Enable the verified exact checkpoint wrapper for the sole long pilot request."""
from common import *
from schedule import jobs,cancel,merge,log,run
from queueing import build
from pilot_tail import present,PYTHON
from checkpoint_store import TARGET,enabled
import subprocess
import signal


def archive_queue(tag):
    archive=HERE/'pilot_tail_history'/f'{tag}_{time.time_ns()}';archive.mkdir(parents=True)
    for name in ('pilot_CACHE_PREFLIGHT.json','pilot_MISSING_CASES.json','pilot_REUSED_ROWS.json','pilot_PREFLIGHT.receipt.json'):
        p=HERE/name
        if p.exists():p.rename(archive/name)
    (HERE/'QUEUES/pilot').rename(archive/'queue')
    return archive


def main():
    cases=read(HERE/'pilot_CASES.json');target=next(c for c in cases if c['rollout_uid']==TARGET)
    assert enabled(target)
    if all(present(c)for c in cases):return
    assert {c['rollout_uid']for c in cases if not present(c)}=={TARGET}
    initial_archive=read(HERE/'PILOT_STALL_RECOVERY.json')['original_queue_archive']
    # Only replace our polling coordinator; source data and completed raw remain.
    for script in ('pilot_tail.py','pilot_recover.py'):
        for entry in (str(HERE/script),'diagnostics/cross_scene_correction_v1/'+script):
            p=subprocess.run(['pgrep','-f','^'+PYTHON+' -B '+entry+'$'],capture_output=True,text=True)
            for pid in p.stdout.split():os.kill(int(pid),signal.SIGTERM)
    old=[r for r in jobs()if r['name']in ('xsc_pilot','xsc_pilot_tail')]
    write(HERE/'PILOT_CHECKPOINT_UPGRADE.json',dict(utc=utc(),replaced_jobs=old,target_rollout_uid=TARGET,
        reason='Enable proved exact operational snapshots before another mandatory30min cancellation discards all progress.',
        validation_receipt_sha256=sha(HERE/'CHECKPOINT_REPLAY_CHECK.json'),next_slurm_limit_seconds=3540,
        original_full_plan_archive=initial_archive,no_completed_request_repeated=True,no_scientific_label_from_cancelled_attempt=True))
    log('pilot_checkpoint_wrapper_upgrade',jobs=old,seconds=3540,validation_sha256=sha(HERE/'CHECKPOINT_REPLAY_CHECK.json'))
    for row in old:cancel(row['id'])
    while any(r['name']in ('xsc_pilot','xsc_pilot_tail')for r in jobs()):time.sleep(1)
    merge('pilot_checkpoint_upgrade_partial')
    prior=archive_queue('checkpoint_upgrade')
    todo,reused=register('pilot',cases);assert len(todo)==1 and todo[0]['rollout_uid']==TARGET
    write(HERE/'PILOT_CHECKPOINT_UPGRADE_PREFLIGHT.json',dict(utc=utc(),reused=len(reused),missing=len(todo),
        previous_queue_archive=str(prior),receipt_sha256=sha(HERE/'pilot_PREFLIGHT.receipt.json')))
    build('pilot');run('pilot',initial_seconds=3540)
    assert all(present(c)for c in cases)
    final=archive_queue('completed_checkpoint_pilot')
    todo,reused=register('pilot',cases);assert not todo;build('pilot')
    write(HERE/'PILOT_TAIL_COMPLETED.json',dict(utc=utc(),requests=len(cases),all_reused_after_merge=True,
        original_queue_archive=initial_archive,latest_queue_archive=str(final),no_completed_request_reexecuted=True,
        exact_operational_checkpoint_resume=True,checkpoint_proof_sha256=sha(HERE/'CHECKPOINT_REPLAY_CHECK.json')))
    subprocess.run([PYTHON,'-B',str(HERE/'advance.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(HERE/'CHECKPOINT_UPGRADE_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise
