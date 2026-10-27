"""Drain existing workers after their current case; attach the finer scheduler."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_released import snapshot
from coordinator_fast import jobs_by_stage
import subprocess
import signal
import time


def main():
    for n in ('PIPELINED_SCHEDULING.receipt.json','FINE_QUEUE_EQUIVALENCE.receipt.json',
              'WATCHDOG_OWNER_GUARD.receipt.json','PIPELINED_MIGRATION_PLAN.receipt.json'):
        verify_receipt(n)
    old=read(HERE/'COORDINATOR_PID.json')['pid']
    rows=[s.strip().split(None,2) for s in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()]
    targets={old}
    while True:
        expanded=targets|{int(r[0]) for r in rows if int(r[1]) in targets}
        if expanded==targets:break
        targets=expanded
    owners=[r for r in rows if int(r[0]) in targets]
    assert owners and all(r[2].endswith(('coordinator_fast.py','downstream_fast.py')) for r in owners),owners
    for pid in targets:
        try:os.kill(pid,signal.SIGSTOP)
        except ProcessLookupError:pass
    for pid in targets:
        try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    state=snapshot();mapping=jobs_by_stage(state)
    owned={j:s for j,s in mapping.items() if s.startswith('learning_labels_')}
    assert len(owned)==len(mapping),(owned,mapping)
    preserved={str(q):sha(q) for scene in NAMES for q in (HERE/('learning_labels_'+scene)/'raw').glob('*.json')}
    write(HERE/'PIPELINED_MIGRATION_START.json',dict(utc=utc(),old_owners=owners,draining_jobs=owned,
        preserved_raw=preserved,hard_cancellations=0))
    for job in owned:
        result=subprocess.run(['scancel','--batch','--signal=USR1',job],capture_output=True,text=True)
        print('Drain after current case',job,result.returncode,flush=True)
    while True:
        live={j['job_id'] for j in snapshot()['jobs']};pending=sorted(set(owned)&live)
        write(HERE/'PIPELINED_MIGRATION_PROGRESS.json',dict(utc=utc(),draining_jobs=pending,hard_cancellations=0))
        print('Remaining draining workers',pending,flush=True)
        if not pending:break
        time.sleep(10)
    assert all(sha(path)==expected for path,expected in preserved.items())
    with (HERE/'coordinator.log').open('a') as out:
        proc=subprocess.Popen([sys.executable,'-B',str(HERE/'coordinator_pipelined.py')],
            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'ACTIVE_ENTRYPOINT.json',dict(utc=utc(),coordinator='coordinator_pipelined.py',pid=proc.pid,
        worker='worker_fine.py',submit='fine_dispatch.py',analysis='analysis_jobs_cached.py',
        downstream='downstream_pipelined.py',optional='optional_delivery_pipelined.py',
        final_reports='final_reports_pipelined.py',resources='resources_released.py',
        watchdog='scheduler_watchdog_guarded.py',total_cpu=18,total_gpu_shards=2))
    while read(HERE/'COORDINATOR_PID.json')['pid']!=proc.pid:
        assert proc.poll() is None;time.sleep(1)
    with (HERE/'SCHEDULER_WATCHDOG.log').open('a') as out:
        watcher=subprocess.Popen([sys.executable,'-B',str(HERE/'scheduler_watchdog_guarded.py')],
            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'PIPELINED_MIGRATION_COMPLETE.json',dict(utc=utc(),coordinator_pid=proc.pid,
        watchdog_pid=watcher.pid,preserved_raw_count=len(preserved),all_preserved_raw_hashes_unchanged=True,
        old_jobs_drained_after_current_case=True,hard_cancellations=0,unfinished_not_unknown=True))
    with (HERE/'ITERATION_LOG.md').open('a') as out:
        out.write('\n\n### '+utc()+' Persistent scheduling repair after user required continued monitoring\n\n'
            'The previous 32-request claims let a worker retain many long CBS episodes while other CPU shares were idle. '
            'The old global learning barrier also prevented a finished scene from training its critic. '
            'Finer queues contain the identical sealed missing cases: one label episode per claim, and paired seeds together for evaluation. '
            'All 23552 missing label payloads compare exactly to their original queue; the per-case worker execution and save body is byte-identical. '
            'Current workers received USR1 and finished their current episode; no live episode was cancelled. '
            'All pre-migration raw hashes remain unchanged. The new scene scheduler checks every 10 seconds, submits critic training when that scene alone is ready, '
            'uses the unchanged frozen model and choice functions before its first evaluation submission, and backfills mandatory trace diagnoses during slow tails. '
            'A separate watchdog checks every 30 seconds, distinguishes a live downstream from a dead parent, and restarts only clean coordinator exits; scientific action failures are surfaced. '
            '18 CPU shares including local reserve and 2 GPU shards remain the user-wide caps. '
            'See PIPELINED_SCHEDULING.receipt.json, FINE_QUEUE_EQUIVALENCE.receipt.json, WATCHDOG_OWNER_GUARD.receipt.json and PIPELINED_MIGRATION_COMPLETE.json.\n')
    print('Pipelined coordinator and watchdog started',proc.pid,watcher.pid,flush=True)


if __name__=='__main__':main()
