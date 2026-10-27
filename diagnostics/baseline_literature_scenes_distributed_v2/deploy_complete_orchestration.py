"""Select optional fine dispatch; leave every running Slurm worker untouched."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_released import snapshot
import subprocess
import signal
import time


def descendants(root):
    rows=[s.strip().split(None,2) for s in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()]
    ids={root}
    while True:
        expanded=ids|{int(r[0]) for r in rows if int(r[1]) in ids}
        if expanded==ids:return [r for r in rows if int(r[0]) in ids]
        ids=expanded


def main():
    verify_receipt('OPTIONAL_FINE_SCHEDULING.receipt.json')
    verify_receipt('COMPLETE_ORCHESTRATION_DEPLOYMENT.receipt.json')
    old=read(HERE/'COORDINATOR_PID.json')['pid']
    while True:
        owners=descendants(old)
        if owners and all(r[2].endswith(('coordinator_pipelined.py','downstream_pipelined.py')) for r in owners):break
        print('Waiting for local atomic action to finish',owners,flush=True);time.sleep(10)
    jobs=snapshot()['jobs']
    old_watch=read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid']
    try:os.kill(old_watch,signal.SIGTERM)
    except ProcessLookupError:pass
    for row in owners:
        try:os.kill(int(row[0]),signal.SIGSTOP)
        except ProcessLookupError:pass
    # Check again after stopping so a newly spawned scientific action is never
    # interrupted between its data write and its immutable receipt.
    latest=descendants(old)
    assert {r[0] for r in latest}=={r[0] for r in owners},latest
    for row in owners:
        try:os.kill(int(row[0]),signal.SIGTERM);os.kill(int(row[0]),signal.SIGCONT)
        except ProcessLookupError:pass
    with (HERE/'coordinator.log').open('a') as out:
        proc=subprocess.Popen([sys.executable,'-B',str(HERE/'coordinator_complete.py')],
            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
    entry.update(utc=utc(),coordinator='coordinator_complete.py',pid=proc.pid,
        downstream='downstream_complete.py',optional='optional_delivery_fine.py',
        optional_worker='optional_worker_fine.py',final_reports='final_reports_complete.py',
        watchdog='scheduler_watchdog_complete.py')
    write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
    while read(HERE/'COORDINATOR_PID.json')['pid']!=proc.pid:
        assert proc.poll() is None;time.sleep(1)
    with (HERE/'SCHEDULER_WATCHDOG.log').open('a') as out:
        watcher=subprocess.Popen([sys.executable,'-B',str(HERE/'scheduler_watchdog_complete.py')],
            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'COMPLETE_ORCHESTRATION_ACTIVE.json',dict(utc=utc(),coordinator_pid=proc.pid,
        watchdog_pid=watcher.pid,previous_local_owners=owners,live_jobs_at_switch=jobs,
        running_worker_signals_sent=0,worker_cancellations=0,case_execution_unchanged=True))
    with (HERE/'ITERATION_LOG.md').open('a') as out:
        out.write('\n\n### '+utc()+' Extend fine dispatch to the remaining optional stages\n\n'
            'User again required continued execution. Current label/evaluation workers remain untouched. '
            'Future optional queues use the identical case partition algorithm already verified on the live learning queues; '
            'optional_worker_fine.py retains the byte-identical per-case body of optional_worker.py. '
            'The cooperative optional diagnosis worker retains the identical per-trace numerical body. '
            'Only claims, source receipts, process recycling and idle-slot scheduling change. '
            'No optional case is planned or submitted before MIGRATION_NUMBERS is sealed; optional_cases.py and all dynamics remain unchanged. '
            'The final report also discloses observed training-menu label support, including the zero-positive narrow-passage menu, '
            'without changing its 12 actions, fitting, choices, labels or success standards. '
            'Current root coordinator_complete.py / downstream_complete.py; future optional_delivery_fine.py; watchdog scheduler_watchdog_complete.py. '
            'Source and numerical-body checks are sealed in OPTIONAL_FINE_SCHEDULING.receipt.json. No signal was sent to any running Slurm worker.\n')
    print('Complete orchestration active',proc.pid,watcher.pid,flush=True)


if __name__=='__main__':main()
