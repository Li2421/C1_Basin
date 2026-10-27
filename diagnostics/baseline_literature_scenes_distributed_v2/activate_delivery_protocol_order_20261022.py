"""Replace only idle local managers; science workers and merger are untouched."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import fcntl
import signal
import subprocess
import time


def command(pid):
    p = Path('/proc')/str(pid)/'cmdline'
    try:
        return p.read_bytes().replace(b'\0', b' ').decode().strip()
    except FileNotFoundError:
        return ''


def start(name):
    with (HERE/(Path(name).stem+'.log')).open('a') as log:
        return subprocess.Popen([sys.executable, '-B', str(HERE/name)], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def science():
    return subprocess.check_output(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i|%j|%T|%C'], text=True).splitlines()


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    ready = 'FINAL_DELIVERY_PROTOCOL_ORDER_20261022.receipt.json'
    verify_receipt(ready)
    verify_receipt('SEALED_MAIN_OPTIONAL_RESUME_READY_20261022.receipt.json')
    verify_receipt('MIGRATION_NUMBERS.receipt.json')
    prefix = 'FINAL_DELIVERY_PROTOCOL_ORDER_DEPLOYMENT_20261022'
    assert not (HERE/(prefix+'.receipt.json')).exists()
    assert read(HERE/'STATUS.json')['phase'] == 'paper_dynamics_optional'
    assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
    entry = read(HERE/'ACTIVE_ENTRYPOINT.json')
    manager_names = [entry['watchdog'], entry['coordinator'], entry['downstream'], entry['optional']]
    managers = []
    for name in manager_names:
        exact = sys.executable+' -B '+str(HERE/name)
        pids = [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and command(int(p.name)) == exact]
        assert len(pids) == 1, (name, pids)
        managers.append((pids[0], name))
    before = science()
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # An action subprocess means a merger/planner/report boundary is active.
        # Abort before any signal in that case; science workers are Slurm children.
        manager_pids = {pid for pid, _ in managers}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            try:
                parent = int((p/'stat').read_text().split(') ', 1)[1].split()[1])
            except (FileNotFoundError, ProcessLookupError):
                continue
            if parent in manager_pids and int(p.name) not in manager_pids:
                assert not command(int(p.name)).startswith(sys.executable+' -B '+str(HERE)), 'Local action active; retry at boundary'
        write(HERE/(prefix+'_PRE.json'), dict(utc=utc(), active_entry=entry, managers=managers, science_jobs=before))
        for pid, _ in managers:
            os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not any(command(pid) == sys.executable+' -B '+str(HERE/name) for pid, name in managers):
                break
            time.sleep(.1)
        assert not any(command(pid) == sys.executable+' -B '+str(HERE/name) for pid, name in managers)
        entry.update(utc=utc(), coordinator='coordinator_sealed_main_resume_20261022.py',
                     downstream='sealed_main_optional_resume_20261022.py',
                     downstream_receipt='SEALED_MAIN_OPTIONAL_RESUME_READY_20261022.receipt.json',
                     optional='optional_delivery_protocol_order_20261022.py',
                     optional_receipt=ready, final_reports='final_reports_protocol_order_20261022.py',
                     protocol_order_receipt=ready)
        write(HERE/'ACTIVE_ENTRYPOINT.json', entry)
        root = start(entry['coordinator'])
        entry['pid'] = root.pid
        write(HERE/'ACTIVE_ENTRYPOINT.json', entry)
    deadline = time.monotonic()+45
    while read(HERE/'COORDINATOR_PID.json')['pid'] != root.pid:
        assert root.poll() is None and time.monotonic() < deadline
        time.sleep(.2)
    watch = start(entry['watchdog'])
    expected = sys.executable+' -B '+str(HERE/entry['optional'])
    while True:
        assert root.poll() is None and watch.poll() is None
        assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
        running = any(command(int(p.name)) == expected for p in Path('/proc').iterdir() if p.name.isdigit())
        if running and read(HERE/'STATUS.json')['phase'] == 'paper_dynamics_optional':
            break
        assert time.monotonic() < deadline
        time.sleep(.5)
    after = science()
    assert {s.split('|')[0] for s in before} == {s.split('|')[0] for s in after}, (before, after)
    write(HERE/(prefix+'.json'), dict(utc=utc(), passed=True, scientific_worker_signals=0,
          merger_signals=0, science_jobs_before=before, science_jobs_after=after,
          coordinator_pid=root.pid, watchdog_pid=watch.pid, active_entry=read(HERE/'ACTIVE_ENTRYPOINT.json')))
    seal(prefix+'.receipt.json', [Path(__file__), HERE/(prefix+'_PRE.json'), HERE/(prefix+'.json'), HERE/ready], report_only=True, scientific_worker_signals=0, merger_signals=0)
    with (HERE/'ITERATION_LOG.md').open('a') as log:
        log.write('\n'+utc()+'：最终报告仅调整段落顺序，完整性审计仍为首段，训练选定固定主对照紧随其后。通过已封存主线恢复入口替换本地管理 wrapper；相同17个科学作业持续运行，0科学worker信号、0merger信号、0新科学请求。见'+prefix+'.receipt.json。\n')
    print('PASS:report-only order fix active;identical science jobs continued', flush=True)


if __name__ == '__main__':
    main()
