"""Repair missing CLI alias;preserve sealed code and completed evidence."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess,signal,time,fcntl,ast


def command(pid):
    path=Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0',b' ').decode().strip() if path.exists() else ''


def start(script,log):
    with (HERE/log).open('a') as stream:return subprocess.Popen([sys.executable,'-B',str(HERE/script)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    proof='BOOTSTRAP_EXACT_MEMO_LAUNCHER_FIX_20261022.receipt.json';assert not (HERE/proof).exists()
    verify_receipt('BOOTSTRAP_EXACT_MEMO_READY_20261022.receipt.json')
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json');error=read(HERE/'COORDINATOR_ACTION_ERROR.json')
    log=HERE/'coordinator_actions/correction_combined_original_revised_numbers_1792646288409912293.log';assert "can't open file" in log.read_text() and 'statistics_bootstrap_memo_20261022.py' in log.read_text()
    assert not (HERE/'PARALLEL_NUMERIC_INPUTS_BOOTSTRAP_MEMO_20261022.receipt.json').exists()
    # Both existing call sites now resolve to this literal alias;the actual
    # driver and its complete5000-draw mathematical kernel stay sealed.
    child=(HERE/entry['downstream']).read_text();assert "'statistics_bootstrap_memo_20261022.py'" in child
    ast.parse((HERE/'statistics_bootstrap_memo_20261022.py').read_text())
    write(HERE/'BOOTSTRAP_EXACT_MEMO_LAUNCHER_FIX_20261022.json',dict(utc=utc(),cause='Python and sbatch source filenames share Path.stem;batch target overwrote the Python target in the static substitution map',repair='Add explicitly sealed CLI alias that calls the already sealed statistics_driver_bootstrap_memo_20261022.main',original_error=error,completed_main_requests=188352,rollouts_to_repeat=0,scientific_worker_signals=0,math_and_statistics_unchanged=True))
    seal(proof,[Path(__file__),HERE/'statistics_bootstrap_memo_20261022.py',HERE/'statistics_driver_bootstrap_memo_20261022.py',HERE/entry['downstream'],HERE/'BOOTSTRAP_EXACT_MEMO_READY_20261022.receipt.json',HERE/'BOOTSTRAP_EXACT_MEMO_LAUNCHER_FIX_20261022.json',log],launch_path_fixed=True,statistics_unchanged=True,new_rollouts=0)
    pid=read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid']
    if command(pid)==sys.executable+' -B '+str(HERE/entry['watchdog']):os.kill(pid,signal.SIGTERM)
    for _ in range(40):
        if command(pid)!=sys.executable+' -B '+str(HERE/entry['watchdog']):break
        time.sleep(.1)
    for name in ('CORRECTION_COORDINATOR.lock','CORRECTION_CONTINUATION.lock'):
        with (HERE/name).open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    archive=HERE/'COORDINATOR_ACTION_ERROR_BOOTSTRAP_LAUNCH_20261022.json';assert not archive.exists();(HERE/'COORDINATOR_ACTION_ERROR.json').rename(archive)
    entry.update(utc=utc(),downstream_receipt=proof,main_statistics='statistics_bootstrap_memo_20261022.py',main_statistics_ready=proof)
    root=start(entry['coordinator'],'coordinator_correction_20261021.log');entry['pid']=root.pid;write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
    for _ in range(100):
        assert root.poll() is None,'Root failed to restart'
        if read(HERE/'COORDINATOR_PID.json')['pid']==root.pid:break
        time.sleep(.1)
    watcher=start(entry['watchdog'],'scheduler_watchdog_bootstrap_launcher_recovery_20261022.log')
    write(HERE/'BOOTSTRAP_EXACT_MEMO_LAUNCHER_RECOVERY_20261022.json',dict(utc=utc(),root_pid=root.pid,watchdog_pid=watcher.pid,preserved_error=str(archive),fix_receipt=proof,scientific_worker_signals=0))
    print('Recovered sealed statistics CLI;no rollout repeated',root.pid,watcher.pid,flush=True)


if __name__=='__main__':main()
