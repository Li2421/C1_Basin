"""Switch only orchestration; old workers finish their current complete episode."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_released import snapshot
import subprocess
import signal
import time
import fcntl


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def stop_local(pid,expected):
    path=Path(f'/proc/{pid}/cmdline')
    if not path.exists():return
    command=path.read_bytes().replace(b'\0',b' ').decode()
    assert expected in command,(pid,command)
    assert 'worker_' not in command,command
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not alive(pid):return
        time.sleep(.1)
    raise RuntimeError(('Local orchestration did not exit',pid))


def main():
    with (HERE/'FLAT_CACHE_DEPLOYMENT.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json')
        verify_receipt('CBS_FLAT_CACHE_EQUIVALENCE.receipt.json')
        assert read(HERE/'CBS_FLAT_CACHE_FULL_REPLAY.json')['passed']
        assert read(HERE/'CBS_FLAT_CACHE_MATRIX.json')['passed']
        from local_cpu_guard import allocated_cpus
        occupied,_=allocated_cpus()
        assert len(os.sched_getaffinity(0))==1 and not (set(os.sched_getaffinity(0)) & occupied)
        entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
        assert entry['coordinator']=='coordinator_complete.py',entry
        assert read(HERE/'STATUS.json')['phase']=='scene_learning_pipeline'
        assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
        state=snapshot();assert state['free_cpu_shares']==0,state
        # No submissions can race this migration: all17 scheduled CPU shares
        # are occupied, there is no ready critic, and scheduler is in its loop.
        duty=read(HERE/'SCHEDULER_DUTY.json');assert not duty['ready_critics'],duty
        assert duty['pending_rollout_stages']==['learning_labels_lit_circle_five'],duty
        rows={}
        for path in (HERE/'submissions').glob('*.json'):
            row=read(path);rows[str(row['job_id'])]=row
        old_jobs=[j['job_id'] for j in state['jobs'] if rows.get(j['job_id'],{}).get('stage')=='learning_labels_lit_circle_five']
        assert len(old_jobs)==17,old_jobs
        before={str(path):sha(path) for path in (HERE/'learning_labels_lit_circle_five'/'raw').glob('*.json')}
        before_path=HERE/'CBS_FLAT_CACHE_PREMIGRATION_RAW_HASHES.json';write(before_path,before)
        write(HERE/'CBS_FLAT_CACHE_MIGRATION_BEFORE.json',dict(utc=utc(),entrypoint=entry,resources=state,old_jobs=old_jobs,existing_raw_count=len(before),no_rollout_signals_yet=True))
        watchdog=read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid']
        stop_local(watchdog,'scheduler_watchdog_complete.py')
        processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
        children=[]
        for line in processes:
            pid,command=line.strip().split(None,1)
            if command==sys.executable+' -B '+str(HERE/'downstream_complete.py'):children.append(int(pid))
        assert len(children)==1,children
        stop_local(entry['pid'],'coordinator_complete.py')
        for pid in children:stop_local(pid,'downstream_complete.py')
        with (HERE/'coordinator.log').open('a') as stream:
            coordinator=subprocess.Popen([sys.executable,'-B',str(HERE/'coordinator_flat_static.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        entry.update(local_cpu_guard='local_cpu_guard.py',native_backend='tiled_planning_flat_static.py',utc=utc(),coordinator='coordinator_flat_static.py',pid=coordinator.pid,worker='worker_flat_static.py',submit='fine_dispatch_flat_static.py',downstream='downstream_flat_static.py',optional='optional_delivery_flat_static.py',optional_worker='optional_worker_flat_static.py',watchdog='scheduler_watchdog_flat_static.py',computation_acceleration='Exact native geometry-cache dictionary layout')
        write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if (HERE/'COORDINATOR_ACTION_ERROR.json').exists():raise RuntimeError(read(HERE/'COORDINATOR_ACTION_ERROR.json'))
            if read(HERE/'COORDINATOR_PID.json')['pid']==coordinator.pid:
                commands=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
                if any(c==sys.executable+' -B '+str(HERE/'downstream_flat_static.py') for c in commands):break
            time.sleep(1)
        else:raise RuntimeError('New coordinator/downstream not ready; existing workers remain untouched')
        with (HERE/'scheduler_watchdog.log').open('a') as stream:
            watch=subprocess.Popen([sys.executable,'-B',str(HERE/'scheduler_watchdog_flat_static.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        # USR1 is handled after the current saved episode by the unchanged old
        # worker. No hard cancellation, truncated label or unknown is produced.
        for job in old_jobs:
            result=subprocess.run(['scancel','--batch','--signal=USR1',job],capture_output=True,text=True)
            assert result.returncode==0,(job,result.stderr)
        write(HERE/'CBS_FLAT_CACHE_MIGRATION_INITIATED.json',dict(utc=utc(),coordinator_pid=coordinator.pid,watchdog_pid=watch.pid,old_jobs=old_jobs,existing_raw_count=len(before),raw_hash_file=str(before_path),graceful_current_episode_drain=True,hard_cancelled_rollouts=0,resources=snapshot()))
        print('New coordinator ready;17 old workers will finish current episode then yield to equivalent cache backend',flush=True)


if __name__=='__main__':main()
