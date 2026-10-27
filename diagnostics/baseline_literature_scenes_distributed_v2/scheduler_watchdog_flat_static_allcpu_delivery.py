"""Persistent scheduler health checks; clean crashes can restart the coordinator.

Never cancels a live rollout, changes scientific code or retries an unknown.
"""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_idle_cpu import snapshot
import subprocess
import fcntl
import time


def alive(pid):
    try:
        os.kill(pid,0)
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ',1)[1].split()[0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False


def main():
    with (HERE/'SCHEDULER_WATCHDOG.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('PIPELINED_SCHEDULING.receipt.json');verify_receipt('WATCHDOG_OWNER_GUARD.receipt.json');verify_receipt('OPTIONAL_FINE_SCHEDULING.receipt.json');verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json')
        write(HERE/'SCHEDULER_WATCHDOG_PID.json',dict(utc=utc(),pid=os.getpid()))
        last_restart=0
        while not (HERE/'V2_COMPLETE.json').exists():
            try:
                entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
                pid=read(HERE/'COORDINATOR_PID.json')['pid'];running=alive(pid)
                state=snapshot();status=read(HERE/'STATUS.json')
                action_error=(HERE/'COORDINATOR_ACTION_ERROR.json').exists()
                processes=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
                child_alive=any(line.startswith(sys.executable+' -B '+str(HERE/'downstream_flat_static_allcpu_delivery.py')) for line in processes)
                # Scientific failures are surfaced, not blindly retried against
                # immutable receipts. Other running workers remain untouched.
                restarted=None
                if not running and not child_alive and not action_error and time.time()-last_restart>120:
                    assert entry['coordinator']=='coordinator_flat_static_allcpu_delivery.py',entry
                    verify_receipt('PIPELINED_SCHEDULING.receipt.json');verify_receipt('WATCHDOG_OWNER_GUARD.receipt.json');verify_receipt('OPTIONAL_FINE_SCHEDULING.receipt.json');verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json')
                    with (HERE/'coordinator.log').open('a') as out:
                        proc=subprocess.Popen([sys.executable,'-B',str(HERE/entry['coordinator'])],
                            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                    last_restart=time.time();restarted=proc.pid
                    entry.update(pid=proc.pid,utc=utc());write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
                duty=read(HERE/'SCHEDULER_DUTY.json') if (HERE/'SCHEDULER_DUTY.json').exists() else None
                write(HERE/'SCHEDULER_WATCHDOG.json',dict(utc=utc(),poll_seconds=30,
                    coordinator_pid=pid,coordinator_alive=running,downstream_alive=child_alive,restarted_pid=restarted,
                    action_error=action_error,phase=status['phase'],resources=state,
                    scheduler_duty=duty,no_rollout_cancellations=True,no_scientific_changes=True))
            except Exception as exc:
                write(HERE/'SCHEDULER_WATCHDOG_ERROR.json',dict(utc=utc(),error=repr(exc)))
            time.sleep(30)
        write(HERE/'SCHEDULER_WATCHDOG_EXIT.json',dict(utc=utc(),reason='V2 complete'))


if __name__=='__main__':main()
