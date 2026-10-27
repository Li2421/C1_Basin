"""Keep the dated CPU-release guard alive independently of scientific orchestration."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess,time,fcntl

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def main():
 with (HERE/'CPU_RELEASE_GUARD_WATCHDOG_20261023.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_receipt('CPU_RELEASE_GUARD_WATCHDOG_20261023.receipt.json')
  write(HERE/'CPU_RELEASE_GUARD_WATCHDOG_PID_20261023.json',dict(utc=utc(),pid=os.getpid()))
  child=None
  while not (HERE/'V2_COMPLETE.json').exists():
   entry='resource_guard_corrections_release_cpu_only_20261023.py';registered=read(HERE/'RESOURCE_GUARD_PID.json');pid=int(registered['pid']);running=alive(pid)
   if running:
    assert str(HERE/entry).encode() in Path(f'/proc/{pid}/cmdline').read_bytes(),registered
   if not running and (child is None or child.poll() is not None):
    verify_receipt('CPU_RELEASE_CPU_ONLY_GUARD_20261023.receipt.json')
    with (HERE/'resource_guard_corrections_release_cpu_only_20261023.log').open('a') as out:child=subprocess.Popen([sys.executable,'-B',str(HERE/entry)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(HERE/'CPU_RELEASE_GUARD_RESTART_20261023.json',dict(utc=utc(),previous_pid=pid,new_pid=child.pid,guard_only=True,scientific_workers_changed=False))
   write(HERE/'CPU_RELEASE_GUARD_WATCHDOG_20261023.json',dict(utc=utc(),guard_registered_pid=pid,guard_alive=running,release_policy='RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json',restarted_guard_pid=None if child is None else child.pid,poll_seconds=10))
   time.sleep(10)
if __name__=='__main__':main()
