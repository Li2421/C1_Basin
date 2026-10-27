"""Monitor only user-authorized corrected coordinator, never restart old semantics."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_corrections_release_20261023 import snapshot
import subprocess,time,fcntl

def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False

def owner_lock_held():
 with (HERE/'CORRECTION_COORDINATOR.lock').open('a') as probe:
  try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(probe,fcntl.LOCK_UN);return False

def main():
 with (HERE/'SCHEDULER_WATCHDOG.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_receipt('CPU_RELEASE_CODE_20261023.receipt.json')
  write(HERE/'SCHEDULER_WATCHDOG_PID.json',dict(utc=utc(),pid=os.getpid(),entrypoint='scheduler_watchdog_correction_release_20261023.py'));last_restart=0
  while not (HERE/'V2_COMPLETE.json').exists():
   entry=read(HERE/'ACTIVE_ENTRYPOINT.json');pid=read(HERE/'COORDINATOR_PID.json')['pid'];running=alive(pid)
   commands=subprocess.check_output(['ps','-eo','args='],text=True).splitlines();child=any(c==sys.executable+' -B '+str(HERE/'correction_after_distributed_fixed_20261021.py') for c in commands)
   error=(HERE/'COORDINATOR_ACTION_ERROR.json').exists();restarted=None
   candidate=int(entry.get('pid',0));candidate_alive=alive(candidate) if candidate>0 else False
   if not running and candidate_alive:
    cmd=Path(f'/proc/{candidate}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(HERE/entry['coordinator']) in cmd:pid=candidate;running=True
   lock_held=owner_lock_held() if not running else True
   if not running and not child and not lock_held and not error and time.time()-last_restart>120:
    assert entry['coordinator']=='coordinator_correction_release_20261023.py',entry;verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json')
    with (HERE/'coordinator_correction_20261021.log').open('a') as out:proc=subprocess.Popen([sys.executable,'-B',str(HERE/entry['coordinator'])],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    last_restart=time.time();restarted=proc.pid;entry.update(pid=proc.pid,utc=utc());write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
   write(HERE/'SCHEDULER_WATCHDOG.json',dict(utc=utc(),poll_seconds=30,coordinator_pid=pid,coordinator_alive=running,downstream_alive=child,restarted_pid=restarted,action_error=error,phase=read(HERE/'STATUS.json')['phase'],resources=snapshot(),correction_prompt_active=True,old_semantics_never_restarted=True,coordinator_owner_lock_held=lock_held))
   time.sleep(30)
if __name__=='__main__':main()
