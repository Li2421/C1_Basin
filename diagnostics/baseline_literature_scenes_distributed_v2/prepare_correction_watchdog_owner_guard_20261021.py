"""Keep restart checks behind both PID registration and the actual owner lock."""
from experiment import *
from atomic_write import write
from deploy_idle_cpu_authorization import stop,launch
import ast,difflib
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
old=HERE/'scheduler_watchdog_correction_recovery_20261021.py';text=old.read_text()
needle='def main():\n'
insert='''def owner_lock_held():
 with (HERE/'CORRECTION_COORDINATOR.lock').open('a') as probe:
  try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(probe,fcntl.LOCK_UN);return False

def main():
'''
text=text.replace(needle,insert)
text=text.replace("CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json","CORRECTION_WATCHDOG_OWNER_GUARD_20261021.receipt.json")
text=text.replace('scheduler_watchdog_correction_recovery_20261021.py','scheduler_watchdog_correction_owner_guard_20261021.py')
text=text.replace("error=(HERE/'COORDINATOR_ACTION_ERROR.json').exists();restarted=None", "error=(HERE/'COORDINATOR_ACTION_ERROR.json').exists();restarted=None\n   candidate=int(entry.get('pid',0));candidate_alive=alive(candidate) if candidate>0 else False\n   if not running and candidate_alive:\n    cmd=Path(f'/proc/{candidate}/cmdline').read_bytes().replace(b'\\0',b' ').decode()\n    if str(HERE/entry['coordinator']) in cmd:pid=candidate;running=True\n   lock_held=owner_lock_held() if not running else True")
text=text.replace('if not running and not child and not error and time.time()-last_restart>120:', 'if not running and not child and not lock_held and not error and time.time()-last_restart>120:')
text=text.replace('old_semantics_never_restarted=True)', 'old_semantics_never_restarted=True,coordinator_owner_lock_held=lock_held)')
ast.parse(text);new=HERE/'scheduler_watchdog_correction_owner_guard_20261021.py';assert not new.exists();new.write_text(text)
(HERE/'CORRECTION_WATCHDOG_OWNER_GUARD_20261021.diff').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.splitlines(True),fromfile=old.name,tofile=new.name)))
seal('CORRECTION_WATCHDOG_OWNER_GUARD_20261021.receipt.json',[HERE/n for n in ('prepare_correction_watchdog_owner_guard_20261021.py',new.name,'CORRECTION_WATCHDOG_OWNER_GUARD_20261021.diff','CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json')],scientific_changes=0,worker_signals=0,reason='One redundant startup attempt encountered held coordinator lock; original live coordinator continued. Guard now checks registered PID, entrypoint startup PID and owner lock before restart.')
entry=read(HERE/'ACTIVE_ENTRYPOINT.json');pid=read(HERE/'COORDINATOR_PID.json')['pid'];assert Path(f'/proc/{pid}/cmdline').exists()
stop(read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid'],'scheduler_watchdog_correction_recovery_20261021.py')
entry.update(utc=utc(),pid=pid,watchdog=new.name);write(HERE/'ACTIVE_ENTRYPOINT.json',entry);watchdog=launch(new.name,'scheduler_watchdog_correction_20261021.log')
write(HERE/'CORRECTION_WATCHDOG_OWNER_GUARD_DEPLOYED_20261021.json',dict(utc=utc(),coordinator_pid=pid,watchdog_pid=watchdog,coordinator_not_restarted=True,worker_signals=0))
w=read(HERE/'CONTINUATION_WATCH_STATE.json');w.update(utc=utc(),active_entrypoint=entry,coordinator_pid=pid,corrected_watchdog_pid=watchdog);write(HERE/'CONTINUATION_WATCH_STATE.json',w)
with (HERE/'ITERATION_LOG.md').open('a') as out:out.write('\n\n'+utc()+': Corrected watchdog startup race found in integration: a redundant coordinator launch lost the owner lock; true coordinator373851 remained live, all17worker jobs continued,0worker interruption. New owner-guard watchdog checks entrypoint PID plus actual CORRECTION_COORDINATOR.lock before restart, and ACTIVE_ENTRYPOINT PID restored from authoritative COORDINATOR_PID. New receipt CORRECTION_WATCHDOG_OWNER_GUARD_20261021.\n')
print(dict(coordinator_unchanged=pid,new_watchdog=watchdog,worker_signals=0))
