"""Adapt scheduler log lookup to the new revised-worker stdout prefix."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from deploy_idle_cpu_authorization import stop,launch
import ast,difflib

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
 changes=[]
 def add(old,new,replacements):
  source=(HERE/old).read_text();text=source
  for a,b in replacements:assert a in text,(old,a);text=text.replace(a,b)
  normalized=text
  for a,b in reversed(replacements):normalized=normalized.replace(b,a)
  assert normalized==source;ast.parse(text);assert not (HERE/new).exists();(HERE/new).write_text(text)
  diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile=old,tofile=new)))
  changes.append(dict(original=old,original_sha256=sha(HERE/old),new=new,new_sha256=sha(HERE/new),diff=diff.name))
 add('infrastructure_recovery_active.py','infrastructure_recovery_correction_20261021.py',[("else 'worker'", "else 'revised' if '_revised_' in stage else 'worker'")])
 add('fine_dispatch_correction_20261021.py','fine_dispatch_correction_recovery_20261021.py',[
  ('from infrastructure_recovery_active import classify','from infrastructure_recovery_correction_20261021 import classify'),
  ("def submit(stage):\n", "def submit(stage):\n    verify_receipt('CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json')\n")])
 add('coordinator_correction_20261021.py','coordinator_correction_recovery_20261021.py',[
  ('import fine_dispatch_correction_20261021 as fine','import fine_dispatch_correction_recovery_20261021 as fine'),
  ("def main():\n", "def main():\n verify_receipt('CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json')\n"),
  ("entrypoint='coordinator_correction_20261021.py'","entrypoint='coordinator_correction_recovery_20261021.py'")])
 add('scheduler_watchdog_correction_20261021.py','scheduler_watchdog_correction_recovery_20261021.py',[
  ('CORRECTION_MONITOR_CODE_20261021.receipt.json','CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json'),
  ('scheduler_watchdog_correction_20261021.py','scheduler_watchdog_correction_recovery_20261021.py'),
  ('coordinator_correction_20261021.py','coordinator_correction_recovery_20261021.py')])
 write(HERE/'CORRECTION_RECOVERY_ADAPTER_SOURCE_DIFF_20261021.json',dict(utc=utc(),changes=changes,reason='Revised sbatch stdout is logs/revised_JOB.log; old recovery default looked for worker_JOB.log. New adapter reads actual revised log before deciding retries.',worker_or_scientific_source_changes=0,old_sealed_sources_unchanged=True))
 seal('CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json',[HERE/'repair_revised_recovery_paths_20261021.py',HERE/'CORRECTION_RECOVERY_ADAPTER_SOURCE_DIFF_20261021.json',HERE/'CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json']+[HERE/r[k] for r in changes for k in ('new','diff')],scheduling_log_adapter_only=True,no_worker_restart=True)
 entry=read(HERE/'ACTIVE_ENTRYPOINT.json');assert entry['coordinator']=='coordinator_correction_20261021.py'
 stop(read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid'],'scheduler_watchdog_correction_20261021.py');stop(entry['pid'],'coordinator_correction_20261021.py')
 entry.update(utc=utc(),coordinator='coordinator_correction_recovery_20261021.py',submit='fine_dispatch_correction_recovery_20261021.py',watchdog='scheduler_watchdog_correction_recovery_20261021.py',recovery='infrastructure_recovery_correction_20261021.py');write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
 entry['pid']=launch(entry['coordinator'],'coordinator_correction_20261021.log');write(HERE/'ACTIVE_ENTRYPOINT.json',entry);watchdog=launch(entry['watchdog'],'scheduler_watchdog_correction_20261021.log')
 write(HERE/'CORRECTION_RECOVERY_ADAPTER_DEPLOYED_20261021.json',dict(utc=utc(),coordinator=entry['pid'],watchdog=watchdog,worker_signals=0,worker_sources_unchanged=True))
 watch=read(HERE/'CONTINUATION_WATCH_STATE.json');watch.update(utc=utc(),coordinator_pid=entry['pid'],corrected_watchdog_pid=watchdog,active_entrypoint=entry,recovery_adapter='CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json');write(HERE/'CONTINUATION_WATCH_STATE.json',watch)
 for name in ('ITERATION_LOG.md','HANDOFF.md'):
  with (HERE/name).open('a') as out:out.write('\n\n'+utc()+': Corrected scheduler log recovery integration: revised workers write revised_JOB.log; new infrastructure_recovery_correction_20261021.py recognizes revised stage names and reads that log before recovery. New fine_dispatch_correction_recovery_20261021.py/coordinator_correction_recovery_20261021.py/scheduler_watchdog_correction_recovery_20261021.py active. Receipt CORRECTION_RECOVERY_ADAPTER_20261021; no worker signal, changed request, scientific code edit or rerun. Future continuation must import coordinator_correction_recovery_20261021 helpers/fine dispatcher, not older correction helper.\n')
 print(dict(coordinator=entry['pid'],watchdog=watchdog,worker_signals=0))
if __name__=='__main__':main()
