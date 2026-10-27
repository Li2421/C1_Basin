"""Activate corrected delivery metadata without signaling any Slurm worker."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from deploy_idle_cpu_authorization import stop,launch
import subprocess,time

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
 verify_receipt('IDLE_CPU_DELIVERY_METADATA_DEPLOYMENT_CODE.receipt.json');verify_receipt('IDLE_CPU_DELIVERY_METADATA_CODE.receipt.json')
 entry=read(HERE/'ACTIVE_ENTRYPOINT.json');assert entry['coordinator']=='coordinator_flat_static_allcpu.py'
 stop(read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid'],'scheduler_watchdog_flat_static_allcpu.py')
 commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines();children=[]
 for line in commands:
  pid,cmd=line.strip().split(None,1)
  if cmd==sys.executable+' -B '+str(HERE/'downstream_flat_static_allcpu.py'):children.append(int(pid))
 assert len(children)==1,children
 stop(entry['pid'],'coordinator_flat_static_allcpu.py')
 for pid in children:stop(pid,'downstream_flat_static_allcpu.py')
 stop(read(HERE/'LOCAL_CPU_GUARD_PID.json')['pid'],'local_cpu_guard.py')
 entry.update(utc=utc(),coordinator='coordinator_flat_static_allcpu_delivery.py',downstream='downstream_flat_static_allcpu_delivery.py',optional='optional_delivery_flat_static_allcpu_delivery.py',watchdog='scheduler_watchdog_flat_static_allcpu_delivery.py',local_cpu_guard='local_cpu_guard_idle_cpu.py',final_reports='final_reports_flat_static_allcpu_delivery.py')
 write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
 entry['pid']=launch(entry['coordinator'],'coordinator.log');write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
 local_guard=launch(entry['local_cpu_guard'],'local_cpu_guard.log')
 for _ in range(100):
  if read(HERE/'COORDINATOR_PID.json')['pid']==entry['pid']:break
  time.sleep(.1)
 else:raise RuntimeError('Coordinator did not start')
 watchdog=launch(entry['watchdog'],'scheduler_watchdog.log')
 write(HERE/'IDLE_CPU_DELIVERY_METADATA_DEPLOYED.json',dict(utc=utc(),coordinator=entry['pid'],watchdog=watchdog,local_guard=local_guard,slurm_worker_signals=0,scientific_changes=0))
 watch=read(HERE/'CONTINUATION_WATCH_STATE.json');watch.update(utc=utc(),coordinator_pid=entry['pid'],active_entrypoint=entry,local_cpu_guard_pid=local_guard,pending_actions=[]);write(HERE/'CONTINUATION_WATCH_STATE.json',watch)
 for name in ('HANDOFF.md','ITERATION_LOG.md'):
  with (HERE/name).open('a') as out:out.write('\n\n'+utc()+': IDLE_CPU_DELIVERY_METADATA_CODE.receipt.json corrects future final HANDOFF names/policy and local affinity status to latest24-idle/18-shared CPU authority. ACTIVE_ENTRYPOINT selects coordinator_flat_static_allcpu_delivery.py, downstream_flat_static_allcpu_delivery.py, optional_delivery_flat_static_allcpu_delivery.py, scheduler_watchdog_flat_static_allcpu_delivery.py, local_cpu_guard_idle_cpu.py and final_reports_flat_static_allcpu_delivery.py. Dispatch/worker/scientific computation unchanged. No Slurm worker received a signal.\n')
 print(json.dumps(dict(coordinator=entry['pid'],watchdog=watchdog,local_guard=local_guard)))
if __name__=='__main__':main()
