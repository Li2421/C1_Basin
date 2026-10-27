"""Apply correction prompt resource18/2, draining existing episodes only."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from deploy_idle_cpu_authorization import stop,launch
import ast,subprocess,difflib

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
 verify_receipt('RESOURCE_RELEASE_CORRECTION.receipt.json')
 prompt=ROOT/'docs/prompt_baseline_literature_scenes_distributed_v2_corrections_20261021.md';stat=prompt.stat()
 write(HERE/'CORRECTION_PROMPT_STAT_20261021.json',dict(utc=utc(),path=str(prompt),st_size=stat.st_size,st_mtime_ns=stat.st_mtime_ns,st_ctime_ns=stat.st_ctime_ns,st_ino=stat.st_ino,audit_rule='New user prompt recorded by os.stat only; no protected opening hash mismatch'))
 src=(HERE/'resources_released.py').read_text();dest=HERE/'resources_corrections_20261021.py';assert not dest.exists();dest.write_text(src)
 policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTED.json');assert policy['after']==dict(total_cpu_shares=18,total_gpu_shards=2)
 write(HERE/'CORRECTION_RESOURCE_AUTHORITY_20261021.json',dict(utc=utc(),authority='Latest user explicitly invokes corrections document; its resource section caps all user tasks at18 CPU/2GPU shares, overriding earlier idle24 authorization',resource_module=dest.name,total_cpu=18,local_cpu_reserve=1,maximum_scheduled_cpu=17,gpu_shards=2,existing_old_workers='Already all signaled to finish current episodes; no hard cancellations',prompt_stat_file='CORRECTION_PROMPT_STAT_20261021.json'))
 guard_src=(HERE/'resource_guard_idle_cpu.py').read_text().replace('resources_idle_cpu','resources_corrections_20261021').replace('IDLE_CPU_AUTHORIZATION_CODE.receipt.json','CORRECTION_RESOURCE_CODE_20261021.receipt.json').replace('resource_guard_idle_cpu.py','resource_guard_corrections_20261021.py')
 guard=HERE/'resource_guard_corrections_20261021.py';guard.write_text(guard_src);ast.parse(guard_src)
 heartbeat_src=(HERE/'heartbeat_released.py').read_text().replace('resources_released','resources_corrections_20261021');heartbeat=HERE/'heartbeat_corrections_20261021.py';heartbeat.write_text(heartbeat_src);ast.parse(heartbeat_src)
 seal('CORRECTION_RESOURCE_CODE_20261021.receipt.json',[HERE/n for n in ('activate_correction_resource_policy.py','resources_corrections_20261021.py','resource_guard_corrections_20261021.py','heartbeat_corrections_20261021.py','CORRECTION_RESOURCE_AUTHORITY_20261021.json','CORRECTION_PROMPT_STAT_20261021.json')],total_cpu=18,gpu_shards=2,no_running_rollout_hard_cancel=True,old_sources_unchanged=True)
 stop(read(HERE/'RESOURCE_GUARD_PID.json')['pid'],'resource_guard_idle_cpu.py')
 for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
  pid,cmd=line.strip().split(None,1)
  if cmd==sys.executable+' -B '+str(HERE/'heartbeat_released_allcpu.py'):stop(int(pid),'heartbeat_released_allcpu.py')
 pids=dict(guard=launch(guard.name,'resource_guard_corrections_20261021.log'),heartbeat=launch(heartbeat.name,'heartbeat.log'))
 entry=read(HERE/'ACTIVE_ENTRYPOINT.json');entry.update(utc=utc(),total_cpu=18,total_gpu_shards=2,resources=dest.name,resource_guard=guard.name,state='Old coordinators intentionally stopped; correction implementation and no-rollout gates pending',do_not_restart_old_coordinator=True);write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
 watch=read(HERE/'CONTINUATION_WATCH_STATE.json');watch.update(utc=utc(),correction_prompt_found=True,correction_prompt_read_completely=True,coordinator_intentionally_stopped=True,watchdog_intentionally_stopped=True,total_cpu=18,worker_cpu_slots=17,total_gpu_shards=2,resource_module=dest.name,active_entrypoint=entry,pending_actions=['Implement correction revision and complete all no-rollout unit/static gates','Reclassify instances using saved outcomes and W5/tau.05 diagnosis plus9-grid sensitivity','Seal addendum and Rc numeric table before first revised rollout','Follow mandated revised stage order; never launch old circle learning evaluation'],prior_runtime_forecast_superseded_by_new_correction_order=True);write(HERE/'CONTINUATION_WATCH_STATE.json',watch)
 for name in ('ITERATION_LOG.md','HANDOFF.md'):
  with (HERE/name).open('a') as out:out.write('\n\n'+utc()+': CORRECTION PROMPT FOUND AND READ COMPLETELY. Latest user correction instruction explicitly sets18 totalCPU (17scheduled+1local),2GPU shares, superseding idle24 CPU authorization. resources_corrections_20261021.py and graceful resource_guard_corrections_20261021.py active. Originalsealedsources/menus/results remain unchanged. Old coordinator/downstream/watchdog stay stopped,23 old-circle workers drain only their current episode. No old-circle evaluation. Revised semantics requires new identities except exact η0 reuse after32 saved replay gates. Next: revision/unit/static gates+reclassification, Dfixed4scenes, Dlearned narrow/crowd/door, Cfixed4scenes, circle revisedlearning last, optional revised dynamics/variants, finalreports/audit. New runs are user-requested semantic revisions, not retries of an implementation failure.\n')
 print(json.dumps(pids))
if __name__=='__main__':main()
