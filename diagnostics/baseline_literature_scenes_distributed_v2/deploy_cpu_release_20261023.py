from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess,signal,time
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
verify_receipt('CPU_RELEASE_CODE_20261023.receipt.json')
old=[(390969,'scheduler_watchdog_correction_owner_guard_20261021.py'),(373851,'coordinator_correction_recovery_20261021.py'),(340528,'resource_guard_corrections_20261021.py'),(340529,'heartbeat_corrections_20261021.py'),(366318,'progress_correction_20261021.py')]
for pid,name in old:
 path=Path(f'/proc/{pid}/cmdline')
 if path.exists():
  assert str(HERE/name).encode() in path.read_bytes(),(pid,name)
  os.kill(pid,signal.SIGTERM)
for _ in range(100):
 if all(not Path(f'/proc/{pid}/cmdline').exists() or not Path(f'/proc/{pid}/cmdline').read_bytes() for pid,_ in old):break
 time.sleep(.1)
else:raise RuntimeError('Local scheduler processes failed to exit')
entry=read(HERE/'ACTIVE_ENTRYPOINT.json')
entry.update(utc=utc(),coordinator='coordinator_correction_release_20261023.py',submit='fine_dispatch_correction_release_20261023.py',resources='resources_corrections_release_20261023.py',watchdog='scheduler_watchdog_correction_release_20261023.py',resource_guard='resource_guard_corrections_release_20261023.py',scene_learning_scheduler='scene_learning_scheduler_revised_release_20261023.py',total_cpu=24,total_gpu_shards=2,cpu_authority='Latest user explicit24 now ->18 Oct23 09:00 Asia/Singapore',resource_policy='RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json')
write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
launched={}
def start(name):
 with (HERE/(name+'.log')).open('a') as out:
  p=subprocess.Popen([sys.executable,'-B',str(HERE/(name+'.py'))],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 launched[name]=p.pid;return p
for name in ('resource_guard_corrections_release_20261023','heartbeat_corrections_release_20261023','progress_correction_release_20261023'):start(name)
p=start('coordinator_correction_release_20261023');entry.update(pid=p.pid);write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
for _ in range(300):
 if p.poll() is not None:raise RuntimeError('New root failed')
 registered=read(HERE/'COORDINATOR_PID.json')
 if registered['pid']==p.pid:break
 time.sleep(.1)
else:raise RuntimeError('New coordinator not registered')
watch=start('scheduler_watchdog_correction_release_20261023')
state=read(HERE/'CONTINUATION_WATCH_STATE.json');state.update(utc=utc(),coordinator_pid=p.pid,corrected_watchdog_pid=watch.pid,corrected_progress_pid=launched['progress_correction_release_20261023'],resource_module=entry['resources'],resource_policy=entry['resource_policy'],total_cpu=24,worker_cpu_slots=23,total_gpu_shards=2,resource_caps=dict(total_cpu=24,scheduled_cpu=23,local_cpu_shares=1,gpu_shards=2),active_entrypoint=entry)
write(HERE/'CONTINUATION_WATCH_STATE.json',state)
write(HERE/'CPU_RELEASE_DEPLOYED_20261023.json',dict(utc=utc(),old_local_processes=old,new_local_processes=launched,scientific_workers_interrupted=0,policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTIONS_RELEASE_20261023.json'),coordinator_registered_before_watchdog=True))
with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n'+utc()+'：用户明确纠正资源为当前24 CPU、10月23日09:00新加坡时间降到18 CPU，GPU仍2份额。新增资源模块与全套提交器静态变体并封存；只重启本地协调/监控，不打断现有科学worker。07:30开始6个超额worker逐局收尾；08:59必要时只中断超额新语义作业以保证09:00释放，未完成请求按原身份继续、绝不记unknown。\n')
print(json.dumps(launched),flush=True)
