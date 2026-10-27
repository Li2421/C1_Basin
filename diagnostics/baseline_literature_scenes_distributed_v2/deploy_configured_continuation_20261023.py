from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess,signal,time
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
verify_receipt('CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json');verify_receipt('CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json')
entry=read(HERE/'ACTIVE_ENTRYPOINT.json');oldroot=read(HERE/'COORDINATOR_PID.json')['pid'];oldwatch=read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid']
assert not (HERE/'CORRECTION_CONTINUATION_PID_20261021.json').exists(),'Do not change continuation dispatch while a child is active'
for pid,name in [(oldwatch,entry['watchdog']),(oldroot,entry['coordinator'])]:
 assert str(HERE/name).encode() in Path(f'/proc/{pid}/cmdline').read_bytes(),(pid,name)
 os.kill(pid,signal.SIGTERM)
for _ in range(100):
 if all(not Path(f'/proc/{pid}/cmdline').exists() or not Path(f'/proc/{pid}/cmdline').read_bytes() for pid in (oldwatch,oldroot)):break
 time.sleep(.1)
else:raise RuntimeError('Previous coordinator failed to stop')
entry.update(utc=utc(),coordinator='coordinator_correction_configured_20261023.py',watchdog='scheduler_watchdog_correction_configured_20261023.py',downstream='correction_after_distributed_fixed_parallel_20261023.py',downstream_receipt='CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json',main_statistics='statistics_parallel_correction_20261023.py',main_statistics_ready='CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json',live_revised_fixed_success='LIVE_REVISED_FIXED_SUCCESS.json',live_revised_controller_success='LIVE_REVISED_FIXED_CONTROLLER_SUCCESS.csv',resource_request='CPU_RELEASE_USER_CORRECTED_20261023.json')
write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
def launch(name):
 with (HERE/(name+'.log')).open('a') as out:return subprocess.Popen([sys.executable,'-B',str(HERE/(name+'.py'))],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
p=launch('coordinator_correction_configured_20261023');entry.update(pid=p.pid);write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
for _ in range(300):
 if p.poll() is not None:raise RuntimeError('Configured coordinator failed')
 if read(HERE/'COORDINATOR_PID.json')['pid']==p.pid:break
 time.sleep(.1)
else:raise RuntimeError('New root did not register')
w=launch('scheduler_watchdog_correction_configured_20261023')
state=read(HERE/'CONTINUATION_WATCH_STATE.json');state.update(utc=utc(),coordinator_pid=p.pid,corrected_watchdog_pid=w.pid,active_entrypoint=entry,parallel_numerical_ready='CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json',configured_continuation_ready='CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json');write(HERE/'CONTINUATION_WATCH_STATE.json',state)
write(HERE/'CONFIGURED_CONTINUATION_DEPLOYED_20261023.json',dict(utc=utc(),previous_root_pid=oldroot,previous_watchdog_pid=oldwatch,coordinator_pid=p.pid,watchdog_pid=w.pid,science_worker_processes_interrupted=0,science_request_parameters_changed=False,native_statistical_kernel_unchanged=True,parallel_tasks=27,entrypoint=entry))
with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n'+utc()+'：新增并行统计：27个互相独立的原生计算任务，保留5000次bootstrap、原随机种子、等权/配对、unknown识别界；既有完整旧统计直接复用。128份诊断读取所需字段一致，统计函数AST相同，5000抽样数组逐位一致，串/并行分类增益JSON一致。只重启本地协调/看门狗，不中断任何科学worker。新协调器仅按ACTIVE中有封存收据和源码哈希的本轮入口派发后续进程，不作运行时模块补丁。\n')
print('Configured continuation with parallel native statistics deployed',p.pid,w.pid,flush=True)
