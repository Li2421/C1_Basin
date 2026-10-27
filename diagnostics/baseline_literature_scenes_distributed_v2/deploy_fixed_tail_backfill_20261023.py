from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess,signal,time
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
verify_receipt('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json')
entry=read(HERE/'ACTIVE_ENTRYPOINT.json');oldroot=read(HERE/'COORDINATOR_PID.json')['pid'];oldwatch=read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid']
for pid,name in [(oldwatch,entry['watchdog']),(oldroot,entry['coordinator'])]:
 assert str(HERE/name).encode() in Path(f'/proc/{pid}/cmdline').read_bytes(),(pid,name)
 os.kill(pid,signal.SIGTERM)
for _ in range(100):
 if all(not Path(f'/proc/{pid}/cmdline').exists() or not Path(f'/proc/{pid}/cmdline').read_bytes() for pid in (oldwatch,oldroot)):break
 time.sleep(.1)
else:raise RuntimeError('Previous local coordinator did not exit')
entry.update(utc=utc(),coordinator='coordinator_correction_tail_backfill_20261023.py',watchdog='scheduler_watchdog_correction_tail_backfill_20261023.py',downstream='correction_after_distributed_fixed_tail_backfill_20261023.py',fixed_tail_backfill=True)
write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
def launch(name):
 with (HERE/(name+'.log')).open('a') as out:return subprocess.Popen([sys.executable,'-B',str(HERE/(name+'.py'))],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
p=launch('coordinator_correction_tail_backfill_20261023');entry.update(pid=p.pid);write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
for _ in range(300):
 if p.poll() is not None:raise RuntimeError('Tail-backfill coordinator failed')
 if read(HERE/'COORDINATOR_PID.json')['pid']==p.pid:break
 time.sleep(.1)
else:raise RuntimeError('Tail-backfill root not registered')
w=launch('scheduler_watchdog_correction_tail_backfill_20261023')
state=read(HERE/'CONTINUATION_WATCH_STATE.json');state.update(utc=utc(),coordinator_pid=p.pid,corrected_watchdog_pid=w.pid,active_entrypoint=entry,fixed_tail_backfill_ready='CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json');write(HERE/'CONTINUATION_WATCH_STATE.json',state)
write(HERE/'FIXED_TAIL_BACKFILL_DEPLOYED_20261023.json',dict(utc=utc(),previous_root_pid=oldroot,previous_watchdog_pid=oldwatch,coordinator_pid=p.pid,watchdog_pid=w.pid,scientific_workers_interrupted=0,strict_phase_order_unchanged=True,scheduling_only=True,main_requests_always_priority=True))
with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n'+utc()+'：固定阶段长尾若有空闲CPU且没有未领取的研究请求，用已完成固定轨迹做必要诊断；新研究请求出现即让诊断完整样本收尾退出。只新增静态协调器/下游/看门狗变体并封存，科学worker、case UID、源轨迹、物理、安全、随机序列、阶段顺序均不变，23个现有worker未打断。\n')
print('Fixed-tail backfill deployed',p.pid,w.pid,flush=True)
