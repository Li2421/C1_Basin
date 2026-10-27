"""Replace local schedulers only; all running rollout workers remain untouched."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess,signal,time

def stop(pid,expected):
    path=Path(f'/proc/{pid}/cmdline')
    if not path.exists():return
    cmd=path.read_bytes().replace(b'\0',b' ').decode();assert expected in cmd,(pid,cmd)
    assert 'worker_' not in cmd,cmd
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        try:dead=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]=='Z'
        except FileNotFoundError:dead=True
        if dead:return
        time.sleep(.1)
    raise RuntimeError(('local scheduler did not exit',pid))

def launch(name,log):
    with (HERE/log).open('a') as out:
        return subprocess.Popen([sys.executable,'-B',str(HERE/name)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True).pid

def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('IDLE_CPU_AUTHORIZATION_DEPLOYMENT_CODE.receipt.json')
    verify_receipt('IDLE_CPU_AUTHORIZATION_CODE.receipt.json')
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json');assert entry['coordinator']=='coordinator_flat_static.py',entry
    write(HERE/'IDLE_CPU_AUTHORIZATION_BEFORE.json',dict(utc=utc(),entrypoint=entry,all_jobs=subprocess.check_output(['squeue','-h','-o','%u|%i|%j|%T|%C|%b'],text=True)))
    stop(read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid'],'scheduler_watchdog_flat_static.py')
    stop(read(HERE/'RESOURCE_GUARD_PID.json')['pid'],'resource_guard_released.py')
    commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
    targets=[]
    for line in commands:
        pid,cmd=line.strip().split(None,1)
        for name in ('downstream_flat_static.py','heartbeat_released.py'):
            if cmd==sys.executable+' -B '+str(HERE/name):targets.append((int(pid),name))
    assert sum(name=='downstream_flat_static.py' for _,name in targets)==1,targets
    stop(entry['pid'],'coordinator_flat_static.py')
    for pid,name in targets:stop(pid,name)
    entry.update(utc=utc(),coordinator='coordinator_flat_static_allcpu.py',downstream='downstream_flat_static_allcpu.py',submit='fine_dispatch_flat_static_allcpu.py',scene_learning_scheduler='scene_learning_scheduler_flat_static_allcpu.py',optional='optional_delivery_flat_static_allcpu.py',optional_submit='optional_fine_dispatch_flat_static_allcpu.py',watchdog='scheduler_watchdog_flat_static_allcpu.py',resources='resources_idle_cpu.py',total_cpu=24,total_gpu_shards=2,resource_guard='resource_guard_idle_cpu.py',cpu_authority='Use24 when no other compute jobs, otherwise18')
    write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
    entry['pid']=launch(entry['coordinator'],'coordinator.log');write(HERE/'ACTIVE_ENTRYPOINT.json',entry)
    pids=dict(coordinator=entry['pid'],guard=launch('resource_guard_idle_cpu.py','resource_guard_idle_cpu.log'),heartbeat=launch('heartbeat_released_allcpu.py','heartbeat.log'))
    deadline=time.monotonic()+55
    while time.monotonic()<deadline:
        if read(HERE/'COORDINATOR_PID.json')['pid']==entry['pid']:break
        time.sleep(.5)
    else:raise RuntimeError('new coordinator did not acquire lock')
    pids['watchdog']=launch(entry['watchdog'],'scheduler_watchdog.log')
    from resources_idle_cpu import snapshot
    write(HERE/'IDLE_CPU_AUTHORIZATION_DEPLOYED.json',dict(utc=utc(),pids=pids,resources=snapshot(),running_workers_untouched=True,worker_signals=0,gpu_shards=2,previous_policy_preserved=True))
    for name in ('ITERATION_LOG.md','HANDOFF.md'):
        with (HERE/name).open('a') as out:out.write('\n\n'+utc()+': Latest user CPU authorization: no other Slurm compute jobs; idle total24 CPU=23 worker slots+1 local, GPU remains2 shares. Original sealed18/2 policy preserved. IDLE_CPU_AUTHORIZATION_CODE.receipt.json and new *_allcpu orchestration/static imports implement the new policy; running rollout workers received no signal. resources_idle_cpu.py detects other users queued/running Slurm jobs and restores18 CPU; resource_guard_idle_cpu.py requests complete-episode yielding, never hard-cancels running rollouts. ACTIVE_ENTRYPOINT.json contains current entrypoints.\n')
    print(json.dumps(pids))
if __name__=='__main__':main()
