"""Keep the required status current during long synchronous analysis actions."""
from experiment import *
from atomic_write import write
from resources import snapshot
import fcntl
import time
import subprocess

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def tick():
    if (HERE/'V2_COMPLETE.json').exists():return False
    pid=read(HERE/'COORDINATOR_PID.json')['pid']
    current=read(HERE/'STATUS.json');running=alive(pid)
    write(HERE/'HEARTBEAT.json',dict(utc=utc(),coordinator_pid=pid,coordinator_alive=running,last_status_utc=current['utc'],phase=current['phase'],science_or_submission_changes=False))
    elapsed=time.time()-(HERE/'STATUS.json').stat().st_mtime
    if elapsed<900:return True
    stages=[]
    for path in HERE.glob('*/GROUP_INDEX.json'):
        index=read(path);done=[r for r in index if (path.parent/'claims'/r['group']/'COMPLETE.json').exists()]
        stages.append(dict(stage=path.parent.name,queue_groups=len(index),finished_groups=len(done),completed_requests_in_finished_groups=sum(len(r['rollout_uids']) for r in done),partial_groups_excluded_from_fast_count=True,queue_complete=len(done)==len(index)))
    current.update(utc=utc(),stages=stages,resources=snapshot(),heartbeat_during_long_action=True,coordinator_alive=running)
    current['recent_action_logs']=[str(p) for p in sorted((HERE/'coordinator_actions').glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:3]]
    write(HERE/'STATUS.json',current)
    text=f"{current['utc']}。阶段：{current['phase']}。长操作期间由独立 heartbeat 刷新；协调器 PID {pid} 存活={running}。\n\nv1 已冻结，本轮只读。资源上限/释放时刻见 RESOURCE_SCHEDULE.json；全部用户作业计入。当前 Slurm {current['resources']['scheduled_cpus']} CPU、{current['resources']['gpu_shards']} GPU份额。\n\n"
    text+='\n\n'.join(f"- {r['stage']}：小批次 {r['finished_groups']}/{r['queue_groups']}，完整批次请求 {r['completed_requests_in_finished_groups']}（不含进行中部分）。" for r in stages)
    temp=HERE/'STATUS.heartbeat.tmp';temp.write_text(text+'\n');temp.replace(HERE/'STATUS.md')
    return True

def main():
    with (HERE/'HEARTBEAT.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(HERE/'HEARTBEAT_PID.json',dict(utc=utc(),pid=os.getpid()))
        while True:
            try:
                if not tick():return
            except Exception as exc:
                write(HERE/'HEARTBEAT_ERROR.json',dict(utc=utc(),error=repr(exc)))
            time.sleep(60)

if __name__=='__main__':main()
