"""Switch only local final-delivery managers;keep all science workers running."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
import subprocess,signal,time,fcntl


def command(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode().strip() if p.exists() else ''


def start(script):
    with (HERE/(Path(script).stem+'.log')).open('a') as out:return subprocess.Popen([sys.executable,'-B',str(HERE/script)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    proof='FINAL_DELIVERY_CLARITY_20261022.receipt.json';verify_receipt(proof)
    entry=read(HERE/'ACTIVE_ENTRYPOINT.json');mapping=read(HERE/'FINAL_DELIVERY_CLARITY_MAP_20261022.json')['copies']
    assert read(HERE/'STATUS.json')['phase']=='paper_dynamics_optional'
    assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
    root=read(HERE/'COORDINATOR_PID.json')['pid'];child=read(HERE/'CORRECTION_CONTINUATION_PID_20261021.json')['pid'];watch=read(HERE/'SCHEDULER_WATCHDOG_PID.json')['pid']
    optional=[int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and command(int(p.name))==sys.executable+' -B '+str(HERE/entry['optional'])]
    assert len(optional)==1
    old=[(watch,entry['watchdog']),(root,entry['coordinator']),(child,entry['downstream']),(optional[0],entry['optional'])]
    for pid,name in old:assert command(pid)==sys.executable+' -B '+str(HERE/name)
    prefix='FINAL_DELIVERY_CLARITY_DEPLOYMENT_20261022';assert not (HERE/(prefix+'.receipt.json')).exists()
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Do not cut across an active merger/planner/report action. Physics
        # workers have separate PIDs and are never signalled here.
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                fields=p.joinpath('stat').read_text().split(') ',1)[1].split()
                if int(fields[1])==optional[0]:
                    args=command(int(p.name))
                    assert not (args.startswith(sys.executable+' -B '+str(HERE))),'Optional child action active;retry after its boundary'
            except (FileNotFoundError,ProcessLookupError):continue
        write(HERE/(prefix+'_PRE.json'),dict(utc=utc(),entry=entry,local_managers=old,science_workers=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%C'],text=True),scientific_worker_signals=0))
        for pid,name in old:os.kill(pid,signal.SIGTERM)
        for _ in range(60):
            if not any(command(pid)==sys.executable+' -B '+str(HERE/name) for pid,name in old):break
            time.sleep(.1)
        assert not any(command(pid)==sys.executable+' -B '+str(HERE/name) for pid,name in old)
        new=dict(entry)
        for key in ('downstream','optional','final_reports'):new[key]=mapping[entry[key]]
        new.update(utc=utc(),downstream_receipt=proof,optional_receipt=proof,delivery_clarity_receipt=proof)
        write(HERE/'ACTIVE_ENTRYPOINT.json',new)
        proc=start(new['coordinator']);new['pid']=proc.pid;write(HERE/'ACTIVE_ENTRYPOINT.json',new)
    for _ in range(100):
        assert proc.poll() is None
        if read(HERE/'COORDINATOR_PID.json')['pid']==proc.pid:break
        time.sleep(.1)
    watcher=start(new['watchdog'])
    deadline=time.monotonic()+120
    while True:
        assert proc.poll() is None and watcher.poll() is None
        assert not (HERE/'COORDINATOR_ACTION_ERROR.json').exists()
        current=read(HERE/'CORRECTION_CONTINUATION_PID_20261021.json')['pid']
        if command(current)==sys.executable+' -B '+str(HERE/new['downstream']) and read(HERE/'STATUS.json')['phase']=='paper_dynamics_optional':break
        assert time.monotonic()<deadline;time.sleep(1)
    write(HERE/(prefix+'.json'),dict(utc=utc(),passed=True,coordinator_pid=proc.pid,continuation_pid=current,watchdog_pid=watcher.pid,scientific_worker_signals=0,active_entry=read(HERE/'ACTIVE_ENTRYPOINT.json'),physics_and_statistics_unchanged=True))
    seal(prefix+'.receipt.json',[Path(__file__),HERE/(prefix+'_PRE.json'),HERE/(prefix+'.json'),HERE/proof],passed=True,scientific_worker_signals=0)
    with (HERE/'ITERATION_LOG.md').open('a') as out:out.write('\n'+utc()+'：仅最终交付文字与第二阶段失败归因澄清：资源共享仅本轮自有核，二阶段低成功类别如含碰撞/无解/违反不能直接称自身停住或死锁解决，分类和成功标准不变。只切本地wrapper，无科学worker信号，全部已有raw/journal继续保留。见'+prefix+'.receipt.json。\n')
    print('PASS:final reporting clarification active;science workers untouched',flush=True)


if __name__=='__main__':main()
