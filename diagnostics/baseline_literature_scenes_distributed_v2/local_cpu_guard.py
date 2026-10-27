"""Keep the one local CPU share off every Slurm-allocated physical CPU."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import subprocess
import re
import time
import fcntl


def allocated_cpus():
    jobs=subprocess.check_output(['squeue','-h','-t','RUNNING,CONFIGURING,COMPLETING','-o','%i'],text=True).split()
    occupied=set()
    for job in jobs:
        result=subprocess.run(['scontrol','show','job','-dd',job],capture_output=True,text=True)
        if result.returncode:continue  # Job can finish between the two reads.
        for value in re.findall(r'Nodes=desktop\s+CPU_IDs=(\S+)',result.stdout):
            for part in value.split(','):
                if '-' in part:
                    a,b=map(int,part.split('-'));occupied.update(range(a,b+1))
                elif part.isdigit():occupied.add(int(part))
    return occupied,jobs


def local_processes():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=proc.joinpath('cmdline').read_bytes().split(b'\0')
            if len(args)<3 or not (args[0].endswith(b'/.venv-c1/bin/python') or args[0].endswith(b'/.venv-c1-plots/bin/python')) or args[1]!=b'-B':continue
            script=Path(os.fsdecode(args[2]));script=script if script.is_absolute() else ROOT/script
            if script.parent!=HERE:continue
            # Inspect only whether the scheduler key exists; never log env data.
            env=proc.joinpath('environ').read_bytes().split(b'\0')
            if any(v.startswith(b'SLURM_JOB_ID=') for v in env):continue
            yield proc,script
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue


def tick():
    occupied,jobs=allocated_cpus();free=set(range(os.cpu_count()))-occupied
    current=os.sched_getaffinity(0)
    if not free:
        write(HERE/'LOCAL_CPU_GUARD_STATUS.json',dict(utc=utc(),free_physical_cpus=[],occupied=sorted(occupied),error='No unallocated physical CPU; inspect before running heavy local work'))
        return
    chosen=next(iter(current)) if len(current)==1 and next(iter(current)) in free else max(free)
    processes=[];changes=[]
    for proc,script in local_processes():
        try:
            count=0
            for tid in proc.joinpath('task').iterdir():
                try:
                    previous=os.sched_getaffinity(int(tid.name))
                    if previous!={chosen}:os.sched_setaffinity(int(tid.name),{chosen});count+=1
                except ProcessLookupError:continue
            processes.append(dict(pid=int(proc.name),script=script.name))
            if count:changes.append(dict(pid=int(proc.name),threads_rebound=count))
        except (FileNotFoundError,ProcessLookupError):continue
    write(HERE/'LOCAL_CPU_GUARD_STATUS.json',dict(utc=utc(),local_cpu=chosen,slurm_allocated_cpus=sorted(occupied),free_physical_cpus=sorted(free),local_processes=processes,rebound=changes,poll_seconds=5,local_cpu_shares=1,total_user_cpu_cap=18,no_worker_signals=True))


def main():
    with (HERE/'LOCAL_CPU_GUARD.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('LOCAL_CPU_GUARD_CODE.receipt.json')
        write(HERE/'LOCAL_CPU_GUARD_PID.json',dict(utc=utc(),pid=os.getpid()))
        while not (HERE/'V2_COMPLETE.json').exists():
            try:tick()
            except Exception as exc:write(HERE/'LOCAL_CPU_GUARD_ERROR.json',dict(utc=utc(),error=repr(exc)))
            time.sleep(5)


if __name__=='__main__':main()
