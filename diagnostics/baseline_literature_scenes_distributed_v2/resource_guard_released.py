"""Same release policy, with retry if Slurm races an already-finished job."""
from experiment import *
from resource_guard_released_core import enforce
import time
import traceback

def main():
    policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTED.json');deadline=datetime.fromisoformat(policy['release_at_utc'])
    write(HERE/'RESOURCE_GUARD_PID.json',dict(pid=os.getpid(),utc=utc(),policy=policy,entrypoint='resource_guard_released.py'))
    while True:
        seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
        if seconds<=0:break
        time.sleep(min(30.,seconds))
    while True:
        try:enforce()
        except Exception as exc:
            write(HERE/'RESOURCE_GUARD_RETRY.json',dict(utc=utc(),error=str(exc),traceback=traceback.format_exc(),retry_seconds=1))
            time.sleep(1);continue
        time.sleep(10)

if __name__=='__main__':main()
