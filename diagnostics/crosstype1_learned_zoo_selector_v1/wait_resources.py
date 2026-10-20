"""Wait without reserving compute; submit the mandatory replay exactly once."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import time
import fcntl
import os
from resource_guard import queue

HERE=Path(__file__).resolve().parent

def main():
    lock=(HERE/'RESOURCE_WAIT.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'RESOURCE_WAIT_PID.json').write_text(json.dumps(dict(pid=os.getpid()))+'\n')
    while not (HERE/'REPRODUCTION_SUBMISSION.json').exists():
        q=queue()
        with (HERE/'RESOURCE_WAIT.jsonl').open('a') as stream:
            stream.write(json.dumps(q)+'\n')
        if q['cpu_shards']<=16 and q['gpu_shards']<=2:
            result=subprocess.run([sys.executable,'-B',str(HERE/'submit_reproduction.py')])
            if result.returncode==0:return
        time.sleep(30)

if __name__=='__main__':main()
