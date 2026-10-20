from common import *
from submit_stage import completed,merge,queue
from submit_cpu40 import submit
import time,subprocess,shlex
stage='expert_gate_v3'
def add_pool(number):
 return submit(stage,6,script='expert_worker.py',pool_name=f'{stage}_extra_{number}')
while not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():time.sleep(5)
write(HERE/'EXPERT_RESOURCE_WAIT.json',dict(utc=utc(),queue=queue(),status='waiting under global18CPU2GPU cap'))
job=None
while job is None:
 job=submit(stage,6,script='expert_worker.py')
 if job is None:time.sleep(5)
jobs=[job];print('expert job',job,flush=True);extra=1;count=len(read(HERE/f'{stage}_MISSING_CASES.json'))
while any(not completed(job) for job in jobs):
 cursor=HERE/stage/'cursor';claimed=int(cursor.read_text() or 0) if cursor.exists() else 0
 if claimed<count:
  added=add_pool(extra)
  if added:jobs.append(added);extra+=1;print('additional job',added,flush=True)
 time.sleep(5)
merge(stage)
write(HERE/'EXPERT_EXECUTION_COMPLETE.json',dict(utc=utc(),jobs=jobs,stage=stage))
print('expert complete',jobs,flush=True)
