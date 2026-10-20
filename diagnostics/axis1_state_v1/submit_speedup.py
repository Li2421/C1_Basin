from common import *
import subprocess,shlex
from quota_watch import queue,allowance
assert (HERE/'SPEEDUP_PREFLIGHT.receipt.json').exists();rows=queue();slots,other,pending=allowance(rows,'NO_EXISTING_HELPER_ARRAY');assert slots>0,rows
cmd=f'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"run_speedup.py"))} main $SLURM_ARRAY_TASK_ID 18'
args=['sbatch','--parsable','-p','cpu','-J','r32_queue','--cpus-per-task=1','--mem=4G','--time','24:00:00','--array',f'0-17%{slots}','-o',str(HERE/'speedup_%A_%a.log'),'--wrap',cmd]
seal('SPEEDUP_SUBMISSION_PREFLIGHT.receipt.json',[HERE/'SPEEDUP_PREFLIGHT.receipt.json',HERE/'shared_queue.py',HERE/'run_speedup.py'],before_submission=True,queue=rows,initial_slots=slots,target_total_CPU=18)
job=subprocess.check_output(args,text=True).strip();write(HERE/'SPEEDUP_SUBMISSION.json',dict(utc=utc(),job=job,args=args,initial_slots=slots,queue=rows));print(job,slots)
