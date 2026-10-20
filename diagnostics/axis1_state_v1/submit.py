from common import *
import subprocess,shlex
phase=sys.argv[1];prereg();assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists()
if phase=='main':assert read(HERE/'CONSISTENCY_GATE.json')['passed'];assert (HERE/'CHOICES_FROZEN.receipt.json').exists()
queue=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%T|%C|%b|%j'],text=True)
# Include pending reservations in quota, so simultaneous starts cannot oversubscribe.
occupied=sum(int(x.split('|')[2]) for x in queue.splitlines());slots=min(7,18-occupied-1);assert slots>0,queue
rows=read(HERE/f'{phase}_MISSING_CASES.json');slots=min(slots,len(rows));assert slots>0
py='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
cmd=f'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(py)} -B {shlex.quote(str(HERE/"run.py"))} {phase} $SLURM_ARRAY_TASK_ID {slots}'
args=['sbatch','--parsable','-p','cpu','-J','r32_'+phase,'--cpus-per-task=1','--mem=4G','--time','24:00:00','--array',f'0-{slots-1}%{slots}','-o',str(HERE/(phase+'_%A_%a.log')),'--wrap',cmd]
seal(f'{phase}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/f'{phase}_PREFLIGHT.receipt.json',HERE/'run.py',HERE/'arms.py'],queue=queue,slots=slots,CPU_cap=18,GPU_cap=2)
job=subprocess.check_output(args,text=True).strip();write(HERE/f'{phase}_SUBMISSION.json',dict(job=job,slots=slots,utc=utc(),args=args,prior_queue=queue));print(job,slots)
