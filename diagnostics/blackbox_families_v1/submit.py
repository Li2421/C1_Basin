from common import *
import subprocess,shlex
phase=sys.argv[1];prereg();assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists()
q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True)
used=sum(int(x.split('|')[3]) for x in q.splitlines() if x.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING','PENDING'])
slots=min(18-used,len(read(HERE/f'{phase}_MISSING_CASES.json')),int(sys.argv[2]) if len(sys.argv)>2 else 18);assert slots>0,(q,used)
cmd=f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"worker.py"))} {shlex.quote(phase)} $SLURM_ARRAY_TASK_ID'
args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=4G','--time','24:00:00','--array',f'0-{slots-1}%{slots}','-J','r38_'+phase,'-o',str(HERE/(phase+'_%A_%a.log')),'--wrap',cmd]
seal(f'{phase}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/f'{phase}_PREFLIGHT.receipt.json',HERE/'worker.py',HERE/'controllers.py'],queue=q,CPU_slots=slots,before_submission=True)
job=subprocess.check_output(args,text=True).strip();write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),job=job,slots=slots,args=args,queue=q));print(job,slots)
