from core import *
import subprocess,shlex

def main(phase):
    prereg();assert read(HERE/f'{phase}_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
    if phase=='main':gate()
    else:assert read(HERE/'STATE_DEDUP_AUDIT_V2.json')['passed']
    q=subprocess.check_output(['squeue','-h','-r','-t','RUNNING','-u',str(os.getuid()),'-o','%C|%j|%b'],text=True);other=sum(int(x.split('|')[0]) for x in q.splitlines());slots=min(17,18-other-1);assert slots>0,(other,'No CPU slots')
    slots=min(slots,len(read(HERE/f'{phase}_MISSING_CASES.json')));assert slots>0
    cmd=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu {shlex.quote(PY)} -B {shlex.quote(str(HERE/"run.py"))} {phase} $SLURM_ARRAY_TASK_ID {slots}'
    args=['sbatch','--parsable','-J','r10_'+phase,'-p','cpu','--cpus-per-task=1','--mem=4G','--time','08:00:00' if phase=='main' else '00:30:00','--array',f'0-{slots-1}%{slots}','-o',str(HERE/(phase+'_%A_%a.log')),'--wrap',cmd]
    job=subprocess.check_output(args,text=True).strip();x=read(HERE/'SLURM_SUBMISSIONS.json') if (HERE/'SLURM_SUBMISSIONS.json').exists() else [];r=dict(utc=utc(),phase=phase,job=job,workers=slots,command=args,other_cpu=other,shared_cpu_cap=18,shared_gpu_cap=3,prior_queue=q);x.append(r);write(HERE/'SLURM_SUBMISSIONS.json',x);print(json.dumps(r))
if __name__=='__main__':main(sys.argv[1])
