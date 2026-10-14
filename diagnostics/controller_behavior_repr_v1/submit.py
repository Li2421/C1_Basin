"""User's shared cap: subtract all other active/pending jobs, never cancel."""
from core import *
import subprocess,shlex
def submit(indices,mode='fit',label='fits'):
    gate();assert read(HERE/'STAGE1.json')['status']=='COMPLETE'
    q=subprocess.check_output(['squeue','-h','-r','-o','%i|%j|%T|%C|%b'],text=True);cpus=0;shards=0
    for line in q.splitlines():
        jid,job,state,c,gres=line.split('|');cpus+=int(c)
        if 'shard:' in gres:shards+=int(gres.rsplit(':',1)[1])
        elif 'gpu:' in gres:shards+=3
    slots=min(3,3-shards,18-cpus-1);assert slots>0,(q,'No shared GPU slots')
    args=['sbatch','--parsable','-J','r7_'+label,'-p','gpu','--cpus-per-task=1','--gres=shard:1','--mem=5G','--time=01:00:00','--array',','.join(map(str,indices))+f'%{slots}','-o',str(HERE/(label+'_%A_%a.log'))]
    cmd=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda {shlex.quote(PY)} -B {shlex.quote(str(HERE/"fit.py"))} {mode} $SLURM_ARRAY_TASK_ID';args+=['--wrap',cmd]
    job=subprocess.check_output(args,text=True).strip();p=HERE/'SLURM_SUBMISSIONS.json';old=read(p) if p.exists() else [];rec=dict(utc=utc(),job=job,slots=slots,indices=indices,mode=mode,prior_queue=q,other_cpu=cpus,other_gpu_shards=shards,shared_caps=dict(cpu=18,gpu_shards=3),command=args);write(p,old+[rec]);print(json.dumps(rec));return job
if __name__=='__main__':
    from data import name
    mode=sys.argv[1] if len(sys.argv)>1 else 'fit';ts=read(HERE/('FIT_TASKS.json' if mode=='fit' else 'TASKS.json'));ids=[i for i,t in enumerate(ts) if not t['reuse'] and not (HERE/'fits'/name(t)/'complete.json').exists()];submit(ids,mode,'fits' if mode=='fit' else 'predict')
