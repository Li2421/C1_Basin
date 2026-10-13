"""Source-defined conditional LOCO branch, no rollout or DB mutations."""
from util import *
import subprocess
from submit import submit
def run(script,args=None):
    env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([PY,str(HERE/script)]+(args or []),cwd=ROOT,env=env,check=True)
def after_blend():
    assert all((HERE/'fits'/task_name(t)/'complete.json').exists() for t in read(HERE/'FIT_TASKS_BLEND.json'))
    run('verify_statistics.py');run('evaluate.py',['blend'])
    elig=read(HERE/'STAGE2_ELIGIBILITY.json')
    if elig['regime_logistic_real_execute']:run('logistic.py',['real'])
    tasks=read(HERE/'FIT_TASKS_REAL.json')
    if tasks:
        job=submit('real_fits','fit_real.py',list(range(len(tasks))))
        submit_finalize(job)
    else:finish()
def submit_finalize(job):
    cmd=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} {HERE}/pipeline.py finish'
    args=['sbatch','--parsable','-J','r6_final','-p','cpu','--cpus-per-task=1','--mem=5G','--time=02:00:00','--dependency','afterok:'+job,'-o',str(HERE/'final_%j.log'),'--wrap',cmd]
    new=subprocess.check_output(args,text=True).strip().split(';')[0];write(HERE/'FINAL_DISPATCH.json',dict(job=new,dependency=job,command=cmd,utc=utc()))
def finish():
    run('evaluate.py',['real']);run('plots.py');run('report.py');run('audit.py')
if __name__=='__main__':globals()[sys.argv[1]]()
