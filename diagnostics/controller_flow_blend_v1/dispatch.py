"""Slurm split pools: 18+6 CPU or 4+4 GPU; extra pool has a hard cutoff."""
from common import *
import subprocess,math

def submit(name,command,*,array=None,gpu=False,extra=False,hours=2,dependency=None,dependency_kind='afterok'):
    p=read(HERE/'RESOURCE_AUTHORIZATION.json');now=datetime.now(timezone.utc);cutoff=datetime.fromisoformat(p['extra_capacity_cutoff_utc'])
    seconds=hours*3600
    if extra:
        seconds=min(seconds,int((cutoff-now).total_seconds())-120)
        assert seconds>=60,'extra allocation window closed'
    mins=max(1,int(seconds//60));h,m=divmod(mins,60)
    args=['sbatch','--parsable','-J',('r5x_' if extra else 'r5_')+name,'-p','gpu' if gpu else 'cpu','--cpus-per-task=1','--mem=5G' if gpu else '--mem=4G',f'--time={h:02}:{m:02}:00','-o',str(HERE/(name+('_%A_%a.log' if array else '_%j.log')))]
    if array:args+=['--array',array]
    if gpu:args+=['--gres=shard:1']
    assert dependency_kind in ('afterok','afterany')
    if dependency:args+=['--dependency',f'{dependency_kind}:{dependency}']
    args+=['--wrap',command]
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
    record=dict(job_id=job,name=('r5x_' if extra else 'r5_')+name,command=command,array=array,gpu=gpu,extra=extra,walltime_minutes=mins,submitted_utc=utc(),dependency=dependency,dependency_kind=dependency_kind)
    log=HERE/'SLURM_SUBMISSIONS.json';old=read(log) if log.exists() else []
    write(log,old+[record]);return record

def pilot():
    assert not (HERE/'stage_1/PILOT_TIMING.json').exists()
    command=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PYTHON} {HERE}/rollout.py pilot --index $SLURM_ARRAY_TASK_ID --workers 24'
    base=submit('pilot',command,array='0-17%18',hours=1)
    extra=submit('pilotx',command+' --extra',array='18-23%6',hours=1,extra=True)
    finish=submit('timing',f'{PYTHON} {HERE}/rollout.py timing --workers 24',dependency=base['job_id']+':'+extra['job_id'],hours=1)
    write(HERE/'stage_1/PILOT_DISPATCH.json',dict(base=base,extra=extra,timing=finish))
    print(json.dumps(dict(cpu_concurrency=24,extra_cutoff=read(HERE/'RESOURCE_AUTHORIZATION.json')['extra_capacity_cutoff_local'],jobs=[base['job_id'],extra['job_id'],finish['job_id']])))

def stage1():
    timing=read(HERE/'stage_1/PILOT_TIMING.json')
    command=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu {PYTHON} {HERE}/rollout.py worker --index $SLURM_ARRAY_TASK_ID --workers 24'
    hours=max(1,math.ceil(timing['stage1_wall_hours_24']*2+1))
    base=submit('s1raw',command,array='0-17%18',hours=hours)
    extra=submit('s1rawx',command+' --extra',array='18-23%6',hours=hours,extra=True)
    finish=submit('s1final',f'bash {HERE}/finalize_stage1.sh',dependency=base['job_id']+':'+extra['job_id'],hours=1)
    write(HERE/'stage_1/FULL_DISPATCH.json',dict(base=base,extra=extra,finalize=finish,actual_timing_sha256=sha(HERE/'stage_1/PILOT_TIMING.json')))
    print(json.dumps(dict(cpu_concurrency=24,jobs=[base['job_id'],extra['job_id'],finish['job_id']],timing=timing)))

if __name__=='__main__':globals()[sys.argv[1]]()
