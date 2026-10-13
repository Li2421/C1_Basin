"""Explicit frozen stage gates; no automatic growth or data-driven redesign."""
from common import *
import subprocess,math
from dispatch import submit

def command(script,args='',gpu=False):
    return f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS={"cuda" if gpu else "cpu"} {PYTHON} {HERE}/{script} {args}'
def call(script,args=''):
    subprocess.run([PYTHON,str(HERE/script)]+args.split(),cwd=ROOT,check=True)
def preflight(folder):
    with (folder/'PREFLIGHT_STDOUT.json').open('w') as out:
        subprocess.run([PYTHON,'-m','shared_rollout_db.plan','--manifest',str(folder/'PLANNED_ROLLOUTS.json'),'--output',str(folder/'CACHE_PREFLIGHT.json')],cwd=ROOT,stdout=out,check=True)

def stage1_done():
    gate=read(HERE/'STAGE_1_GATE.json')
    if not gate['passed']:
        call('render_report.py');call('final_audit.py');return
    d=HERE/'stage_2'
    if (d/'DISPATCH.json').exists():print('already dispatched');return
    call('stage2_design.py','create');preflight(d);call('stage2_design.py','inventory')
    call('probe.py','create');preflight(d/'probes')
    timing=read(HERE/'stage_1/PILOT_TIMING.json');post=read(HERE/'stage_1/POSTFLIGHT.json')
    # Full-pilot elapsed costs improve the forecast without changing allocation.
    records=[read(f) for f in (HERE/'stage_1/raw').glob('*.json')];secs=sum(r['elapsed_seconds'] for r in records)/len(records)
    missing=read(d/'CACHE_INVENTORY.json')['missing'];rawhours=max(2,math.ceil(missing*secs/(24*3600)*2+2))
    write(d/'COST_GATE.json',dict(utc=utc(),new_closed_loop_rollouts=missing,new_short_probes=832,total_new=missing+832,cap=65000,
        mean_seconds_full_pilot=secs,predicted_cpu_hours=missing*secs/3600,predicted_wall_hours_24=missing*secs/86400,predicted_wall_hours_18=missing*secs/64800,
        chosen_train_states=96,allocation='within count cap; no outcome-driven reduction',raw_job_walltime_hours=rawhours,stage1_unknown=post['unknown']))
    jobs=[]
    # Static GPU jobs start first; rollouts fill remaining CPUs and expand as
    # feature jobs complete. Node Slurm enforces the total 24-CPU capacity.
    jobs.append(submit('s2feat',command('stage2_features.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,array='0-24:2%4',hours=3))
    jobs.append(submit('s2featx',command('stage2_features.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,extra=True,array='1-25:2%4',hours=3))
    featdep=jobs[-2]['job_id']+':'+jobs[-1]['job_id']
    jobs.append(submit('s2raw',command('stage2_run.py','--index $SLURM_ARRAY_TASK_ID --workers 24'),array='0-17%18',hours=rawhours))
    jobs.append(submit('s2rawx',command('stage2_run.py','--index $SLURM_ARRAY_TASK_ID --workers 24 --extra'),array='18-23%6',hours=rawhours,extra=True))
    rawdep=jobs[-2]['job_id']+':'+jobs[-1]['job_id']
    jobs.append(submit('s2labels',f'bash {HERE}/finalize_stage2.sh',dependency=rawdep,hours=2));labeldep=jobs[-1]['job_id']
    jobs.append(submit('s2probe',command('probe.py','worker --index $SLURM_ARRAY_TASK_ID --workers 24'),array='0-17%18',dependency=rawdep,hours=2))
    jobs.append(submit('s2probex',command('probe.py','worker --index $SLURM_ARRAY_TASK_ID --workers 24'),array='18-23%6',dependency=rawdep,hours=2,extra=True))
    probedep=jobs[-2]['job_id']+':'+jobs[-1]['job_id']+':'+labeldep
    jobs.append(submit('s2probemerge',command('probe.py','merge'),dependency=probedep,hours=1));probedone=jobs[-1]['job_id']
    fitdep=featdep+':'+labeldep+':'+probedone
    # First freeze five common SPARSE models; remaining SPARSE folds only infer.
    jobs.append(submit('s2initfit',command('train.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,array='5-8%4',dependency=fitdep,hours=3))
    jobs.append(submit('s2initfitx',command('train.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,extra=True,array='9%1',dependency=fitdep,hours=3))
    shareddep=jobs[-2]['job_id']+':'+jobs[-1]['job_id']
    jobs.append(submit('s2denseearlyx',command('train.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,extra=True,array='0-2%3',dependency=fitdep,hours=3))
    earlydep=jobs[-1]['job_id']
    # Unique DENSE fits plus reused SPARSE prediction tasks. Balanced partition
    # is task-index based, never outcome based. Eight GPU shards when available.
    active=[i for i in range(60) if i not in list(range(5,10))+[0,1,2]];base=active[::2];extra=active[1::2]
    jobs.append(submit('s2fit',command('train.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,array=','.join(map(str,base))+'%4',dependency=shareddep,hours=3))
    jobs.append(submit('s2fitx',command('train.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,extra=True,array=','.join(map(str,extra))+'%4',dependency=shareddep+':'+earlydep,hours=3))
    enddep=jobs[-2]['job_id']+':'+jobs[-1]['job_id']+':'+earlydep
    jobs.append(submit('s2analysis',command('analyze.py'),dependency=enddep,hours=1))
    jobs.append(submit('s2report',command('render_report.py'),dependency=jobs[-1]['job_id'],hours=1))
    jobs.append(submit('s2audit',command('final_audit.py'),dependency=jobs[-1]['job_id'],hours=2))
    write(d/'DISPATCH.json',dict(gate_sha256=sha(HERE/'STAGE_1_GATE.json'),jobs=jobs,new_closed_loop_cap=65000,maximum_gpu_shards=8,
        extra_cutoff_local=read(HERE/'RESOURCE_AUTHORIZATION.json')['extra_capacity_cutoff_local'],source_fit_reuse='SPARSE identical-source models optimized once, SHA-verified across all folds'))
    call('render_report.py');print(json.dumps(dict(stage=2,dispatched=True,jobs=[r['job_id'] for r in jobs])))

if __name__=='__main__':globals()[sys.argv[1]]()
