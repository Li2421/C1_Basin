"""Resume frozen work after capacity release, never re-run a published outcome.

Only operational scheduling changes. No protocol, label, feature or fit selection
changes. Completed fits and raw records are reused by the original entry points.
"""
from common import *
from dispatch import submit
from pipeline import command


def schedule(wait_for):
    d = HERE/'stage_2'
    dest = HERE/'RESOURCE_RECOVERY_DISPATCH.json'
    if dest.exists():
        return read(dest)
    if (HERE/'FINAL_CONSISTENCY_AUDIT.json').exists() and read(HERE/'FINAL_CONSISTENCY_AUDIT.json')['status']=='PASS':
        result=dict(utc=utc(),status='ALREADY COMPLETE; NO ALLOCATION',jobs=[])
        write(dest,result)
        return result
    # This study has already passed its pilot and frozen Stage 2. Do not create
    # new states, controller UIDs, cases or a different experimental design.
    assert read(HERE/'STAGE_1_GATE.json')['passed'] and (d/'DISPATCH.json').exists()
    assert datetime.now(timezone.utc)>=datetime.fromisoformat(read(HERE/'RESOURCE_AUTHORIZATION.json')['extra_capacity_cutoff_utc'])
    jobs=[]
    def add(name,cmd,**kw):
        result=submit('resume_'+name,cmd,**kw)
        jobs.append(result)
        return result['job_id']
    # Wait until retained old workers finish before repartitioning cases. This
    # prevents two workers from executing the same still-unpublished request.
    prerequisite=None
    if wait_for:
        prerequisite=':'.join(wait_for)
    feat=add('features',command('stage2_features.py','$SLURM_ARRAY_TASK_ID',True),
             gpu=True,array='0-25%4',hours=3,dependency=prerequisite,dependency_kind='afterany')
    # No overlaps between 18-way CPU stages and four-way GPU stages after the
    # cutoff. A GPU shard job also consumes a CPU and counts toward the CPU cap.
    raw=add('raw',command('stage2_run.py','--index $SLURM_ARRAY_TASK_ID --workers 18'),
            array='0-17%18',hours=30,dependency=feat)
    labels=add('labels',f'bash {HERE}/finalize_stage2.sh',hours=2,dependency=raw)
    probes=add('probes',command('probe.py','worker --index $SLURM_ARRAY_TASK_ID --workers 18'),
               array='0-17%18',hours=3,dependency=labels)
    merge=add('probemerge',command('probe.py','merge'),hours=1,dependency=probes)
    init=add('initfits',command('train.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,
             array='5-9%4',hours=3,dependency=merge)
    rest=[i for i in range(60) if i not in range(5,10)]
    fits=add('fits',command('train.py','$SLURM_ARRAY_TASK_ID',True),gpu=True,
             array=','.join(map(str,rest))+'%4',hours=3,dependency=init)
    analysis=add('analysis',command('analyze.py'),hours=1,dependency=fits)
    report=add('report',command('render_report.py'),hours=1,dependency=analysis)
    add('audit',command('final_audit.py'),hours=2,dependency=report)
    result=dict(utc=utc(),status='FROZEN WORK RESUMED',jobs=jobs,
                retained_jobs_waited_for=wait_for,cpu_total_cap=18,gpu_shard_cap=4,
                numerical_failures_retried=False,published_raw_reused=True,
                changed_experimental_protocol=False)
    write(dest,result)
    return result
