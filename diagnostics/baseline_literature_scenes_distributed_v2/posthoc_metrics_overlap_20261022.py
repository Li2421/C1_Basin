"""Run independent saved-evidence metrics alongside authorized stage-two rollouts."""
from experiment import *
from registry import verify_receipt
from atomic_write import write
from resources_corrections_release_20261023_quota18_now_20261022 import snapshot,require_capacity
from infrastructure_recovery_correction_20261021 import classify
import subprocess,time,fcntl

READY='POSTHOC_METRICS_OVERLAP_READY_20261022.receipt.json'
STATE=HERE/'POSTHOC_METRICS_OVERLAP_STATE_20261022.json'
TASKS={
'original':('derived_metrics_original_supplement_20261022.py','MAIN_ADDITIONAL_METRICS.receipt.json',16),
'revised':('derived_metrics_supplement_20261022.py','MAIN_ADDITIONAL_METRICS_REVISED_20261021.receipt.json',16),
'door':('door_controller_gain_components_supplement_20261022.py','DOOR_CONTROLLER_GAIN_COMPONENTS_20261021.receipt.json',4)}


def command(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode().strip() if p.exists() else ''


def retained_alive(row):
    pid=row.get('retained_pid')
    if not pid:return False
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return False
    stat=p.read_text().split(') ',1)[1].split()
    return stat[0]!='Z' and stat[19]==row['retained_start_ticks'] and command(pid)==row['retained_command']


def ensure():
    verify_receipt(READY);verify_receipt('MIGRATION_NUMBERS.receipt.json')
    with (HERE/'POSTHOC_METRICS_OVERLAP.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(STATE);live={r['job_id'] for r in snapshot()['jobs']}
        for tag,(script,proof,initial_memory) in TASKS.items():
            row=state['tasks'].setdefault(tag,{})
            if (HERE/proof).exists():verify_receipt(proof);row['complete']=True;continue
            if retained_alive(row):continue
            if row.get('job') in live:continue
            if row.get('needs_review'):continue
            if row.get('job') and not row.get('reviewed'):
                review=classify('analysis_posthoc_'+tag,row['job']);row['reviewed']=True
                write(HERE/'posthoc_metrics_reviews_20261022'/f"{row['job']}.json",review)
                if not review['infrastructure']:
                    row['needs_review']=True
                    write(HERE/f'POSTHOC_METRICS_{tag}_REVIEW_20261022.json',dict(utc=utc(),tag=tag,review=review,other_scientific_work_continues=True));continue
                if review['slurm']['state']=='OUT_OF_MEMORY':row['memory_gb']=min(96,2*row.get('memory_gb',initial_memory))
            with (HERE/'SUBMISSION.lock').open('a') as submission:
                fcntl.flock(submission,fcntl.LOCK_EX)
                if not snapshot()['free_cpu_shares']:continue
                before=require_capacity();memory=row.get('memory_gb',initial_memory)
                job=subprocess.check_output(['sbatch','--parsable',f'--mem={memory}G',f'--export=ALL,BLV2_POSTHOC_TAG={tag}',str(HERE/'posthoc_metrics_overlap_20261022.sbatch')],text=True).strip()
                row.update(job=job,memory_gb=memory,reviewed=False,submitted_utc=utc())
                write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),job_id=job,stage='posthoc_metrics_'+tag,resource_snapshot=before,memory_gb=memory,posthoc_only=True,independent_of_stage_two=True))
                print('Submitted unchanged saved-evidence metrics',tag,job,flush=True)
        state['utc']=utc();write(STATE,state)
    return state


def wait():
    from coordinator_correction_release_20261023_quota18_now_20261022 import status
    while True:
        state=ensure();live={r['job_id'] for r in snapshot()['jobs']}
        done=all((HERE/proof).exists() for script,proof,mem in TASKS.values())
        active=any(row.get('job') in live or retained_alive(row) for row in state['tasks'].values())
        if done and not active:return
        status('posthoc_metrics_background_completion',tasks=state['tasks']);time.sleep(10)


def worker(tag):
    assert os.environ.get('SLURM_JOB_ID') and tag in TASKS
    verify_receipt(READY);verify_receipt('MIGRATION_NUMBERS.receipt.json')
    import re
    details=subprocess.check_output(['scontrol','show','job','-dd',os.environ['SLURM_JOB_ID']],text=True);cpus=set()
    for value in re.findall(r'CPU_IDs=(\S+)',details):
        for item in value.split(','):
            if '-' in item:
                left,right=map(int,item.split('-'));cpus.update(range(left,right+1))
            elif item.isdigit():cpus.add(int(item))
    assert cpus;os.sched_setaffinity(0,{min(cpus)})
    script,proof,memory=TASKS[tag]
    if (HERE/proof).exists():verify_receipt(proof);return
    # Execute the original sealed source verbatim in this allocated CPU.
    os.execv(sys.executable,[sys.executable,'-B',str(HERE/script)])


if __name__=='__main__':worker(os.environ['BLV2_POSTHOC_TAG'])
