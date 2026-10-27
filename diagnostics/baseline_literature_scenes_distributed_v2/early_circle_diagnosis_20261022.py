"""Diagnose landed circle groups with the future canonical exclusive claims."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from result_inventory import find_raw
from resources_corrections_release_20261023_quota18_now_20261022 import require_capacity,snapshot
import subprocess,fcntl,time,signal

ORIGINAL='learning_eval_revised_20261021_lit_circle_five'
TARGET='analysis_'+ORIGINAL
FOLDER=HERE/'early_circle_diagnosis_20261022'
READY='EARLY_CIRCLE_DIAGNOSIS_READY_20261022.receipt.json'
STOP=False


def stop(*args):
    global STOP
    STOP=True


def prepare():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    assert not (HERE/READY).exists()
    assert not (HERE/TARGET/'WORK_QUEUE.receipt.json').exists(),'Use the existing complete canonical queue instead'
    verify_receipt(ORIGINAL+'_PREFLIGHT.receipt.json')
    cases=read(HERE/f'{ORIGINAL}_CASES.json');seen=set();ordered=[]
    for c in cases:
        if c['rollout_uid'] not in seen:ordered.append(c);seen.add(c['rollout_uid'])
    index=[];excluded=[];queue=FOLDER/'queue';queue.mkdir(parents=True,exist_ok=True)
    for offset in range(0,len(ordered),32):
        rows=[];group=f'{offset//32:06d}';missing=[]
        for c in ordered[offset:offset+32]:
            try:path=find_raw(c['rollout_uid'],ORIGINAL,c['state']['scene'])
            except FileNotFoundError:missing.append(c['rollout_uid']);continue
            rows.append(dict(rollout_uid=c['rollout_uid'],path=str(path),sha256=sha(path)))
        if missing:excluded.append(dict(group=group,missing=missing));continue
        path=queue/(group+'.json');write(path,rows)
        index.append(dict(group=group,file=str(path),sha256=sha(path),rollout_uids=[r['rollout_uid'] for r in rows]))
    write(FOLDER/'GROUP_INDEX.json',index)
    write(FOLDER/'PLAN.json',dict(utc=utc(),original=ORIGINAL,canonical_target=TARGET,groups=len(index),requests=sum(len(r['rollout_uids']) for r in index),excluded_not_landed_groups=excluded,case_manifest_sha256=sha(HERE/f'{ORIGINAL}_CASES.json'),group_rule='Exactly frozen analysis_jobs_current.prepare:deduplicate rollout UID preserving CASES order;32 rows per group;offset//32 identifiers',exclusive_claims='Use canonical target/claims atomically;future canonical workers skip the same owned/completed groups',original_failure_analysis_function=True,new_rollouts=0,new_training=0,only_saved_trajectories=True))
    seal(READY,[Path(__file__),HERE/'early_circle_diagnosis_20261022.sbatch',FOLDER/'GROUP_INDEX.json',FOLDER/'PLAN.json',HERE/f'{ORIGINAL}_PREFLIGHT.receipt.json',HERE/'analysis_jobs_current.py',HERE/'analysis_jobs_backfill.py',HERE/'failure_analysis.py',HERE/'diagnosis_alias_cache.py']+[Path(r['file']) for r in index],new_rollouts=0,cooperative_yield=True,canonical_group_definition_unchanged=True,shared_atomic_claims_prevent_duplicate_diagnoses=True,total_cpu=18,gpu_shares=2)
    jobs=[]
    for _ in range(min(snapshot()['free_cpu_shares'],len(index))):
        with (HERE/'SUBMISSION.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=snapshot()
            if not state['free_cpu_shares']:break
            require_capacity()
            job=subprocess.check_output(['sbatch','--parsable',str(HERE/'early_circle_diagnosis_20261022.sbatch')],text=True).strip();jobs.append(job)
            write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),job_id=job,stage=TARGET,resource_snapshot=state,posthoc_only=True,early_landed_groups=True,cooperative_yield=True,evidence_receipt=READY))
    write(FOLDER/'SUBMITTED.json',dict(utc=utc(),jobs=jobs,groups=len(index),new_rollouts=0))
    print('Submitted early saved diagnostics',len(jobs),'workers;',len(index),'complete canonical groups;no new rollout',flush=True)


def worker():
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    verify_receipt(READY)
    import numpy as np
    from failure_analysis import summary
    from diagnosis_alias_cache import reuse
    claims=HERE/TARGET/'claims';claims.mkdir(parents=True,exist_ok=True)
    for entry in read(FOLDER/'GROUP_INDEX.json'):
        if STOP:return
        claim=claims/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),utc=utc(),early_queue_receipt_sha256=sha(HERE/READY)))
        assert sha(entry['file'])==entry['sha256']
        for row in read(entry['file']):
            if STOP:return
            dest=HERE/'DIAGNOSTICS'/f"{row['rollout_uid']}.json"
            if dest.exists():continue
            assert sha(row['path'])==row['sha256'];raw=read(row['path']);assert sha(raw['trace_path'])==raw['trace_sha256']
            twin=reuse(raw)
            if twin is None:
                with np.load(raw['trace_path']) as z:result=summary(raw,z)
                reuse_info={}
            else:result=twin['result'];reuse_info=dict(alias_diagnosis_reuse=twin['provenance'])
            write(dest,dict(utc=utc(),rollout_uid=row['rollout_uid'],source=row,trace_sha256=raw['trace_sha256'],**reuse_info,**result))
        write(claim/'COMPLETE.json',dict(utc=utc(),early_saved_diagnosis=True,early_queue_receipt_sha256=sha(HERE/READY),rollout_uids=entry['rollout_uids']))
    print(TARGET,'early saved posthoc worker complete',flush=True)


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else worker()
