"""Split sealed requests into short claims without changing any request bytes."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from infrastructure_recovery_active import classify
from resources_released import snapshot,require_capacity
import collections
import subprocess
import fcntl
import time


def eligible(stage):
    return stage.startswith(('learning_labels_','learning_eval_'))


def prepare(stage):
    assert eligible(stage),stage
    target=HERE/stage/'FINE_QUEUE.receipt.json'
    if target.exists():return
    verify_receipt('PIPELINED_SCHEDULING.receipt.json')
    verify_receipt(f'{stage}_PREFLIGHT.receipt.json')
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
    index=[];before=[];after=[]
    for entry in read(HERE/stage/'GROUP_INDEX.json'):
        assert sha(entry['file'])==entry['sha256']
        cases=read(entry['file']);before.extend(cases)
        # Keep a deterministic controller's seed pair together, preserving the
        # existing seed-1 alias. Labels have only seed 0: one episode per claim.
        start=0
        while start<len(cases):
            end=start+1
            if end<len(cases) and cases[start]['seed']==0 and cases[end]['seed']==1:
                left={k:v for k,v in cases[start].items() if k not in ('seed','seed_key','rollout_uid')}
                right={k:v for k,v in cases[end].items() if k not in ('seed','seed_key','rollout_uid')}
                if left==right:end+=1
            values=cases[start:end];after.extend(values)
            index.append(dict(group=f"{entry['group']}_{start:02d}",
                original_group=entry['group'],file=entry['file'],sha256=entry['sha256'],
                start=start,end=end,rollout_uids=[c['rollout_uid'] for c in values]))
            start=end
    assert before==after
    rids=[c['rollout_uid'] for c in before];assert len(rids)==len(set(rids))
    write(HERE/stage/'FINE_GROUP_INDEX.json',index)
    seal(f'{stage}/FINE_QUEUE.receipt.json',
        [HERE/stage/'FINE_GROUP_INDEX.json',HERE/stage/'WORK_QUEUE.receipt.json',HERE/'fine_dispatch.py'],
        original_requests=len(before),fine_requests=sum(len(r['rollout_uids']) for r in index),
        request_payloads_identical=True,new_registration=False,maximum_requests_per_claim=2)


def sync_original(stage):
    """Original status receipts remain true only when all evidence is landed."""
    for entry in read(HERE/stage/'GROUP_INDEX.json'):
        claim=HERE/stage/'claims'/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        if all((HERE/stage/'raw'/f'{rid}.json').exists() and
               (HERE/stage/'journal_receipts'/f'{rid}.json').exists() for rid in entry['rollout_uids']):
            claim.mkdir(parents=True,exist_ok=True)
            write(claim/'COMPLETE.json',dict(utc=utc(),all_raw_and_journals_present=True,
                fine_queue_receipt_sha256=sha(HERE/stage/'FINE_QUEUE.receipt.json')))


def recover(stage,live_jobs):
    claims=HERE/stage/'fine_claims';claims.mkdir(parents=True,exist_ok=True)
    for claim in list(claims.iterdir()):
        if not claim.is_dir() or (claim/'COMPLETE.json').exists():continue
        owner=claim/'OWNER.json'
        if not owner.exists():
            if time.time()-claim.stat().st_mtime<120:continue
        else:
            job=str(read(owner)['job'])
            if job in live_jobs:continue
            review=classify(stage,job)
            if review['slurm']['state'] in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED'):continue
            write(HERE/stage/'fine_failure_reviews'/f'{job}.json',review)
            if review['unhandled_implementation_error']:
                write(HERE/stage/'NEEDS_IMPLEMENTATION_REVIEW.json',dict(utc=utc(),claim=str(claim),review=review))
                continue
            if review['slurm']['state']=='OUT_OF_MEMORY':
                import re
                memory=re.search(r'MinMemoryNode=(\d+)G',review['slurm']['details'])
                old=int(memory.group(1)) if memory else 4
                write(HERE/stage/'MEMORY_OVERRIDE_GB.json',dict(utc=utc(),memory_gb=min(96,old*2),job=job,reason='Observed OOM; resources only'))
        archive=HERE/stage/'fine_claims_abandoned';archive.mkdir(exist_ok=True)
        claim.rename(archive/f'{claim.name}_{time.time_ns()}')


def available(stage):
    return sum(not (HERE/stage/'fine_claims'/r['group']).exists()
        for r in read(HERE/stage/'FINE_GROUP_INDEX.json'))


def submit(stage):
    verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json')
    verify_receipt('PIPELINED_SCHEDULING.receipt.json')
    verify_receipt(f'{stage}_PREFLIGHT.receipt.json')
    verify_receipt(f'{stage}/FINE_QUEUE.receipt.json')
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);before=require_capacity()
        memory=read(HERE/stage/'MEMORY_OVERRIDE_GB.json')['memory_gb'] if (HERE/stage/'MEMORY_OVERRIDE_GB.json').exists() else 4
        job=subprocess.check_output(['sbatch','--parsable',f'--mem={memory}G',
            f'--export=ALL,BLV2_STAGE={stage}',str(HERE/'worker_flat_static.sbatch')],text=True).strip()
        write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),stage=stage,
            job_id=job,resource_snapshot=before,memory_gb=memory,fine_dispatch=True,
            original_preflight_sha256=sha(HERE/f'{stage}_PREFLIGHT.receipt.json'),
            fine_queue_receipt_sha256=sha(HERE/stage/'FINE_QUEUE.receipt.json')))
        print('Submitted fine claim worker',job,stage,flush=True)
        return job
