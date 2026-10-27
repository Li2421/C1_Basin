"""Resumable posthoc diagnostics on immutable traces, one CPU per job."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from result_inventory import find_raw
from resources_corrections_release_20261023 import require_capacity
import subprocess
import fcntl
import time

def prepare(original):
    stage='analysis_'+original
    if (HERE/stage/'WORK_QUEUE.receipt.json').exists():return stage
    cases=read(HERE/f'{original}_CASES.json');seen=set();rows=[]
    for c in cases:
        rid=c['rollout_uid']
        if rid in seen:continue
        seen.add(rid);path=find_raw(rid,original,c['state']['scene'])
        rows.append(dict(rollout_uid=rid,path=str(path),sha256=sha(path)))
    queue=HERE/stage/'queue';queue.mkdir(parents=True,exist_ok=True);index=[]
    for offset in range(0,len(rows),32):
        values=rows[offset:offset+32];group=f'{offset//32:06d}';path=queue/f'{group}.json';write(path,values)
        index.append(dict(group=group,file=str(path),sha256=sha(path),rollout_uids=[r['rollout_uid'] for r in values]))
    write(HERE/stage/'GROUP_INDEX.json',index)
    seal(f'{stage}/WORK_QUEUE.receipt.json',[HERE/stage/'GROUP_INDEX.json',HERE/'failure_analysis.py',HERE/'atomic_write.py',HERE/'analysis_jobs_current.py'],new_rollouts=0,source_stage=original)
    return stage

def submit(stage):
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('POSTHOC_CACHE_DEPLOYMENT.receipt.json')
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);before=require_capacity()
        job=subprocess.check_output(['sbatch','--parsable',f'--export=ALL,BLV2_STAGE={stage}',str(HERE/'analysis_cached.sbatch')],text=True).strip()
        write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),stage=stage,job_id=job,resource_snapshot=before,posthoc_only=True))
        print('Submitted analysis',job,stage,flush=True)
        return job

def worker(stage):
    import numpy as np
    from failure_analysis import summary
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('POSTHOC_CACHE_DEPLOYMENT.receipt.json');index=read(HERE/stage/'GROUP_INDEX.json')
    claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);groups=0
    for entry in index:
        claim=claims/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),utc=utc()))
        assert sha(entry['file'])==entry['sha256']
        for row in read(entry['file']):
            dest=HERE/'DIAGNOSTICS'/f"{row['rollout_uid']}.json"
            if dest.exists():continue
            assert sha(row['path'])==row['sha256'];raw=read(row['path'])
            assert sha(raw['trace_path'])==raw['trace_sha256']
            with np.load(raw['trace_path']) as z:result=summary(raw,z)
            write(dest,dict(utc=utc(),rollout_uid=row['rollout_uid'],source=row,trace_sha256=raw['trace_sha256'],**result))
        write(claim/'COMPLETE.json',dict(utc=utc()));groups+=1
        if groups>=128:os.execv(sys.executable,[sys.executable,'-B',str(HERE/'analysis_jobs_current.py'),'worker',stage])
    print(stage,'posthoc worker complete',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(sys.argv[2])
    else:worker(sys.argv[2])
