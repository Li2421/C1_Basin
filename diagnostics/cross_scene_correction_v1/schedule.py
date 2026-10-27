"""Quota-aware CPU dispatcher with observed RUNNING confirmations and stall recovery."""
from common import *
from queueing import build, complete
import subprocess
import fcntl
import re
import math

PYTHON = '/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def log(event,**values):
    with (HERE/'SCHEDULER_EVENTS.jsonl').open('a')as f:f.write(canonical(dict(utc=utc(),event=event,**values))+'\n')


def jobs():
    proc=subprocess.run(['squeue','-h','-r','-a','-o','%i|%u|%T|%C|%M|%R|%j|%Z'],capture_output=True,text=True,check=True)
    rows=[]
    for line in proc.stdout.splitlines():
        parts=line.split('|',7)
        if len(parts)==8:rows.append(dict(zip(('id','user','state','cpus','elapsed','reason','name','cwd'),parts)))
    return rows


def cancel(job,graceful=False):
    args=['scancel']+(['--signal=USR1','--full']if graceful else[])+[job]
    result=subprocess.run(args,capture_output=True,text=True)
    if result.returncode:
        if not any(r['id']==job for r in jobs()):
            log('cancel_raced_with_completed_job',job=job,stderr=result.stderr)
            return
        raise RuntimeError(('SLURM_CANCEL_FAILED',job,result.stderr))


def shorter_limit(seconds):
    if seconds>3600:return max(3600,seconds//2)
    if seconds>3360:return seconds-60
    raise RuntimeError('Cannot shorten Slurm below56min while retaining55min new-claim headroom; report protocol conflict.')


def quota(rows):
    # User explicitly authorized all idle CPU capacity on 2026-10-24.
    total=int(subprocess.check_output(['sinfo','-N','-h','-o','%c'],text=True).splitlines()[0])
    others=sum(int(r['cpus'])for r in rows if r['state']=='RUNNING'and not r['name'].startswith('xsc_')and r['name']!='cross_scene_correction')
    # One CPU stays available to the active coordinator / preparation process.
    return max(0,total-others-1)


def pressure_limit(rows,stage,base):
    """Yield only when releasing our CPU/memory can actually start another job."""
    pending=[r for r in rows if r['user']!=os.environ.get('USER','zhihan')and r['state']=='PENDING'
             and any(x in r['reason']for x in ('Resources','Priority'))]
    if not pending:return base
    node=subprocess.check_output(['scontrol','show','node','-d','-o'],text=True)
    def number(key,default=0):
        found=re.search(r'\b'+key+r'=(\d+)',node)
        return int(found.group(1))if found else default
    total=number('CPUTot');alloc=number('CPUAlloc');memory=number('RealMemory');used=number('AllocMem')
    running=sum(r['name']==f'xsc_{stage}'and r['state']=='RUNNING'for r in rows)
    other_cpu=alloc-running;other_mem=max(0,used-running*3072)
    gres=re.search(r'\bGres=([^\s]+)',node);inuse=re.search(r'\bGresUsed=([^\s]+)',node)
    def resource(text,name):
        found=re.search(r'(?:^|,)'+name+r'(?::[^:,()]+)?:(\d+)',text or '')
        return int(found.group(1))if found else 0
    available=gres.group(1)if gres else '';busy=inuse.group(1)if inuse else ''
    for row in pending:
        details=subprocess.check_output(['scontrol','show','job',row['id'],'-o'],text=True)
        req_gres=re.search(r'\bTresPerNode=([^\s]+)',details)
        req=req_gres.group(1).replace('gres/','')if req_gres else ''
        shards=resource(req,'shard');gpus=resource(req,'gpu')
        if shards and (resource(busy,'gpu')or shards>resource(available,'shard')-resource(busy,'shard')):continue
        if gpus and (resource(busy,'shard')or gpus>resource(available,'gpu')-resource(busy,'gpu')):continue
        cpus=int(row['cpus']);m=re.search(r'\bReqTRES=[^\s]*?mem=([0-9.]+)([KMGT]?)',details)
        reqmem=float(m.group(1))*{'K':1/1024,'M':1,'G':1024,'T':1048576,'':1}[m.group(2)]if m else 0.
        if cpus<=total-alloc and reqmem<=memory-used:continue
        if cpus>total-other_cpu or reqmem>memory-other_mem:continue
        target=min(base,max(0,total-other_cpu-cpus-1),max(0,int((memory-other_mem-reqmem-1024)//3072)))
        log('actual_cpu_memory_pressure',other_job=row['id'],requested_cpus=cpus,requested_memory_mb=reqmem,
            free_cpus=total-alloc,free_memory_mb=memory-used,target_workers=target)
        return target
    return base


def release_stale(stage,live):
    folder=HERE/'QUEUES'/stage
    for claim in (folder/'claims').glob('*'):
        if not claim.is_dir()or not(claim/'OWNER.json').exists():continue
        owner=read(claim/'OWNER.json')
        if owner.get('job')in live:continue
        # Slurm gives each array element a numeric SLURM_JOB_ID while squeue %i
        # displays ArrayJobID_TaskID. Never infer death from that format difference.
        check=subprocess.run(['squeue','-h','-j',str(owner.get('job')),'-o','%T'],capture_output=True,text=True)
        if check.returncode==0 and check.stdout.strip():continue
        if (folder/'done'/claim.name).exists():continue
        target=folder/'abandoned'/f'{claim.name}_{time.time_ns()}';target.parent.mkdir(parents=True,exist_ok=True)
        claim.rename(target);log('release_stale_claim',stage=stage,owner=owner)


def merge(stage):
    with (HERE/'MERGER.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=subprocess.run([PYTHON,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,capture_output=True,text=True)
        write(HERE/f'{stage}_MERGE.json',dict(utc=utc(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        assert p.returncode==0,p.stderr
    log('merged',stage=stage)


def run(stage,initial_seconds=10800):
    build(stage);last_result=time.time();last_count=-1;submitted={};first_seen={};limits=initial_seconds
    log('stage_started',stage=stage)
    while not complete(stage):
        rows=jobs();our=[r for r in rows if r['name']==f'xsc_{stage}'];live={r['id']for r in rows}
        release_stale(stage,live)
        count=sum(1 for _ in (HERE/stage/'raw').glob('*.json'))
        if count!=last_count:last_count=count;last_result=time.time()
        cap=pressure_limit(rows,stage,quota(rows));running=[r for r in our if r['state']=='RUNNING'];pending=[r for r in our if r['state']=='PENDING']
        # Only our own work is adjusted; dependency- or GPU-blocked jobs do not steal CPU headroom.
        for r in rows:first_seen.setdefault(r['id'],time.time())
        for r in pending:
            if time.time()-first_seen[r['id']]>900 and len(running)<cap:
                cancel(r['id']);limits=shorter_limit(limits)
                log('pending_over_15min_resubmit_shorter',stage=stage,job=r['id'],seconds=limits)
        if time.time()-last_result>1800 and our:
            for r in our:cancel(r['id'])
            limits=shorter_limit(limits);last_result=time.time()
            log('no_new_result_30min_cancel_resubmit_shorter',stage=stage,seconds=limits)
            time.sleep(5);continue
        if len(running)>cap:
            # Gracefully finish the current episode, then release the allocation.
            for r in running[cap:]:
                cancel(r['id'],graceful=True)
                log('yield_cpu_to_actual_demand',job=r['id'])
        desired=max(0,cap-len(our))
        unfinished=sum(not(HERE/'QUEUES'/stage/'done'/r['group']).exists()for r in read(HERE/'QUEUES'/stage/'INDEX.json'))
        desired=min(desired,max(0,unfinished-len(our)))
        if desired:
            hh=limits//3600;mm=(limits%3600)//60
            command=['sbatch','--parsable',f'--job-name=xsc_{stage}',f'--array=0-{desired-1}',f'--time={hh:02d}:{mm:02d}:00',
                f'--export=ALL,CROSS_SCENE_STAGE={stage},CROSS_SCENE_TIME_LIMIT_S={limits}',str(HERE/'worker.sbatch')]
            job=subprocess.check_output(command,text=True).strip().split(';')[0]
            submitted[job]=time.time();log('submitted',stage=stage,job=job,count=desired,limit_seconds=limits,quota=cap)
            # Immediate actual RUNNING confirmation, not just acceptance by sbatch.
            observed=[]
            for _ in range(20):
                observed=[r for r in jobs()if r['id'].split('_')[0]==job]
                if observed and all(r['state']=='RUNNING'for r in observed):break
                time.sleep(1)
            log('immediate_state_check',stage=stage,job=job,states=observed)
        for job,start in list(submitted.items()):
            if time.time()-start>=180:
                log('three_minute_state_check',stage=stage,job=job,states=[r for r in rows if r['id'].split('_')[0]==job]);del submitted[job]
        write(HERE/'PROGRESS.json',dict(utc=utc(),stage=stage,new_results=count,running=len(running),pending=len(pending),quota=cap,last_new_result_age_seconds=time.time()-last_result))
        time.sleep(30)
    # No more work can be claimed; allow active last-group workers to finish and exit.
    for r in jobs():
        if r['name']==f'xsc_{stage}':cancel(r['id'])
    merge(stage)
    log('stage_complete',stage=stage)


if __name__=='__main__':run(sys.argv[1])
