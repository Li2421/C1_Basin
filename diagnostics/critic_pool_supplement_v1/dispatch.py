"""Official-preflight missing-only pool; source-verified checkpoint handovers."""
from common import *
from schedule import jobs,quota,pressure_limit,merge,log
from schedule import cancel
from queueing import build
import subprocess,re

def owner_alive(owner):
    try:cmd=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return False
    return str(HERE/'worker.py')in cmd or'diagnostics/critic_pool_supplement_v1/worker.py'in cmd

def finished(stage,case):
    rid=case['rollout_uid']
    return(HERE/stage/'raw'/f'{rid}.json').exists()and(HERE/stage/'journal_receipts'/f'{rid}.json').exists()

def run(stage):
    if(HERE/f'{stage}_COMPLETE.json').exists():return
    assert(HERE/f'{stage}_OFFICIAL_PREFLIGHT.json').exists()
    index=build(stage);cases=read(HERE/f'{stage}_MISSING_CASES.json');folder=HERE/'QUEUES'/stage;submitted={};first={};retry_jobs=set();retry=0;limit=3600
    log('started',stage=stage)
    while True:
        fatal=[p for p in(HERE/'WORKER_FATAL').glob('*.json')if read(p)['stage']==stage]
        if fatal:raise RuntimeError(('IMPLEMENTATION_FAILURE',str(fatal[0])))
        done=[c for c in cases if finished(stage,c)]
        if len(done)==len(cases):break
        rows=jobs();ours=[r for r in rows if r['name']=='xcs_'+stage];live={r['id']for r in rows};active={}
        for claim in(folder/'uid_claims').glob('*'):
            if not(claim/'OWNER.json').exists()or(claim/'DONE.json').exists():continue
            owner=read(claim/'OWNER.json');rid=claim.name
            if(HERE/stage/'raw'/f'{rid}.json').exists()and(HERE/stage/'journal_receipts'/f'{rid}.json').exists():continue
            if owner['job']in live or owner_alive(owner):active[owner['job']]=rid;continue
            control=subprocess.run(['scontrol','show','job',owner['job'],'-o'],capture_output=True,text=True)
            state=control.stdout
            if owner['job']not in retry_jobs and any(k in state for k in('JobState=TIMEOUT','JobState=FAILED','JobState=NODE_FAIL','JobState=OUT_OF_MEMORY','JobState=PREEMPTED')):
                retry+=1;retry_jobs.add(owner['job']);log('infrastructure_retry',stage=stage,job=owner['job'],attempt=retry,scontrol=state)
            target=folder/'abandoned'/f'{rid}_{time.time_ns()}';target.parent.mkdir(parents=True,exist_ok=True);claim.rename(target)
            log('released_ended_uid_owner',stage=stage,owner=owner)
        if retry>3:
            write(HERE/f'{stage}_GAP.json',dict(utc=utc(),reason='Infrastructure retry cap exceeded',missing=[c['rollout_uid']for c in cases if not finished(stage,c)]));break
        cap=pressure_limit(rows,stage,quota(rows));running=[r for r in ours if r['state']=='RUNNING'];pending=[r for r in ours if r['state']=='PENDING']
        for r in ours:first.setdefault(r['id'],time.time())
        for r in pending:
            if time.time()-first[r['id']]>900 and len(running)<cap:cancel(r['id']);limit=max(600,limit//2);log('pending_15min_shorten',stage=stage,job=r['id'],seconds=limit)
        # Yield idle workers first; a running search is checkpointed only when
        # real resource pressure requires it, never because results are slow.
        order=sorted(running,key=lambda r:r['id']in active)
        for r in order[:max(0,len(running)-cap)]:cancel(r['id'],True)
        unclaimed=[c for c in cases if not finished(stage,c)and not(folder/'uid_claims'/c['rollout_uid']).exists()]
        if not unclaimed:
            for r in ours:
                if r['id']not in active:cancel(r['id'],True)
        desired=max(0,min(cap-len(ours),len(unclaimed)))
        if desired:
            cmd=['sbatch','--parsable',f'--job-name=xcs_{stage}',f'--array=0-{desired-1}',f'--time={limit//3600:02d}:{limit%3600//60:02d}:00',
                f'--export=ALL,XDT_STAGE={stage},XDT_TIME_LIMIT_S={limit}',str(HERE/'worker.sbatch')]
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0];submitted[job]=time.time();log('submitted',stage=stage,job=job,count=desired,seconds=limit)
            observed=[]
            for _ in range(20):
                observed=[r for r in jobs()if r['id'].split('_')[0]==job]
                if observed and all(r['state']=='RUNNING'for r in observed):break
                time.sleep(1)
            log('immediate_RUNNING_readback',stage=stage,job=job,states=observed)
        for job,t in list(submitted.items()):
            if time.time()-t>=180:log('three_minute_RUNNING_readback',stage=stage,job=job,states=[r for r in rows if r['id'].split('_')[0]==job],completed=len(done));del submitted[job]
        write(HERE/'PROGRESS.json',dict(utc=utc(),stage=stage,new_results=len(done),total_missing=len(cases),remaining=len(cases)-len(done),running=len(running),pending=len(pending),worker_cap=cap,infrastructure_retries=retry,gpu_shares=0))
        (HERE/'STATUS.md').write_text(f'# 当前状态\n\n{utc()}：{stage}，完成{len(done)}/{len(cases)}个缺失请求，运行{len(running)}，补交{retry}/3。全部选择已冻结。\n\n逐UID独占领取，运行中仿真仅因时限或实际资源压力使用已验证的检查点交接；没有无新结果超时撤销规则。预计剩余时间受规划长尾影响。\n')
        time.sleep(10)
    for r in jobs():
        if r['name']=='xcs_'+stage:cancel(r['id'],True)
    for _ in range(30):
        if not any(r['name']=='xcs_'+stage for r in jobs()):break
        time.sleep(1)
    assert not any(r['name']=='xcs_'+stage for r in jobs())
    for e in index:
        if all(finished(stage,c)for c in read(e['path'])):write(folder/'done'/e['group'],dict(utc=utc(),count=e['count'],uid_exclusive_execution=True))
    merge(stage);log('completed',stage=stage)
    write(HERE/f'{stage}_COMPLETE.json',dict(utc=utc(),stage=stage,requests=len(cases),completed=sum(finished(stage,c)for c in cases),infrastructure_retries=retry,gaps=(HERE/f'{stage}_GAP.json').exists()))

if __name__=='__main__':run(sys.argv[1])
