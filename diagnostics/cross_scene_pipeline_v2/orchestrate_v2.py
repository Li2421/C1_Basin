"""Quota and pressure checks, bounded infrastructure retries and frequent status."""
from common import *
from schedule import jobs,quota,pressure_limit,log,release_stale
from queue_v2 import build,complete
import subprocess

def cancel(job,graceful=False):
    row=next((r for r in jobs()if r['id']==job),None)
    if row is None:return
    assert row['name'].startswith('xp2_'),('REFUSE_NON_V2_CANCELLATION',row)
    p=subprocess.run(['scancel']+(['--signal=USR1','--full']if graceful else[])+[job],capture_output=True,text=True)
    if p.returncode and any(r['id']==job for r in jobs()):raise RuntimeError(p.stderr)

def status(stage,index,rows,retries):
    done=sum((HERE/'QUEUES'/stage/'done'/e['group']).exists()for e in index)
    count=sum(1 for _ in(HERE/stage/'raw').glob('*.json'))
    current=[r for r in rows if r['name']=='xp2_'+stage]
    write(HERE/'PROGRESS.json',dict(utc=utc(),stage=stage,completed_tasks=count,completed_groups=done,total_groups=len(index),running=sum(r['state']=='RUNNING'for r in current),
        pending=sum(r['state']=='PENDING'for r in current),infrastructure_retries=retries,total_cpu_cap=18,gpu_shares=0))
    text=f'# 当前状态\n\n更新时间：{utc()}。阶段：{stage}。完成任务{count}，任务组{done}/{len(index)}；运行{sum(r["state"]=="RUNNING"for r in current)}，基础设施补交{retries}/3。\n\n'
    text+='没有按无新结果时长撤销作业；预计剩余时间随实测训练与试跑成本更新。保留最终检验的新场景只审计os.stat元数据。\n'
    (HERE/'STATUS.md').write_text(text)

def run(stage):
    if(HERE/f'{stage}_COMPLETE.json').exists():return
    index=build(stage);first={};submitted={};limit=10800;retries=0;seen_failed=set();log('started',stage=stage)
    while not complete(stage):
        fatal=[p for p in(HERE/'WORKER_FATAL').glob('*.json')if read(p)['stage']==stage]
        if fatal:
            write(HERE/'IMPLEMENTATION_FAILURE.json',dict(stage=stage,failures=[str(p)for p in fatal],must_inspect_logs=True))
            raise RuntimeError(('IMPLEMENTATION_FAILURE',str(fatal[0])))
        rows=jobs();ours=[r for r in rows if r['name']=='xp2_'+stage];live={r['id']for r in rows}
        # Finished allocations with an unfinished group can be a natural
        # deadline handover or infrastructure failure; inspect Slurm history.
        for claim in(HERE/'QUEUES'/stage/'claims').glob('*'):
            if not(claim/'OWNER.json').exists()or(HERE/'QUEUES'/stage/'done'/claim.name).exists():continue
            owner=read(claim/'OWNER.json');job=owner['job']
            if job in live or job in seen_failed:continue
            result=subprocess.run(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode'],capture_output=True,text=True)
            state=result.stdout
            if any(s in state for s in('NODE_FAIL','TIMEOUT','OUT_OF_MEMORY','FAILED','PREEMPTED','BOOT_FAIL')):
                retries+=1;seen_failed.add(job);log('infrastructure_retry',stage=stage,job=job,attempt=retries,sacct=state)
                if retries>3:
                    write(HERE/f'{stage}_GAP.json',dict(stage=stage,reason='Infrastructure retry limit exceeded',sacct=state));return
        release_stale(stage,live)
        cap=min(17,pressure_limit(rows,stage,quota(rows)));running=[r for r in ours if r['state']=='RUNNING'];pending=[r for r in ours if r['state']=='PENDING']
        for r in ours:first.setdefault(r['id'],time.time())
        for r in pending:
            if time.time()-first[r['id']]>900 and len(running)<cap:cancel(r['id']);limit=max(3360,limit//2);log('pending_15min_shorten',stage=stage,job=r['id'],seconds=limit)
        for r in running[cap:]:cancel(r['id'],True);log('yield_actual_cpu_memory_need',stage=stage,job=r['id'])
        remaining=sum(not(HERE/'QUEUES'/stage/'done'/e['group']).exists()for e in index)
        desired=max(0,min(cap-len(ours),remaining-len(ours)))
        if desired:
            command=['sbatch','--parsable',f'--job-name=xp2_{stage}',f'--array=0-{desired-1}',f'--time={limit//3600:02d}:{limit%3600//60:02d}:00',
                f'--export=ALL,XP2_STAGE={stage},XP2_TIME_LIMIT_S={limit}',str(HERE/'worker_v2.sbatch')]
            job=subprocess.check_output(command,text=True).strip().split(';')[0];submitted[job]=time.time();log('submitted',stage=stage,job=job,count=desired,seconds=limit)
            observed=[]
            for _ in range(20):
                observed=[r for r in jobs()if r['id'].split('_')[0]==job]
                if observed and all(r['state']=='RUNNING'for r in observed):break
                time.sleep(1)
            log('immediate_RUNNING_readback',stage=stage,job=job,states=observed)
        for job,t in list(submitted.items()):
            if time.time()-t>=180:log('three_minute_RUNNING_readback',stage=stage,job=job,states=[r for r in rows if r['id'].split('_')[0]==job]);del submitted[job]
        status(stage,index,rows,retries);time.sleep(10)
    # Every fit receipt has been durably published before idle allocations stop.
    for r in jobs():
        if r['name']=='xp2_'+stage:cancel(r['id'],True)
    log('completed',stage=stage);write(HERE/f'{stage}_COMPLETE.json',dict(utc=utc(),stage=stage,infrastructure_retries=retries))

if __name__=='__main__':run(sys.argv[1])
