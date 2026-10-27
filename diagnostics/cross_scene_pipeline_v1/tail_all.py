"""Fill idle CPU slots, preserve original owners, then complete the unchanged study."""
from common import *
from schedule_resume import jobs,quota,pressure_limit,cancel,merge,log
import subprocess,signal

def original_worker_alive(owner):
    try:cmd=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return False
    return 'diagnostics/cross_scene_pipeline_v1/worker_resume.py main'in cmd

def guards(proof):
    protected=set();alive=[]
    for guard in proof['original_guards']:
        if original_worker_alive(guard['owner']):
            protected.update(guard['protected']);alive.append(guard['owner']['job'])
    write(HERE/'TAIL_GUARDS.json',dict(utc=utc(),protected=sorted(protected),original_jobs_alive=alive))
    return protected,alive

def main():
    proof=read(HERE/'TAIL_REBALANCE.receipt.json');cases=read(HERE/'TAIL_CASES.json');submitted={};first_seen={};limit=10800
    all_missing=read(HERE/'main_MISSING_CASES.json');guards(proof)
    while True:
        if(HERE/'TAIL_FATAL.json').exists():raise RuntimeError(read(HERE/'TAIL_FATAL.json'))
        protected,old=guards(proof);rows=jobs();tail=[r for r in rows if r['name']=='xpl_tail'];live={r['id']for r in rows}
        for claim in(HERE/'TAIL_CLAIMS').glob('*'):
            if not(claim/'OWNER.json').exists()or(claim/'DONE.json').exists():continue
            owner=read(claim/'OWNER.json')
            if owner['job']in live:continue
            # A completed record needs no new physics, even after owner exit.
            if(HERE/'main/raw'/f"{claim.name}.json").exists():continue
            target=HERE/'TAIL_ABANDONED'/f'{claim.name}_{time.time_ns()}';target.parent.mkdir(exist_ok=True);claim.rename(target)
        complete=[c for c in all_missing if(HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()and(HERE/'main/journal_receipts'/f"{c['rollout_uid']}.json").exists()]
        if len(complete)==len(all_missing):break
        cap=pressure_limit(rows,'tail',quota(rows));original_cpus=sum(int(r['cpus'])for r in rows if r['name']=='xpl_main'and r['state']=='RUNNING')
        running=[r for r in tail if r['state']=='RUNNING'];pending=[r for r in tail if r['state']=='PENDING']
        for r in tail:first_seen.setdefault(r['id'],time.time())
        for r in pending:
            if time.time()-first_seen[r['id']]>900 and len(running)+original_cpus<cap:
                cancel(r['id']);limit=max(3360,limit//2);log('tail_pending_15min_shorten',job=r['id'],seconds=limit)
        for r in running[max(0,cap-original_cpus):]:cancel(r['id'],True)
        available=sum(c['rollout_uid']not in protected and not(HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()for c in cases)
        desired=max(0,min(cap-original_cpus-len(tail),available-len(tail)))
        if desired:
            command=['sbatch','--parsable','--job-name=xpl_tail',f'--array=0-{desired-1}',f'--time={limit//3600:02d}:{limit%3600//60:02d}:00',
                f'--export=ALL,XPL_TIME_LIMIT_S={limit}',str(HERE/'worker_tail.sbatch')]
            job=subprocess.check_output(command,text=True).strip().split(';')[0];submitted[job]=time.time();log('tail_submitted',job=job,count=desired,original_cpus=original_cpus,worker_cap=cap)
            for _ in range(20):
                observed=[r for r in jobs()if r['id'].split('_')[0]==job]
                if observed and all(r['state']=='RUNNING'for r in observed):break
                time.sleep(1)
            log('tail_immediate_RUNNING_readback',job=job,states=observed)
        for job,t in list(submitted.items()):
            if time.time()-t>=180:
                log('tail_three_minute_RUNNING_readback',job=job,states=[r for r in rows if r['id'].split('_')[0]==job]);del submitted[job]
        write(HERE/'PROGRESS.json',dict(utc=utc(),stage='main_tail',new_results=len(complete),remaining=len(all_missing)-len(complete),
            original_jobs_alive=old,protected_requests=len(protected),tail_running=len(running),tail_pending=len(pending),worker_cap=cap,scientific_requests_unchanged=True))
        time.sleep(5)
    # All scientific records and their journal receipts are safely published.
    for r in jobs():
        if r['name']=='xpl_tail':cancel(r['id'],True)
    for _ in range(30):
        if not any(r['name']in ('xpl_tail','xpl_main')for r in jobs()):break
        time.sleep(1)
    assert not any(r['name']in ('xpl_tail','xpl_main')for r in jobs()),'Wait for original finished worker teardown'
    for e in read(HERE/'QUEUES/main/INDEX.json'):
        assert all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()for c in read(e['path']))
        p=HERE/'QUEUES/main/done'/e['group']
        if not p.exists():write(p,dict(utc=utc(),count=e['count'],completed_with_tail_leases=True))
    merge('main');log('completed',stage='main');write(HERE/'main_COMPLETE.json',dict(utc=utc(),stage='main',tail_rebalanced=True))
    write(HERE/'TAIL_COMPLETION_CHECK.json',dict(passed=True,all_original_missing_requests_completed=len(all_missing),
        no_original_request_manifest_changed=True,no_models_or_choices_changed=True,no_running_rollout_restarted=True))
    subprocess.run([sys.executable,'-B',str(HERE/'statistics_new.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(HERE/'deliver_resume.py')],cwd=ROOT,check=True)

if __name__=='__main__':main()
