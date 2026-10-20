"""Enforce explicitly leased array slots; never touch unrelated Slurm jobs."""
from common import *
import subprocess,time,re,fcntl

def snapshot_jobs():
    out=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j|%T'],text=True)
    return [line.split('|') for line in out.splitlines() if line]

def owned_leases():
    result={}
    # Metadata only from this round's submitted jobs. The next round copies
    # this implementation and writes all of its operational records there.
    roots=[HERE]
    for root in roots:
        for path in root.glob('*_SUBMISSION.json'):
            r=read(path);job=str(r['job_id']).split(';')[0];base=2 if r['gpu'] else 18
            for i in range(base,r['workers']):result[f'{job}_{i}']=dict(job=job,index=i,gpu=r['gpu'],receipt=str(path),root=str(root))
    return result

def event(action,job,**extra):
    with (HERE/'RESOURCE_RELEASE_EVENTS.jsonl').open('a') as f:f.write(json.dumps(clean(dict(utc=utc(),action=action,job=job,**extra)))+'\n')

def tick():
    policy=read(HERE/'RESOURCE_RELEASE_POLICY.json');deadline=datetime.fromisoformat(policy['deadline_utc']);now=datetime.now(timezone.utc);jobs=snapshot_jobs();leases=owned_leases();actions=[]
    for job,name,state in jobs:
        if job not in leases or not name.startswith(('crosstype5_','crosstype6_')):continue
        if now>=deadline:
            out=subprocess.run(['scancel',job],capture_output=True,text=True);event('cancel_expired_lease',job,state=state,returncode=out.returncode,stderr=out.stderr);actions.append(job);continue
        if state not in ('RUNNING','CONFIGURING'):continue
        out=subprocess.run(['scontrol','show','job',job,'--oneliner'],capture_output=True,text=True,check=True);fields=dict(re.findall(r'(\w+)=([^ ]+)',out.stdout));start=fields.get('StartTime');end=fields.get('EndTime')
        if not start or start=='Unknown' or not end or end=='Unknown':continue
        # Slurm timestamps are in the host local zone (Asia/Singapore).
        begin=datetime.fromisoformat(start).astimezone(timezone.utc);current_end=datetime.fromisoformat(end).astimezone(timezone.utc)
        if current_end<=deadline:continue
        minutes=int((deadline-begin).total_seconds()//60)
        if minutes<1:continue
        limit=f'{minutes//60:02d}:{minutes%60:02d}:00'
        update=subprocess.run(['scontrol','update',f'JobId={job}',f'TimeLimit={limit}'],capture_output=True,text=True)
        event('cap_slurm_walltime',job,start=start,previous_end=end,target_deadline=policy['user_confirmed_local_deadline'],time_limit=limit,returncode=update.returncode,stderr=update.stderr);actions.append(job)
    write(HERE/'RESOURCE_RELEASE_STATUS.json',dict(utc=utc(),deadline=policy['user_confirmed_local_deadline'],past_deadline=now>=deadline,leases=leases,leased_jobs_in_queue=[j for j,n,s in jobs if j in leases],actions_this_tick=actions,other_jobs_untouched=True))

def main():
    with (HERE/'RESOURCE_RELEASE_WATCH.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                tick()
                if (HERE/'DELIVERY.receipt.json').exists() and (HERE/'COORDINATOR_COMPLETE.json').exists():
                    prefix='crosstype5_' if 'crosstype5_' in HERE.name else 'crosstype6_'
                    if not any(name.startswith(prefix) for job,name,state in snapshot_jobs()):
                        write(HERE/'RESOURCE_RELEASE_WATCH_COMPLETE.json',dict(utc=utc(),reason='Round delivered and all own Slurm jobs ended; stop before next round makes this directory read-only',next_round_must_copy_release_policy=True));return
            except Exception as exc:event('watch_error',None,error=repr(exc))
            time.sleep(10)
if __name__=='__main__':main()
