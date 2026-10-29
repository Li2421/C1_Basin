"""Scheduled user CPU allowance; cancel only brd1 workers."""
import getpass,subprocess,datetime,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def allowance(now=None):
    path = HERE / 'RESOURCE_LIMIT.json'
    policy = json.loads(path.read_text()) if path.exists() else {'total_cpus':18}
    current = now or datetime.datetime.now(datetime.timezone.utc)
    if current.tzinfo is None:
        raise ValueError('CPU schedule requires a timezone-aware time')
    cutoff = policy.get('release_at')
    limit = int(policy['total_cpus'])
    if cutoff and current >= datetime.datetime.fromisoformat(cutoff):
        limit = int(policy['after_release_cpus'])
    if limit < 1:
        raise ValueError('CPU limit must include coordinator')
    return limit


def task_allowance(now=None):
    policy=json.loads((HERE/'RESOURCE_LIMIT.json').read_text())
    marker=policy.get('peer_completion_marker')
    finished=policy.get('peer_completion_verified',False) or bool(marker and Path(marker).is_file())
    if policy.get('borrow_after_peer_completion') and finished:
        return allowance(now)
    return allowance(now) // 2

def counts(rows):
    active=[r for r in rows if r['state'] in ('RUNNING','CONFIGURING','COMPLETING')]
    return sum(r['cpus'] for r in active),sum(r['cpus'] for r in active if r['name'].startswith(('brd1_','pbp1_')))

def available():
    total,own=counts(jobs())
    return max(0,min(allowance()-total,task_allowance()-own))


def jobs():
    out=subprocess.check_output(['squeue','-h','-u',getpass.getuser(),'-o','%i|%j|%T|%C'],text=True)
    return [dict(job_id=a,name=b,state=c,cpus=int(d))for a,b,c,d in(line.split('|')for line in out.splitlines())]

def enforce():
    rows=jobs();active=[j for j in rows if j['state']in('RUNNING','CONFIGURING','COMPLETING')];total,own=counts(rows);cancelled=[]
    for job in sorted([j for j in active if j['name'].startswith('brd1_')and j['name']!='brd1_pipeline'],key=lambda j:int(j['job_id'].split('_')[0]),reverse=True):
        if total<=allowance() and own<=task_allowance():break
        subprocess.run(['scancel',job['job_id']],check=True);cancelled.append(job);total-=job['cpus'];own-=job['cpus']
    data=dict(time=datetime.datetime.now().astimezone().isoformat(),user_total_limit=allowance(),task_limit=task_allowance(),observed_jobs=rows,cancelled=cancelled,only_prefix='brd1_')
    tmp=HERE/'RESOURCE_STATUS.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(HERE/'RESOURCE_STATUS.json')
    return data
