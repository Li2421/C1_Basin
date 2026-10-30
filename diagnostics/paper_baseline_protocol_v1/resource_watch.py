"""Count all user CPU allocations; cancellation is confined to pbp1_ jobs."""
import guard
from guard import HERE
import getpass
import json
import subprocess
import datetime
from pathlib import Path

DEFAULT_LIMIT = 18


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
    if policy.get('task_cpu_override') is not None:
        return min(allowance(now),int(policy['task_cpu_override']))
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
    output = subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%T|%C'], text=True)
    result = []
    for line in output.splitlines():
        jobid, name, state, cpus = line.split('|')
        result.append(dict(jobid=jobid, name=name, state=state, cpus=int(cpus), ours=name.startswith('pbp1_')))
    return result


def cancel_own(jobid):
    # Query rather than trusting a caller-supplied job name or saved ID.
    found = [row for row in jobs() if row['jobid'] == str(jobid) or row['jobid'].split('_')[0] == str(jobid)]
    if not found:
        return
    assert all(row['ours'] for row in found), ('Refusing to cancel another task', jobid, found)
    subprocess.run(['scancel', str(jobid)], check=True)


def enforce():
    rows=jobs();active=[r for r in rows if r['state']in('RUNNING','CONFIGURING','COMPLETING')]
    total,own=counts(rows);cancelled=[]
    ours=sorted([r for r in active if r['ours']and r['name']!='pbp1_pipeline'],
                key=lambda r:int(r['jobid'].split('_')[0]),reverse=True)
    for row in ours:
        if total<=allowance() and own<=task_allowance():break
        cancel_own(row['jobid']);total-=row['cpus'];own-=row['cpus'];cancelled.append(row)
    value=dict(time=datetime.datetime.now().astimezone().isoformat(),limit=allowance(),task_limit=task_allowance(),
               observed_total=sum(r['cpus']for r in active),cancelled=cancelled,
               only_prefix='pbp1_',other_tasks_untouched=True)
    path=HERE/'RESOURCE_STATUS.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
    return value
