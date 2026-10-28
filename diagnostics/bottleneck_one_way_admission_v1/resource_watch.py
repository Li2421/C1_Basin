"""Latest user allocation: 24 CPUs until Oct29 09:00 Singapore, then 18."""
import getpass,subprocess,datetime,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEADLINE=datetime.datetime.fromisoformat('2026-10-29T09:00:00+08:00')
def allowance(now=None):
    return 24 if (now or datetime.datetime.now(datetime.timezone.utc))<DEADLINE else 18
def jobs():
    out=subprocess.check_output(['squeue','-h','-u',getpass.getuser(),'-o','%i|%j|%T|%C'],text=True)
    return [dict(job_id=a,name=b,state=c,cpus=int(d))for a,b,c,d in (line.split('|')for line in out.splitlines())]

def enforce():
    rows=jobs();active=[j for j in rows if j['state']in('RUNNING','CONFIGURING','COMPLETING')]
    limit=allowance();total=sum(j['cpus']for j in active);cancelled=[]
    # Only the user's current two C1 study prefixes. Old-prefix cancellation
    # is authorized solely by the explicit scheduled CPU reduction, not by
    # the new experiment's ordinary resource allocation.
    own=[j for j in active if j['name'].startswith(('owa1_','stb3_'))and j['name']not in('owa1_pipeline','stb3_pipeline')]
    for job in sorted(own,key=lambda j:int(j['job_id'].split('_')[0]),reverse=True):
        if total<=limit:break
        subprocess.run(['scancel',job['job_id']],check=True);cancelled.append(job);total-=job['cpus']
    data=dict(time=datetime.datetime.now().astimezone().isoformat(),user_total_limit=limit,observed_jobs=rows,cancelled=cancelled,
              deadline=DEADLINE.isoformat(),authorization='User: 今天早上9点释放6个cpu',unrelated_jobs_never_cancelled=True)
    tmp=HERE/'RESOURCE_STATUS.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(HERE/'RESOURCE_STATUS.json')
    if cancelled:
        with(HERE/'resource_events.jsonl').open('a')as f:f.write(json.dumps(data)+'\n')

if __name__=='__main__':
    while True:
        try:enforce()
        except Exception as exc:
            with(HERE/'resource_events.jsonl').open('a')as f:f.write(json.dumps(dict(error=repr(exc),time=time.time()))+'\n')
        time.sleep(2)
