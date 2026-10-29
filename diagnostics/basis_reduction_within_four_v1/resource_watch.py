"""The user's total allocation is18 CPU shares; cancel only brd1 workers."""
import getpass,subprocess,datetime,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def allowance(now=None):return 18

def jobs():
    out=subprocess.check_output(['squeue','-h','-u',getpass.getuser(),'-o','%i|%j|%T|%C'],text=True)
    return [dict(job_id=a,name=b,state=c,cpus=int(d))for a,b,c,d in(line.split('|')for line in out.splitlines())]

def enforce():
    rows=jobs();active=[j for j in rows if j['state']in('RUNNING','CONFIGURING','COMPLETING')];total=sum(j['cpus']for j in active);cancelled=[]
    for job in sorted([j for j in active if j['name'].startswith('brd1_')and j['name']!='brd1_pipeline'],key=lambda j:int(j['job_id'].split('_')[0]),reverse=True):
        if total<=18:break
        subprocess.run(['scancel',job['job_id']],check=True);cancelled.append(job);total-=job['cpus']
    data=dict(time=datetime.datetime.now().astimezone().isoformat(),user_total_limit=18,observed_jobs=rows,cancelled=cancelled,only_prefix='brd1_')
    tmp=HERE/'RESOURCE_STATUS.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(HERE/'RESOURCE_STATUS.json')
    return data
