"""Latest user cap: 24 until 09:00 Singapore, then 18; one management share."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json,time,sys
HERE=Path(__file__).resolve().parent
DEADLINE=datetime(2026,10,23,1,0,tzinfo=timezone.utc)
def cap():return 18 if datetime.now(timezone.utc)>=DEADLINE else 24
def jobs():
    rows=[]
    for line in subprocess.check_output(['squeue','-h','-r','-o','%i|%u|%j|%T|%C'],text=True).splitlines():
        jid,user,name,state,cpu=line.split('|');rows.append(dict(id=jid,user=user,name=name,state=state,cpus=int(cpu)))
    return rows
def capacity():
    rows=jobs();user=os.environ.get('USER','zhihan');own=sum(r['cpus'] for r in rows if r['user']==user)
    return max(0,min(cap()-1-own,24-1-sum(r['cpus'] for r in rows))),rows
def guard():
    (HERE/'RESUME_RESOURCE_GUARD_PID.json').write_text(json.dumps(dict(pid=os.getpid(),deadline=DEADLINE.isoformat())))
    while not (HERE/'COMPLETE.json').exists():
        rows=jobs();own=[r for r in rows if r['user']==os.environ.get('USER','zhihan')]
        excess=sum(r['cpus'] for r in own if r['state']=='RUNNING')+1-cap();actions=[]
        if excess>0:
            for r in sorted(own,key=lambda x:x['id'],reverse=True):
                if excess<=0:break
                if r['state']!='RUNNING' or not r['name'].startswith('lit_resume_'):continue
                detail=subprocess.check_output(['scontrol','show','job','-o',r['id']],text=True)
                if str(HERE) not in detail:continue
                subprocess.run(['scancel',r['id']],check=True);excess-=r['cpus'];actions.append(r['id'])
        with (HERE/'RESOURCE_GUARD_RESUME.jsonl').open('a') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),cap=cap(),jobs=rows,cancelled_for_user_release=actions))+'\n')
        time.sleep(min(45,max(.5,(DEADLINE-datetime.now(timezone.utc)).total_seconds())) if datetime.now(timezone.utc)<DEADLINE else 45)
if __name__=='__main__':guard()
