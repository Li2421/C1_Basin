"""User-mandated 09:00 Singapore six-CPU release, persistent and auditable."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent
DEADLINE=datetime(2026,10,23,1,0,tzinfo=timezone.utc)
def limit():return 18 if datetime.now(timezone.utc)>=DEADLINE else 24
def snapshot():
    r=subprocess.check_output(['squeue','-u',os.environ.get('USER','zhihan'),'-h','-r','-o','%i|%j|%T|%C'],text=True)
    out=[]
    for line in r.splitlines():
        jid,name,state,cpus=line.split('|');out.append(dict(job=jid,name=name,state=state,cpus=int(cpus)))
    return out
def save(name,value):
    p=HERE/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
    save('CPU_RELEASE_GUARD_PID.json',dict(pid=os.getpid(),deadline_utc=DEADLINE.isoformat(),deadline_singapore='2026-10-23 09:00:00 +08:00'))
    while True:
        now=datetime.now(timezone.utc);jobs=snapshot();budget=limit();allocated=sum(j['cpus'] for j in jobs if j['state']=='RUNNING')
        actions=[]
        if now>=DEADLINE:
            # Keep one local management share. Never cancel another experiment.
            excess=max(0,allocated+1-budget)
            own=sorted((j for j in jobs if j['state']=='RUNNING' and j['name'].startswith('stage2_fix_')),key=lambda j:j['job'],reverse=True)
            for j in own:
                if excess<=0:break
                detail=subprocess.check_output(['scontrol','show','job','-o',j['job']],text=True)
                if str(HERE) not in detail:continue
                subprocess.run(['scancel',j['job']],check=True)
                actions.append(dict(job=j['job'],cpus=j['cpus'],reason='User-authorized six-CPU release; immutable completed raw/journals retained; unfinished requests recover under 18-share cap'))
                excess-=j['cpus']
            if actions or not (HERE/'CPU_RELEASE_EXECUTED.json').exists():
                save('CPU_RELEASE_EXECUTED.json',dict(utc=now.isoformat(),budget=budget,management=1,maximum_workers=17,actions=actions,remaining_excess=excess))
            if not any(j['name'].startswith('stage2_fix_') for j in jobs):return
        save('RESOURCE_STATUS.json',dict(utc=now.isoformat(),budget=budget,maximum_workers=budget-1,management=1,jobs=jobs,deadline_utc=DEADLINE.isoformat()))
        time.sleep(min(45,max(.5,(DEADLINE-now).total_seconds())) if now<DEADLINE else 45)
if __name__=='__main__':main()
