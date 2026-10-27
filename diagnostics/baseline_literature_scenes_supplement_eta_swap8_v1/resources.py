from pathlib import Path
import subprocess,os,json,time
HERE=Path(__file__).resolve().parent
CAP=18

def jobs():
    return [dict(zip(('id','user','name','state','cpus'),line.split('|'))) for line in subprocess.check_output(['squeue','-h','-r','-o','%i|%u|%j|%T|%C'],text=True).splitlines()]
def capacity():
    rows=jobs();mine=sum(int(r['cpus']) for r in rows if r['user']==os.environ.get('USER','zhihan'))
    return max(0,min(CAP-1-mine,24-1-sum(int(r['cpus']) for r in rows))),rows
if __name__=='__main__':
    (HERE/'RESOURCE_GUARD_PID.json').write_text(json.dumps(dict(pid=os.getpid(),cap=CAP,deadline_already_passed='2026-10-23T09:00:00+08:00')))
    while not (HERE/'COMPLETE.json').exists():
        rows=jobs();mine=[r for r in rows if r['user']==os.environ.get('USER','zhihan')];excess=sum(int(r['cpus']) for r in mine if r['state']=='RUNNING')+1-CAP;actions=[]
        for r in sorted(mine,key=lambda r:r['id'],reverse=True):
            if excess<=0:break
            if r['state']!='RUNNING' or not r['name'].startswith('lit_supp_'):continue
            detail=subprocess.check_output(['scontrol','show','job','-o',r['id']],text=True)
            if str(HERE) not in detail:continue
            subprocess.run(['scancel',r['id']],check=True);excess-=int(r['cpus']);actions.append(r['id'])
        with (HERE/'RESOURCE_STATUS.jsonl').open('a') as f:f.write(json.dumps(dict(utc=time.time(),cap=CAP,jobs=rows,cancelled=actions))+'\n')
        time.sleep(45)
