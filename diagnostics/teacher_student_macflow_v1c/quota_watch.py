"""Adjust only existing array throttle to observed free CPU quota; never submit trials."""
from core import *
import subprocess,time,re

def main():
    records=[];known={}
    while True:
        status=read(HERE/'RUN_STATUS.json')
        if status['status'] in ['COMPLETE','STOPPED']:break
        job=status.get('job')
        if not job:time.sleep(20);continue
        lines=subprocess.check_output(['squeue','-r','-h','-u',os.environ['USER'],'-o','%i|%T|%C'],text=True).splitlines();rows=[]
        for line in lines:
            jid,state,cpu=line.split('|');rows.append(dict(job=jid,state=state,cpus=int(cpu),own=jid.split('_')[0]==job))
        own=[r for r in rows if r['own']];other=sum(r['cpus'] for r in rows if not r['own'] and r['state'] in ['RUNNING','COMPLETING','CONFIGURING']);target=max(1,min(18,20-other))
        if own:
            probe=subprocess.run(['scontrol','show','job',own[0]['job'],'-o'],capture_output=True,text=True);match=re.search(r'ArrayTaskThrottle=(\d+)',probe.stdout);current=int(match.group(1)) if match else known.get(job)
            if current is not None and target!=current:
                cmd=['scontrol','update','JobId='+job,'ArrayTaskThrottle='+str(target)];out=subprocess.run(cmd,capture_output=True,text=True);readback=subprocess.run(['scontrol','show','job',own[0]['job'],'-o'],capture_output=True,text=True);m=re.search(r'ArrayTaskThrottle=(\d+)',readback.stdout);actual=int(m.group(1)) if m else None;record=dict(utc=utc(),job=job,other_cpu=other,previous=current,target=target,returncode=out.returncode,stderr=out.stderr,actual_throttle=actual,scope='existing array only; no new rollout requests or resubmissions');records.append(record);write(HERE/'RESOURCE_ADJUSTMENTS.json',records)
                if actual is not None:assert actual==target,record
                known[job]=actual
        time.sleep(25)
    write(HERE/'QUOTA_WATCH_COMPLETE.json',dict(utc=utc(),adjustments=len(records),final_status=status['status']))
if __name__=='__main__':main()
