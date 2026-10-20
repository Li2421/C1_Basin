from common import *
import subprocess,time,signal,re

def main():
    paused=set();previous=read(HERE/'RESOURCE_MONITOR.json') if (HERE/'RESOURCE_MONITOR.json').exists() else {};events=previous.get('events',[]);samples=previous.get('samples',[]);tick=0;throttles={}
    while True:
        status=read(HERE/'RUN_STATUS.json') if (HERE/'RUN_STATUS.json').exists() else {};stop=status.get('status')=='COMPLETE'
        owned={read(p)['job'] for p in HERE.glob('*_SUBMISSION.json') if 'job' in read(p)};rows=[]
        for line in subprocess.check_output(['squeue','-r','-h','-o','%i|%u|%T|%C'],text=True).splitlines():
            job,user,state,cpu=line.split('|');rows.append(dict(job=job,user=user,state=state,cpus=int(cpu),own=job.split('_')[0] in owned))
        active=[r for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']];others=sum(r['cpus'] for r in active if not r['own']);slots=max(0,18-others);processes=[]
        for line in subprocess.check_output(['ps','-u',os.environ['USER'],'-o','pid=,args='],text=True).splitlines():
            fields=line.split()
            if len(fields)>=6 and fields[1].endswith('/python') and 'diagnostics/blackbox_addendum_v1/worker.py' in fields:processes.append(int(fields[0]))
        for job in sorted({r['job'].split('_')[0] for r in rows if r['own']}):
            limit=max(1,min(18,slots))
            if throttles.get(job)!=limit:
                update=subprocess.run(['scontrol','update',f'JobId={job}',f'ArrayTaskThrottle={limit}'],capture_output=True,text=True)
                if update.returncode==0:throttles[job]=limit;events.append(dict(utc=utc(),event='set own array concurrency',job=job,limit=limit,other_cpu=others))
        live=set(processes);paused&=live;want=set(sorted(processes)[slots:])
        for pid in sorted(want-paused):
            try:os.kill(pid,signal.SIGSTOP);paused.add(pid);events.append(dict(utc=utc(),event='pause own worker',pid=pid,other_cpu=others,allocated_total=sum(r['cpus'] for r in active)))
            except ProcessLookupError:pass
        for pid in sorted(paused-want):
            try:os.kill(pid,signal.SIGCONT);events.append(dict(utc=utc(),event='resume own worker',pid=pid,other_cpu=others));paused.remove(pid)
            except ProcessLookupError:paused.discard(pid)
        if tick%15==0 or stop:
            samples.append(dict(utc=utc(),allocated_total=sum(r['cpus'] for r in active),own_allocated=sum(r['cpus'] for r in active if r['own']),other_cpu=others,own_processes=len(processes),own_paused=len(paused),own_unpaused=len(processes)-len(paused),known_actual_total=others+len(processes)-len(paused),queue=rows));write(HERE/'RESOURCE_MONITOR.json',dict(samples=samples,events=events,cap=18,max_GPU_shards=2,mechanism='Only own processes pause with in-flight state preserved; array allocations may precede governor response, recorded explicitly'))
        if stop:break
        tick+=1;time.sleep(2)
if __name__=='__main__':main()
