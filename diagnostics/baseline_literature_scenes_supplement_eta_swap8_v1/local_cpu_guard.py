"""Keep the one local management share off other users' allocated CPUs."""
from pathlib import Path
import os,subprocess,time,json,re
HERE=Path(__file__).resolve().parent

def cpus(spec):
    out=set()
    for piece in spec.split(','):
        if '-' in piece:
            a,b=piece.split('-');out.update(range(int(a),int(b)+1))
        elif piece:out.add(int(piece))
    return out

def pin():
    rows=subprocess.check_output(['squeue','-h','-r','-o','%i|%u|%j|%T'],text=True).splitlines();used=set();verified_own=set()
    for line in rows:
        jid,user,name,state=line.split('|')
        if state not in ('RUNNING','COMPLETING'):continue
        p=subprocess.run(['scontrol','show','job','-d','-o',jid],capture_output=True,text=True)
        if p.returncode:continue
        allocated=set()
        for spec in re.findall(r'CPU_IDs=([^ ]+)',p.stdout):allocated|=cpus(spec)
        used|=allocated
        if user=='zhihan' and name.startswith('lit_supp_') and str(HERE) in p.stdout:verified_own|=allocated
    free=set(range(24))-used
    mode='unallocated';choices=free
    if not choices:choices=verified_own;mode='share_verified_own_worker'
    if not choices:return
    cpu=max(choices);parents={};commands={}
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,ppid=,args='],text=True).splitlines():
        cols=line.strip().split(None,2)
        if len(cols)==3:parents[int(cols[0])]=int(cols[1]);commands[int(cols[0])]=cols[2]
    roots={os.getpid()}
    for name,script in [('DRIVER_PID.json','driver.py'),('RESOURCE_GUARD_PID.json','resources.py')]:
        p=HERE/name
        if p.exists():
            pid=json.loads(p.read_text())['pid']
            if str(HERE/script) in commands.get(pid,''):roots.add(pid)
    targets=set(roots)
    while True:
        add={pid for pid,ppid in parents.items() if ppid in targets}-targets
        if not add:break
        targets|=add
    bound=[]
    for pid in targets:
        try:os.sched_setaffinity(pid,{cpu});bound.append(pid)
        except ProcessLookupError:pass
    with (HERE/'LOCAL_CPU_AFFINITY.jsonl').open('a') as f:f.write(json.dumps(dict(utc=time.time(),cpu=cpu,mode=mode,unallocated=sorted(free),verified_own=sorted(verified_own),pids=bound))+'\n')
if __name__=='__main__':
    (HERE/'LOCAL_CPU_GUARD_PID.json').write_text(json.dumps(dict(pid=os.getpid())))
    while not (HERE/'COMPLETE.json').exists():
        try:pin()
        except Exception as e:
            with (HERE/'LOCAL_CPU_AFFINITY_ERRORS.jsonl').open('a') as f:f.write(json.dumps(dict(utc=time.time(),error=str(e)))+'\n')
        time.sleep(30)
