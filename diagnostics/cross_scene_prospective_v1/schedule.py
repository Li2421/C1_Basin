"""Quota-limited scheduler, RUNNING readbacks, and deadline-only handovers."""
from common import *
from queueing import build,complete
import subprocess,fcntl,re

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'

def log(event,**kw):
    with(HERE/'SCHEDULER_EVENTS.jsonl').open('a')as f:f.write(canonical(dict(utc=utc(),event=event,**kw))+'\n')

def jobs():
    s=subprocess.check_output(['squeue','-h','-r','-a','-o','%i|%u|%T|%C|%M|%R|%j|%Z'],text=True)
    return [dict(zip(('id','user','state','cpus','elapsed','reason','name','cwd'),line.split('|',7)))for line in s.splitlines()if len(line.split('|',7))==8]

def cancel(job,graceful=False):
    proc=subprocess.run(['scancel']+(['--signal=USR1','--full']if graceful else[])+[job],capture_output=True,text=True)
    if proc.returncode and any(r['id']==job for r in jobs()):raise RuntimeError(proc.stderr)

def quota(rows):
    own_other=sum(int(r['cpus'])for r in rows if r['user']=='zhihan' and r['state']=='RUNNING' and not r['name'].startswith('xsp_'))
    all_other=sum(int(r['cpus'])for r in rows if r['state']=='RUNNING' and not r['name'].startswith('xsp_'))
    total=int(subprocess.check_output(['sinfo','-N','-h','-o','%c'],text=True).splitlines()[0])
    return max(0,min(17-own_other,total-all_other-1))

def pressure_limit(rows,stage,base):
    pending=[r for r in rows if r['user']!='zhihan'and r['state']=='PENDING'and any(k in r['reason']for k in ('Resources','Priority'))]
    if not pending:return base
    node=subprocess.check_output(['scontrol','show','node','-d','-o'],text=True)
    def num(k):
        x=re.search(r'\b'+k+r'=(\d+)',node);return int(x.group(1))if x else 0
    total,alloc,mem,used=[num(k)for k in ('CPUTot','CPUAlloc','RealMemory','AllocMem')]
    ours=sum(int(r['cpus'])for r in rows if r['name'].startswith('xsp_')and r['state']=='RUNNING')
    def resource(s,name):
        m=re.search(r'(?:^|,)'+name+r'(?::[^:,()]+)?:(\d+)',s);return int(m.group(1))if m else 0
    avail=re.search(r'\bGres=([^\s]+)',node);busy=re.search(r'\bGresUsed=([^\s]+)',node)
    av=avail.group(1)if avail else '';bu=busy.group(1)if busy else ''
    for r in pending:
        details=subprocess.check_output(['scontrol','show','job',r['id'],'-o'],text=True)
        gres=re.search(r'\bTresPerNode=([^\s]+)',details);req=gres.group(1).replace('gres/','')if gres else ''
        shards=resource(req,'shard');gpus=resource(req,'gpu')
        if shards and (resource(bu,'gpu')or shards>resource(av,'shard')-resource(bu,'shard')):continue
        if gpus and (resource(bu,'shard')or gpus>resource(av,'gpu')-resource(bu,'gpu')):continue
        cpus=int(r['cpus']);m=re.search(r'\bReqTRES=[^\s]*?mem=([0-9.]+)([KMGT]?)',details)
        need=float(m.group(1))*{'K':1/1024,'M':1,'G':1024,'T':1048576,'':1}[m.group(2)]if m else 0
        if cpus<=total-alloc and need<=mem-used:continue
        if cpus>total-(alloc-ours)or need>mem-(used-ours*3072):continue
        target=min(base,max(0,total-(alloc-ours)-cpus-1),max(0,int((mem-(used-ours*3072)-need-1024)//3072)))
        log('actual_other_cpu_memory_need',job=r['id'],target=target,cpus=cpus,memory_mb=need);return target
    return base

def release_stale(stage,live):
    for claim in(HERE/'QUEUES'/stage/'claims').glob('*'):
        if not claim.is_dir()or not(claim/'OWNER.json').exists():continue
        owner=read(claim/'OWNER.json')
        if owner['job']in live:continue
        check=subprocess.run(['squeue','-h','-j',str(owner['job']),'-o','%T'],capture_output=True,text=True)
        if check.returncode==0 and check.stdout.strip():continue
        if(HERE/'QUEUES'/stage/'done'/claim.name).exists():continue
        target=HERE/'QUEUES'/stage/'abandoned'/f'{claim.name}_{time.time_ns()}';target.parent.mkdir(parents=True,exist_ok=True)
        claim.rename(target);log('released_finished_allocation_claim',stage=stage,owner=owner)

def merge(stage):
    with(HERE/'MERGER.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=subprocess.run([PYTHON,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,capture_output=True,text=True)
        write(HERE/f'{stage}_MERGE.json',dict(utc=utc(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        assert p.returncode==0,p.stderr
    log('merged',stage=stage)

def run(stage):
    build(stage);submitted={};first_seen={};limit=10800;log('started',stage=stage)
    while not complete(stage):
        fatal=list((HERE/'WORKER_FATAL').glob('*.json'))
        if fatal:raise RuntimeError(('WORKER_FAILURE',str(fatal[0])))
        rows=jobs();ours=[r for r in rows if r['name']==f'xsp_{stage}'];live={r['id']for r in rows};release_stale(stage,live)
        cap=pressure_limit(rows,stage,quota(rows));running=[r for r in ours if r['state']=='RUNNING'];pending=[r for r in ours if r['state']=='PENDING']
        for r in rows:first_seen.setdefault(r['id'],time.time())
        for r in pending:
            if time.time()-first_seen[r['id']]>900 and len(running)<cap:
                cancel(r['id']);limit=max(3360,limit//2);log('pending_15min_shorten',job=r['id'],seconds=limit)
        for r in running[cap:]:cancel(r['id'],True);log('yield_actual_cpu_memory_pressure',job=r['id'])
        unfinished=sum(not(HERE/'QUEUES'/stage/'done'/e['group']).exists()for e in read(HERE/'QUEUES'/stage/'INDEX.json'))
        desired=max(0,min(cap-len(ours),unfinished-len(ours)))
        if desired:
            command=['sbatch','--parsable',f'--job-name=xsp_{stage}',f'--array=0-{desired-1}',f'--time={limit//3600:02d}:{limit%3600//60:02d}:00',
                f'--export=ALL,XSP_STAGE={stage},XSP_TIME_LIMIT_S={limit}',str(HERE/'worker.sbatch')]
            job=subprocess.check_output(command,text=True).strip().split(';')[0];submitted[job]=time.time();log('submitted',stage=stage,job=job,count=desired,seconds=limit)
            observed=[]
            for _ in range(20):
                observed=[r for r in jobs()if r['id'].split('_')[0]==job]
                if observed and all(r['state']=='RUNNING'for r in observed):break
                time.sleep(1)
            log('immediate_RUNNING_readback',job=job,states=observed)
        for job,t in list(submitted.items()):
            if time.time()-t>=180:
                log('three_minute_RUNNING_readback',job=job,states=[r for r in rows if r['id'].split('_')[0]==job]);del submitted[job]
        write(HERE/'PROGRESS.json',dict(utc=utc(),stage=stage,new_results=sum(1 for _ in(HERE/stage/'raw').glob('*.json')),
            groups_remaining=unfinished,running=len(running),pending=len(pending),worker_cap=cap,total_cpu_cap=18,gpu_shares=0,
            no_new_result_timeout_rule=False))
        time.sleep(30)
    for r in jobs():
        if r['name']==f'xsp_{stage}':cancel(r['id'])
    merge(stage);log('completed',stage=stage)

if __name__=='__main__':run(sys.argv[1])
