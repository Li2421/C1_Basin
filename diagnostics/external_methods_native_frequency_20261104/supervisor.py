"""Durable scheduler sharing the user's global 18-core envelope.

The agent must still inspect logs/progress and certify pilots before opening the
full queue. Failed interpreters are recorded once, never silently rerolled.
"""
from native_io import *
import ast,subprocess,signal,collections
POOL=list(range(5,17))+list(range(18,24));os.sched_setaffinity(0,{23});HZ=os.sysconf('SC_CLK_TCK')
source=ROOT/'diagnostics/c1_v3_mcbf_circulation_paired_20261104/resource_supervisor.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['processes','pin']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'))

if __name__=='__main__':
    cases=read(D/'ACTIVE_NATIVE_CASES.json');byuid={c['rollout_uid']:c for c in cases}
    pilots={m:min((c for c in cases if c['payload']['method']==m),key=lambda c:(len(c['state']['positions']),c['scenario']['config']['max_steps']*c['scenario']['config']['dt'],c['name']))['rollout_uid'] for m in ['dac','mgr','mcbf','circulation']}
    dump(D/'PILOT_SELECTION.json',dict(pilots=pilots,rule='smallest agent count then shortest physical horizon then name; no selection by outcome'))
    order=sorted(cases,key=lambda c:(c['rollout_uid'] not in pilots.values(),c['payload']['physical_key'],c['payload']['method']))
    active={};children={};history=read(D/'SUPERVISOR_HISTORY.json') if (D/'SUPERVISOR_HISTORY.json').exists() else []
    for pid,p in processes().items():
        if str(D/'worker.py') in p['args']:
            uid=next((u for u in byuid if u in p['args']),None)
            if uid:active[pid]=dict(uid=uid,cpu=p['affinity'][0],paused=p['state']=='T',start=time.time(),adopted=True)
    previous=processes();last=time.monotonic();last_slurm=0.;reserved=0
    while True:
        time.sleep(2);now=time.monotonic();current=processes();external=[];load=collections.defaultdict(float)
        for pid,p in current.items():
            if pid in active or pid==os.getpid() or pid not in previous or p['state']=='Z':continue
            cpu=max(0.,(p['ticks']-previous[pid]['ticks'])/HZ/(now-last))
            if cpu>=.15:
                external.append(dict(pid=pid,cpu=cpu,affinity=p['affinity'],args=p['args'][:200]))
                for k in p['affinity']:load[k]+=cpu/len(p['affinity'])
        if now-last_slurm>15:
            q=subprocess.run(['squeue','-u',os.environ.get('USER','zhihan'),'--noheader','--format=%C'],capture_output=True,text=True)
            reserved=sum(int(x) for x in q.stdout.split()) if q.returncode==0 else 18;last_slurm=now
        used=sum(p['cpu'] for p in external);capacity=max(0,min(18,18-max(reserved,math.ceil(max(0.,used-.1)))))
        for pid,meta in list(active.items()):
            child=children.get(pid);ended=child.poll() is not None if child else pid not in current or current[pid]['state']=='Z'
            if ended:
                path=D/'raw'/(meta['uid']+'.json');r=read(path) if path.exists() else {}
                history.append(dict(**meta,pid=pid,result_exists=path.exists(),outcome=r.get('outcome'),exit_code=child.returncode if child else None,finished=time.time()))
                del active[pid];children.pop(pid,None);print('finished',meta['uid'],r.get('outcome'),flush=True)
        available=sorted(POOL,key=lambda c:(load[c],c));assigned=set();running=list(active)[:capacity]
        for pid,meta in active.items():
            if pid in running:
                cpu=next(c for c in available if c not in assigned);assigned.add(cpu);pin(pid,cpu);meta['cpu']=cpu
                if meta['paused']:os.kill(pid,signal.SIGCONT);meta['paused']=False
            elif not meta['paused']:os.kill(pid,signal.SIGSTOP);meta['paused']=True
        failed={h['uid'] for h in history if not h['result_exists'] or h['outcome']=='unknown'}
        gate=(D/'PILOT_CERTIFICATE.json').exists() and read(D/'PILOT_CERTIFICATE.json').get('passed')
        pending=[c for c in order if not (D/'raw'/(c['rollout_uid']+'.json')).exists() and c['rollout_uid'] not in {v['uid'] for v in active.values()} and c['rollout_uid'] not in failed]
        eligible=[c for c in pending if gate or c['rollout_uid'] in pilots.values()]
        while eligible and len(active)<capacity:
            c=eligible.pop(0);cpu=next(k for k in available if k not in assigned);assigned.add(cpu)
            assert sha(c['payload']['execution_profile'])==c['payload']['execution_profile_sha256']
            log=(D/'logs'/(c['name']+'.log')).open('a')
            cmd=[c['payload']['python'],'-u','-B',str(D/'worker.py'),str(D/'tasks'/(c['rollout_uid']+'.json')),'--output-root',str((D/'raw').resolve().parent)]
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,BASELINE_CPU=str(cpu)));log.close()
            active[p.pid]=dict(uid=c['rollout_uid'],cpu=cpu,paused=False,start=time.time(),adopted=False);children[p.pid]=p
            print('started',c['name'],'cpu',cpu,flush=True)
        completed=sum((D/'raw'/(c['rollout_uid']+'.json')).exists() for c in cases)
        dump(D/'RESOURCE_STATUS.json',dict(time=time.time(),shared_cpu_limit=18,pool=POOL,other_observed_cpu=used,slurm_reserved=reserved,our_capacity=capacity,active=active,completed=completed,total=len(cases),pending=len(pending),pilot_gate=gate,other_jobs=external))
        dump(D/'SUPERVISOR_HISTORY.json',history)
        if not pending and not active:
            dump(D/'SUPERVISOR_EXIT.json',dict(exit_code=0 if not failed else 1,completed=completed,total=len(cases),failed=sorted(failed)));break
        previous=current;last=now
