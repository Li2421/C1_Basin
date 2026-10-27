"""Ordered authorized workflow; no COMPLETE until actual final audits and reports."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,subprocess,fcntl,signal
from resources import capacity,jobs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PY=sys.executable

def read(p):return json.loads(Path(p).read_text())
def write(p,r):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def call(script,*args):
    p=HERE/'driver_actions'/f'{time.time_ns()}_{script}.log';p.parent.mkdir(exist_ok=True)
    with p.open('w') as out:r=subprocess.run([PY,'-B',str(HERE/script),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((script,r.returncode,str(p),p.read_text()[-2500:]))
def ours():return [r for r in jobs() if r['user']==os.environ.get('USER','zhihan') and r['name']=='lit_supp_science']
def guard(label):
    try:call('guard_fast.py',label)
    except Exception:
        write(HERE/'FATAL_STOP.json',dict(utc=datetime.now(timezone.utc).isoformat(),reason='Read-only guard failed',label=label))
        for r in ours():subprocess.run(['scancel',r['id']],check=True)
        raise

def status(stage,iteration=0):
    folder=HERE/stage/'raw';count=sum(1 for _ in folder.glob('*.json')) if folder.exists() else 0
    out=dict(utc=datetime.now(timezone.utc).isoformat(),phase=stage,new_local_rows=count,jobs=ours(),recovery_iteration=iteration,part1_complete=True,cpu_total_cap=18)
    write(HERE/'PROGRESS.json',out);(HERE/'STATUS.md').write_text(f"第二部分执行中：{stage}；当前本地raw {count}。实际作业和时间见PROGRESS.json。尚未COMPLETE，后续依次能力门、旧场景粗筛、新场景主线及粗筛、统计审计交付。CPU总上限18含管理。\n")

def incomplete(stage):
    out=[]
    for e in read(HERE/'QUEUES'/stage/'INDEX.json'):
        cs=read(e['path'])
        if any(not (HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists() for c in cs):out.append(e)
    return out

def execute(stage):
    marker=HERE/f'{stage}_BATCH_DONE.json'
    if marker.exists():return
    if not (HERE/'QUEUES'/stage/'QUEUE.receipt.json').exists():call('queue_supplement.py',stage)
    for iteration in range(8):
        # Existing raw, including unknown, is never rerun.
        missing=incomplete(stage)
        if not missing:
            guard('after_'+stage);call('merge.py');write(marker,dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,completed=True,unknowns_not_retried=True));return
        assert not ours(),'Another owned science batch is still active'
        if iteration:
            folder=HERE/'QUEUES'/stage/'claims';archive=HERE/'QUEUES'/stage/'aborted_claims';archive.mkdir(exist_ok=True)
            for e in missing:
                p=folder/e['group']
                if p.exists():p.rename(archive/(e['group']+'_'+str(time.time_ns())))
        guard('before_'+stage+'_'+str(iteration))
        while True:
            n,before=capacity()
            if n:break
            status(stage,iteration);time.sleep(30)
        n=min(n,len(missing));job=subprocess.check_output(['sbatch','--parsable',f'--array=0-{n-1}',f'--export=ALL,SUPPLEMENT_STAGE={stage}',str(HERE/'worker.sbatch')],text=True).strip()
        p=HERE/'submissions'/f'{time.time_ns()}_{stage}_{job}.json';p.parent.mkdir(exist_ok=True);write(p,dict(utc=datetime.now(timezone.utc).isoformat(),job=job,workers=n,all_user_jobs_before=before,stage=stage))
        time.sleep(3)
        while ours():
            if (HERE/'FATAL_STOP.json').exists():raise RuntimeError('Read-only fatal stop')
            status(stage,iteration);time.sleep(30)
        guard('batch_end_'+stage+'_'+str(iteration))
    raise RuntimeError('Repeated infrastructure failures; preserved raw and claims for review: '+stage)

def ensure(stage,op,*args):
    if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():call('plan_stages.py',op,*args)
    execute(stage)

def main():
    with (HERE/'DRIVER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(HERE/'DRIVER_PID.json',dict(pid=os.getpid(),python=PY,entry=str(HERE/'driver.py')))
        execute('capability')
        if not (HERE/'QUALIFICATION.json').exists():call('plan_stages.py','qualify')
        for which in ('old',):
            ensure(which+'_fixed_train','fixed',which);ensure(which+'_pair_train','pairs',which);ensure(which+'_eval','eval',which)
        ensure('swap_main','main')
        for which in ('new',):
            ensure(which+'_fixed_train','fixed',which);ensure(which+'_pair_train','pairs',which);ensure(which+'_eval','eval',which)
        ensure('moving_eval','moving')
        write(HERE/'COMPUTATION_READY.json',dict(utc=datetime.now(timezone.utc).isoformat(),passed=True))
        status('statistics_pending')
        # Finalization is evidence-only; errors stop without writing COMPLETE.
        call('finish_supplement.py')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(HERE/'DRIVER_ERROR.json',dict(utc=datetime.now(timezone.utc).isoformat(),error=str(exc)));raise
