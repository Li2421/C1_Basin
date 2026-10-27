"""Parallelize unstarted pilot requests while current episodes finish unchanged."""
from common import *
from schedule import jobs,log,merge
from queueing import cost,build
import subprocess
import signal

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def present(c):
    return any((HERE/folder/'raw'/f"{c['rollout_uid']}.json").exists()for folder in ('pilot','pilot_tail'))


def main():
    cases=read(HERE/'pilot_CASES.json')
    if all(present(c)for c in cases):return
    # Stop only the polling coordinator, never a currently executing episode.
    for suffix in ('diagnostics/cross_scene_correction_v1/advance.py',str(HERE/'advance.py'),str(HERE/'schedule.py')+' pilot'):
        found=subprocess.run(['pgrep','-f','^'+PYTHON+' -B '+suffix+'$'],capture_output=True,text=True)
        for pid in found.stdout.split():os.kill(int(pid),signal.SIGTERM)
    old=[r for r in jobs()if r['name']=='xsc_pilot']
    for r in old:
        subprocess.run(['scancel','--signal=USR1','--full',r['id']],check=True)
    # STOP is checked only after the full current request is saved. Its first
    # absent result is conservatively deferred; later requests cannot start.
    deferred=set();owners=[]
    folder=HERE/'QUEUES/pilot'
    live={r['id']for r in old}
    for owner_path in (folder/'claims').glob('*/OWNER.json'):
        owner=read(owner_path);gid=owner_path.parent.name
        if owner['job']not in live or(folder/'done'/gid).exists():continue
        unfinished=[c for c in read(folder/'groups'/f'{gid}.json')if not present(c)]
        if unfinished:deferred.add(unfinished[0]['rollout_uid']);owners.append(owner)
    tail=[c for c in cases if not present(c)and c['rollout_uid']not in deferred]
    write(HERE/'PILOT_TAIL_REBALANCE.json',dict(utc=utc(),unchanged_request_set_sha256=uid('requests',[c['rollout_uid']for c in cases]),
        deferred_current_requests=sorted(deferred),original_owners=owners,parallel_unstarted_requests=[c['rollout_uid']for c in tail],
        running_episodes_not_interrupted=True,reason='Avoid serial waiting behind long episodes; scientific inputs unchanged.'))
    if tail:
        todo,reused=register('pilot_tail',tail)
        q=HERE/'QUEUES/pilot_tail';index=[]
        for i,c in enumerate(sorted(todo,key=lambda c:(-cost(c),c['rollout_uid']))):
            p=q/'groups'/f'{i:06d}.json';write(p,[c])
            index.append(dict(group=f'{i:06d}',path=str(p),sha256=sha(p),count=1,estimated_seconds=cost(c)))
        write(q/'INDEX.json',index)
        write(q/'QUEUE.receipt.json',dict(utc=utc(),stage='pilot_tail',requests=len(todo),groups=len(index),
            preflight_sha256=sha(HERE/'pilot_tail_PREFLIGHT.receipt.json'),index_sha256=sha(q/'INDEX.json')))
        if index:
            command=['sbatch','--parsable','--job-name=xsc_pilot_tail',f'--array=0-{len(index)-1}',
                '--export=ALL,CROSS_SCENE_STAGE=pilot_tail,CROSS_SCENE_TIME_LIMIT_S=10800',str(HERE/'worker.sbatch')]
            job=subprocess.check_output(command,text=True).strip().split(';')[0]
            log('pilot_tail_submitted',job=job,requests=len(index),original_episodes=len(old))
            observed=[]
            for _ in range(20):
                observed=[r for r in jobs()if r['id'].split('_')[0]==job]
                if observed and all(r['state']=='RUNNING'for r in observed):break
                time.sleep(1)
            log('pilot_tail_immediate_state_check',job=job,states=observed)
    started=time.time();checked=False;last_count=-1;last_progress=time.time()
    while True:
        count=sum(present(c)for c in cases)
        live=[r for r in jobs()if r['name']in ('xsc_pilot','xsc_pilot_tail')]
        if count!=last_count:last_count=count;last_progress=time.time()
        write(HERE/'PROGRESS.json',dict(utc=utc(),stage='pilot',new_results=count,running=sum(r['state']=='RUNNING'for r in live),
            pending=sum(r['state']=='PENDING'for r in live),quota=17,last_new_result_age_seconds=time.time()-last_progress,
            tail_rebalanced=True,original_episodes_finish_without_restart=True))
        if not checked and time.time()-started>=180:
            log('pilot_tail_three_minute_state_check',states=live);checked=True
        if count==len(cases)and not live:break
        if not live and count<len(cases):raise RuntimeError(('PILOT_TAIL_INCOMPLETE',count,len(cases)))
        if time.time()-last_progress>1800:raise RuntimeError('PILOT_TAIL_NO_RESULT_30MIN: supervisor intervention required')
        time.sleep(20)
    # All writers have exited. A single official merger handles the entire pilot.
    merge('pilot')
    archive=HERE/'pilot_tail_history'/str(time.time_ns());archive.mkdir(parents=True)
    for name in ('pilot_CACHE_PREFLIGHT.json','pilot_MISSING_CASES.json','pilot_REUSED_ROWS.json','pilot_PREFLIGHT.receipt.json'):
        p=HERE/name
        if p.exists():p.rename(archive/name)
    (HERE/'QUEUES/pilot').rename(archive/'original_queue')
    todo,reused=register('pilot',cases);assert not todo
    build('pilot')
    write(HERE/'PILOT_TAIL_COMPLETED.json',dict(utc=utc(),requests=len(cases),all_reused_after_merge=True,
        original_queue_archive=str(archive),no_completed_request_reexecuted=True))
    # Resume the normal full pipeline with the additional result folder visible.
    subprocess.run([PYTHON,'-B',str(HERE/'advance.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(HERE/'PILOT_TAIL_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise
