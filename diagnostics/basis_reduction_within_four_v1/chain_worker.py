"""Run consecutive frozen eta requests in one process; stop at first success."""
import os
os.environ.pop('BRD_COORDINATOR',None)
import runner
from core import *
from budget import reserve,used
from chain_logic import walk
import fcntl
import sys


def main(queue_name,index):
    queue=read(HERE/'queues'/f'{queue_name}.json')
    drivers=queue['shards'][index]
    if not drivers:return
    stages={d['_chain_stage']for d in drivers};assert len(stages)==1
    stage=stages.pop();plan=read(HERE/'chain_plans'/f'{stage}.json')
    accelerated=plan['accelerated']
    if accelerated:
        from acceleration import enable
        enable(check=False)
    from recording_fix import install
    install()
    locks=HERE/'case_locks';locks.mkdir(exist_ok=True)
    group_locks=HERE/'chain_group_locks';group_locks.mkdir(exist_ok=True)
    own_ids={d['rollout_uid']for d in drivers}
    local=[g for g in plan['groups']if g['driver_uid']in own_ids]
    others=[g for g in plan['groups']if g['driver_uid']not in own_ids]
    others.sort(key=lambda g:(-g['estimated_load'],g['group_id']))
    for group in local+others:
        finished=HERE/'chain_progress'/stage/(group['group_id']+'_finished.json')
        if finished.exists()and read(finished).get('complete'):continue
        with(group_locks/(group['group_id']+'.lock')).open('a+')as group_lock:
            try:fcntl.flock(group_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if finished.exists()and read(finished).get('complete'):continue
            progress=[]
            def evaluate(item):
                request=item['request'];uid=item['query_uid'];assert request['rollout_uid']==uid
                assert bool(request['controller_payload'].get('exact_acceleration'))==accelerated
                if uid in plan['known']:return plan['known'][uid]
                raw=HERE/'raw'/f'{uid}.json'
                with(locks/(uid+'.lock')).open('a+')as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    if raw.exists():return read(raw)
                    # Priority reservations are checked atomically without
                    # changing the frozen hard-budget implementation.
                    with(HERE/'chain_priority_budget.lock').open('a+')as budget_lock:
                        fcntl.flock(budget_lock,fcntl.LOCK_EX)
                        if used()>=plan['ceiling']:return None
                        reserve(request)
                    result=runner.work(request)
                    print(stage,group['identity'],item['index'],result['outcome'],round(result['wall_seconds'],3),flush=True)
                    return result
            def record(item,result):
                progress.append(dict(index=item['index'],query_uid=item['query_uid'],success=bool(result['success'])))
                dump(HERE/'chain_progress'/stage/(group['group_id']+'.json'),dict(identity=group['identity'],records=progress))
            complete=walk(group['items'],evaluate,record)
            dump(HERE/'chain_progress'/stage/(group['group_id']+'_finished.json'),dict(identity=group['identity'],complete=complete,
                 last_index=progress[-1]['index']if progress else None,budget_limited=not complete))


if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
