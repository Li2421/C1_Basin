"""Keep single-core workers busy; exclusive UID locks prevent duplicate rollouts."""
import runner
from core import *
from budget import reserve
import fcntl,os

def worker_main(stage,index):
    os.environ.pop('STB_COORDINATOR',None)
    private=read(HERE/'queues'/f'{stage}.json')['shards'][index]
    # Private names append the array index to the complete archived queue name.
    full=HERE/'queues'/f"{stage.rsplit('_',1)[0]}.json"
    all_shards=read(full)['shards']
    local_ids={c['rollout_uid']for c in private}
    additional=[c for shard in all_shards for c in shard if c['rollout_uid']not in local_ids]
    additional.sort(key=lambda c:(-c['scenario_payload']['config']['max_steps']*c['state']['N']**1.5,c['rollout_uid']))
    folder=HERE/'case_locks';folder.mkdir(exist_ok=True)
    for c in private+additional:
        uid=c['rollout_uid'];raw=HERE/'raw'/f'{uid}.json'
        if raw.exists():continue
        with(folder/f'{uid}.lock').open('a+')as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if raw.exists():continue
            reserve(c)
            r=runner.work(c)
            print(stage,c['scene'],c['controller_payload']['nominal']['type_code'],c['state_index'],
                  c['solo'],r['outcome'],round(r['wall_seconds'],3),'uid_lock',uid,flush=True)
