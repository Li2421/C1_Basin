"""One isolated interpreter per paper episode, with cross-shard UID locking."""
import os
os.environ.pop('STB_COORDINATOR',None)
from core import *
import fcntl,subprocess,sys

def main(stage,index):
    mine=read(HERE/'queues'/f'{stage}.json')['shards'][index]
    full=read(HERE/'queues'/f"{stage.rsplit('_',1)[0]}.json")['shards']
    ids={c['rollout_uid']for c in mine}
    rest=[c for shard in full for c in shard if c['rollout_uid']not in ids]
    rest.sort(key=lambda c:(-c['scenario_payload']['config']['max_steps']*c['state']['N']**1.5,c['rollout_uid']))
    folder=HERE/'case_locks';folder.mkdir(exist_ok=True)
    for c in mine+rest:
        uid=c['rollout_uid'];raw=HERE/'raw'/f'{uid}.json'
        if raw.exists():continue
        with(folder/f'{uid}.lock').open('a+')as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if raw.exists():continue
            path=HERE/'paper_requests'/f'{uid}.json';dump(path,c)
            env=dict(os.environ);env.pop('STB_COORDINATOR',None)
            child=subprocess.run([sys.executable,c['controller_payload']['paper_episode_entry'],str(path)],env=env,check=False)
            if child.returncode and not raw.exists():raise RuntimeError(f'Paper subprocess exited {child.returncode} before saving {uid}')

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
