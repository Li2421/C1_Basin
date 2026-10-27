"""One fresh interpreter per paper episode, avoiding vendor global-state leaks."""
import os
os.environ.pop('STB_COORDINATOR',None)
from core import *
import subprocess,sys
if __name__=='__main__':
    stage,index=sys.argv[1],int(sys.argv[2]);cases=read(HERE/'queues'/f'{stage}.json')['shards'][index]
    for c in cases:
        if(HERE/'raw'/f"{c['rollout_uid']}.json").exists():continue
        path=HERE/'paper_requests'/f"{c['rollout_uid']}.json";dump(path,c)
        episode='paper_episode_impc_v2.py'if c['controller_payload'].get('impc_grid_adapter_v2')else'paper_episode.py'
        completed=subprocess.run([sys.executable,str(HERE/episode),str(path)],check=False)
        if completed.returncode and not(HERE/'raw'/f"{c['rollout_uid']}.json").exists():raise RuntimeError(f'Paper process exited {completed.returncode} before a complete raw result')
