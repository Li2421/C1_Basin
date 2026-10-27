"""One fresh interpreter per paper episode, avoiding vendor global-state leaks."""
from core import *
import subprocess,sys
if __name__=='__main__':
    stage,index=sys.argv[1],int(sys.argv[2]);cases=read(HERE/'queues'/f'{stage}.json')['shards'][index]
    for c in cases:
        if(HERE/'raw'/f"{c['rollout_uid']}.json").exists():continue
        path=HERE/'paper_requests'/f"{c['rollout_uid']}.json";dump(path,c)
        subprocess.run([sys.executable,str(HERE/'paper_episode.py'),str(path)],check=True)
