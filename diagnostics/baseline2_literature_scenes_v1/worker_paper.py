from experiment import *
from registry import save
import time

def main(shard,shards):
    prereg();cases=read(HERE/'paper_primary_MISSING_CASES.json')
    # Prevent importing workflow JAX or changing existing environments: this
    # worker runs only in the isolated external-paper environment.
    from paper import run
    count=0;start=time.monotonic()
    for i,c in enumerate(cases):
        if i%shards!=shard:continue
        out=HERE/'paper_primary/raw'/f"{c['rollout_uid']}.json"
        if out.exists():continue
        result,arrays=run(c);save('paper_primary',c,result,arrays);count+=1
        print(c['state']['scene'],c['state']['N'],c['state']['index'],c['method'],result['outcome'],result['elapsed_seconds'],flush=True)
        write(HERE/f'paper_primary/worker_{shard}.json',dict(utc=utc(),completed=count,elapsed=time.monotonic()-start,last=c['rollout_uid']))
    write(HERE/f'paper_primary/worker_{shard}_COMPLETE.json',dict(utc=utc(),completed=count))

if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
