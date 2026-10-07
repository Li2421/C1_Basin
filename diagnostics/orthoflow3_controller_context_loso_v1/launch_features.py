"""Six single-core workers; no rollout evaluator or shared DB writer."""
import os,subprocess,sys
from pathlib import Path
out=Path(__file__).resolve().parent
cpus=sorted(os.sched_getaffinity(0));jobs=[]
todo=[('toy_giveway',0,1),('double_bottleneck',0,1),
      ('four_way_intersection',0,2),('four_way_intersection',1,2),('ring_exchange',0,2),('ring_exchange',1,2)]
assert len(cpus)>=6
for i,(scene,shard,shards) in enumerate(todo):
 log=(out/f'features_{scene}_{shard}.log').open('a')
 cmd=['taskset','-c',str(cpus[i]),sys.executable,'-m','diagnostics.orthoflow3_controller_context_loso_v1.context',
      'build','--scene',scene,'--shard',str(shard),'--shards',str(shards)]
 jobs.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
codes=[p.wait() for p in jobs]
assert not any(codes),codes
