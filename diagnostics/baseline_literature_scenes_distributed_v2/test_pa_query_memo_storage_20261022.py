"""Exercise real native paths across eviction and oversized-key fallback."""
from experiment import *
import numpy as np
from types import SimpleNamespace
import tiled_planning_start_signature_v2_20261022 as reference
import tiled_planning_pa_memo_candidate_20261022 as candidate
import pa_query_memo_candidate_20261022 as memo


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    os.nice(10)
    env=SimpleNamespace(walls=np.zeros((0,2,2)),config=SimpleNamespace(agent_radius=.04,agent_collision_margin=0.),instance=SimpleNamespace(geometry=SimpleNamespace(margin=.04)))
    start=np.asarray([.01,.02])
    for index in range(memo.MAX_ENTRIES+1):
        grid=candidate.Grid(env,.2,.1);grid.max_time=4
        answer=grid.search(start,np.asarray([.4+index*1e-12,.4]))
        assert answer is not None
    assert memo.snapshot().get('evictions',0)>0,memo.snapshot()
    for index in (0,memo.MAX_ENTRIES):
        values=[]
        for module in (reference,candidate):
            grid=module.Grid(env,.2,.1);grid.max_time=4
            values.append(grid.search(start,np.asarray([.4+index*1e-12,.4])))
        assert values[0].tobytes()==values[1].tobytes()
    # A long constant reservation is cheap for native search but too large to store.
    reserved=np.zeros((memo.MAX_BYTES//16+1,2),np.float64)
    values=[]
    for module in (reference,candidate):
        grid=module.Grid(env,.2,.1);grid.max_time=4;grid.reserved=[reserved]
        values.append(grid.search(np.asarray([.61,.62]),np.asarray([.6,.6])))
    assert values[0].tobytes()==values[1].tobytes()
    stats=memo.snapshot()
    assert stats.get('oversize',0)>0 and stats['entries']<=memo.MAX_ENTRIES and stats['estimated_payload_bytes']<=memo.MAX_BYTES,stats
    path=HERE/'PA_QUERY_MEMO_STORAGE_20261022.json';assert not path.exists()
    write(path,dict(utc=utc(),passed=True,entry_eviction_native_paths_equal=True,oversized_native_fallback_equal=True,cache_stats=stats,new_rollouts=0,environment_steps=0))
    print('PASS bounded PA cache',stats,flush=True)


if __name__=='__main__':main()
