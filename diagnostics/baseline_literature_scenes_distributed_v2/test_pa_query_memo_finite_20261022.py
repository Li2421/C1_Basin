"""Byte-exact finite A* tests with changing reservations and initial predicates."""
from experiment import *
import numpy as np
from types import SimpleNamespace
import tiled_planning_start_signature_v2_20261022 as reference
import tiled_planning_pa_memo_candidate_20261022 as candidate
import pa_query_memo_candidate_20261022 as memo


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    os.nice(10)
    box=np.asarray([[[-1,-1],[1,-1]],[[1,-1],[1,1]],[[1,1],[-1,1]],[[-1,1],[-1,-1]]],float)
    rng=np.random.default_rng(2026102211)
    checks=0
    def check(env,res,horizon,start,goal,reserved):
        nonlocal checks
        paths=[]
        for module in (reference,candidate):
            grid=module.Grid(env,res,.1)
            grid.max_time=horizon
            grid.reserved=[p.copy() for p in reserved]
            paths.append(grid.search(start,goal))
        assert (paths[0] is None)==(paths[1] is None),(checks,start,goal,reserved)
        if paths[0] is not None:
            assert paths[0].shape==paths[1].shape and paths[0].tobytes()==paths[1].tobytes(),checks
        checks+=1
    for index in range(128):
        res=(.1,.2,.25)[index%3]
        horizon=(8,12,20,28)[index%4]
        walls=box if index%5 else np.concatenate([box,[[[.04,-.8],[.04,.8]]]])
        margin=.04+(index%3)*.005
        env=SimpleNamespace(walls=walls,config=SimpleNamespace(agent_radius=.04,agent_collision_margin=(index%3)*.003),instance=SimpleNamespace(geometry=SimpleNamespace(margin=margin)))
        start=rng.uniform(-.65,.65,2)
        goal=rng.uniform(-.65,.65,2)
        if index%7==0:goal=start+res*.05
        reserved=[rng.uniform(-.7,.7,((index+j)%4+1,2)) for j in range(index%4)]
        for delta,ref_delta in ((0.,0.),(1e-14,0.),(1e-9,1e-14),(.07,.15),(0.,0.)):
            refs=[p.copy() for p in reserved]
            for p in refs:p[0]+=ref_delta
            check(env,res,horizon,start+delta,goal,refs)
        if index%16==15:print('PA finite',index+1,'checks',checks,flush=True)
    env=SimpleNamespace(walls=box,config=SimpleNamespace(agent_radius=.04,agent_collision_margin=0.),instance=SimpleNamespace(geometry=SimpleNamespace(margin=.04)))
    for start in ([-0.,+0.],[+0.,-0.],[-0.,-0.],[+0.,+0.]):
        for refs in ([],[np.asarray([[.3,.3]])],[np.asarray([[.11,.11],[.3,.3]])]):
            check(env,.2,12,np.asarray(start),np.asarray([.4,.4]),refs)
    # Persistent blocking must cache a genuine no-path result.
    for delta in (0.,1e-14,1e-9,0.):
        check(env,.2,12,np.asarray([.011+delta,.017-delta]),np.asarray([.4,.4]),[np.asarray([[.4,.4]])])
    stats=memo.snapshot()
    assert stats.get('hits',0)>0 and stats.get('negative_hits',0)>0,stats
    assert stats['entries']<=memo.MAX_ENTRIES and stats['estimated_payload_bytes']<=memo.MAX_BYTES
    output=HERE/'PA_QUERY_MEMO_FINITE_20261022.json'
    assert not output.exists()
    write(output,dict(utc=utc(),passed=True,bitwise_path_checks=checks,cache_stats=stats,
                      scope='Resolution,horizon,walls,margin,separation,start cells,goals,0..3 reservations,length1 persistent reservations,changed first reservation points,signed zero,negative paths',
                      new_rollouts=0,environment_steps=0))
    print('PASS',checks,stats,flush=True)


if __name__=='__main__':main()
