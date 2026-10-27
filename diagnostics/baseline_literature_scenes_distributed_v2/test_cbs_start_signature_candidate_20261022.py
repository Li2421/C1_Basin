"""Offline finite native grids, no actor rollout or runtime patching."""
from experiment import *
from atomic_write import write
import tiled_planning_flat_static as reference
import tiled_planning_start_signature_candidate_20261022 as candidate
from types import SimpleNamespace
import numpy as np
import ctypes,time


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    candidate.CBS.start_signature_cache_reset()
    candidate.CBS.start_signature_cache_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    rng=np.random.default_rng(2026102206);checks=0;examples=[];started=time.monotonic()
    box=np.asarray([[[-1,-1],[1,-1]],[[1,-1],[1,1]],[[1,1],[-1,1]],[[-1,1],[-1,-1]]],float)
    for index in range(128):
        resolution=(.1,.2,.25)[index%3];horizon=(8,12,20,28)[index%4]
        wall=box if index%5 else np.concatenate([box,[[[.04,-.6],[.04,.6]]]])
        env=SimpleNamespace(walls=wall,config=SimpleNamespace(agent_radius=.04,agent_collision_margin=0.),instance=SimpleNamespace(geometry=SimpleNamespace(margin=.04)))
        origin=rng.uniform(-.65,.65,2);goal=rng.uniform(-.65,.65,2)
        if index%7==0:goal=origin.copy()+resolution*.05
        if index%11==0:origin=np.asarray([-.0,+.0]);goal=np.asarray([.6,.2])
        constraints=[]
        if index%4==0:
            x,y=np.rint(origin/resolution).astype(int)
            constraints=[(0,*origin,(x+1)*resolution,y*resolution)]
        if index%6==0:constraints+=[(2,0.,0.,resolution,0.),(3,resolution,0.,2*resolution,0.)]
        reserved=[] if index%9 else [np.asarray([[-.5,.3],[0.,.3],[.5,.3]])]
        for delta in (np.zeros(2),np.asarray([1e-14,-1e-14]),np.asarray([1e-10,1e-10]),np.asarray([.08,-.09]),np.zeros(2)):
            start=origin+delta;paths=[]
            for backend in (reference,candidate):
                grid=backend.Grid(env,resolution,.1);grid.max_time=horizon;grid.reserved=list(reserved)
                paths.append(grid.search(start,goal,constraints))
            assert (paths[0] is None)==(paths[1] is None),(index,delta)
            if paths[0] is not None:assert paths[0].shape==paths[1].shape and paths[0].tobytes()==paths[1].tobytes(),(index,delta,paths)
            checks+=1
        if index%16==15:print('finite grids',index+1,'bitwise checks',checks,flush=True)
    # Explicit failure caching and time-zero edge signature invalidation.
    env=SimpleNamespace(walls=box,config=SimpleNamespace(agent_radius=.04,agent_collision_margin=0.),instance=SimpleNamespace(geometry=SimpleNamespace(margin=.04)))
    start=np.asarray([.011,.017]);res=.2;x,y=np.rint(start/res).astype(int)
    blocked=[(0,*start,(x+dx)*res,(y+dy)*res) for dx,dy in ((1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(0,0))]
    for delta in (0.,1e-14,1e-9,0.):
        paths=[]
        for backend in (reference,candidate):
            grid=backend.Grid(env,res,.1);grid.max_time=12;paths.append(grid.search(start+delta,np.asarray([.6,.4]),blocked))
        assert (paths[0] is None)==(paths[1] is None)
        if paths[0] is not None:assert paths[0].tobytes()==paths[1].tobytes()
        checks+=1
    stats=np.zeros(5,np.uint64);candidate.CBS.start_signature_cache_stats(stats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
    assert stats[1]>0 and stats[2]>0,stats
    result=dict(utc=utc(),passed=True,bitwise_checks=checks,cache_stats=stats.tolist(),seconds=time.monotonic()-started,
        scope='Changing grid resolution,horizon,walls,goals,constraints,reservations,start cells,time-zero edge predicates,signed zero; exact raw path bytes',
        no_runtime_patching=True,new_rollouts=0,environment_steps=0,
        source_hashes={str(HERE/n):sha(HERE/n) for n in ('cbs_kernel_flat_cache.cpp','cbs_kernel_flat_cache.so','cbs_kernel_start_signature_candidate_20261022.cpp','cbs_kernel_start_signature_candidate_20261022.so','cbs_start_signature_cache_20261022.hpp','tiled_planning_start_signature_candidate_20261022.py',Path(__file__).name)})
    write(HERE/'CBS_START_SIGNATURE_FINITE_GRIDS_20261022.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
