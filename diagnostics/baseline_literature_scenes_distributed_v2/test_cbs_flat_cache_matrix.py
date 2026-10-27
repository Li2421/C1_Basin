"""Native-path equivalence across saved scene queries and finite grid edge cases."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from result_inventory import find_raw
import numpy as np
import ctypes
import time
from types import SimpleNamespace


def equal_paths(a,b):
    return all((a[i] is None and b[i] is None) or (a[i] is not None and b[i] is not None and a[i].dtype==b[i].dtype and a[i].shape==b[i].shape and a[i].tobytes()==b[i].tobytes()) for i in a)


def main():
    verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
    import tiled_planning
    from reservation_certificate_v2 import ReservationNominal
    from flow_scene import LiteratureEnv
    from states import LiteratureConfig
    from controllers import bound
    import jax
    jax.config.update('jax_enable_x64',False)
    native=tiled_planning.CBS
    candidate=ctypes.CDLL(str(HERE/'cbs_kernel_flat_cache.so'))
    candidate.astar_constrained.argtypes=native.astar_constrained.argtypes
    candidate.astar_constrained.restype=native.astar_constrained.restype
    rows=[]
    try:
        for scene in NAMES:
            cases=read(HERE/f'distributed_baseline_{scene}_CASES.json')
            for controller in ('conflict_based_search0','conflict_based_search1'):
                case=next(c for c in cases if c['controller_payload']['nominal']['controller']==controller and c['seed']==0)
                path=find_raw(case['rollout_uid'],'distributed_baseline_'+scene,scene);raw=read(path)
                assert sha(raw['trace_path'])==raw['trace_sha256']
                with np.load(raw['trace_path']) as z:
                    for step in sorted({0,min(100,len(z['nominal'])-1)}):
                        outputs=[];paths=[];records=[];seconds=[]
                        for library in (native,candidate):
                            tiled_planning.CBS=library
                            env=LiteratureEnv(case['state'],LiteratureConfig(**case['scenario_payload']['config']))
                            env.positions=z['positions'][step].copy();env.velocities=z['velocities'][step].copy();env.step_count=step
                            actor=ReservationNominal(case['controller_payload']['nominal']);key=jax.random.PRNGKey(0)
                            started=time.monotonic();value=actor.action(env,key);seconds.append(time.monotonic()-started)
                            value=bound(value,env.config.max_speed).astype(np.float64)
                            value[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
                            outputs.append(value);paths.append(actor.planner.paths);records.append(actor.planner.planning_records)
                        assert outputs[0].tobytes()==outputs[1].tobytes(),(scene,controller,step)
                        assert equal_paths(*paths) and records[0]==records[1],(scene,controller,step)
                        rows.append(dict(scene=scene,controller=controller,step=step,native_seconds=seconds[0],candidate_seconds=seconds[1],paths_and_nominals_bitwise_equal=True,planning_records_equal=True,raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256']))
                        write(HERE/'CBS_FLAT_CACHE_MATRIX_PROGRESS.json',dict(utc=utc(),saved_queries=len(rows),last=rows[-1]))
        rng=np.random.default_rng(2026102109);edge_cases=[]
        for i in range(64):
            resolution=(.25,.5,1.)[i%3];horizon=(8,12,20,24)[i%4]
            start=rng.uniform(-1,1,2);goal=rng.uniform(-1,1,2)
            walls=np.asarray([] if i%4==0 else [[-2.,-2.,2.,-2.],[2.,-2.,2.,2.],[2.,2.,-2.,2.],[-2.,2.,-2.,-2.]],float).reshape(-1,4)
            margin=(.05,.2,.4)[i%3]
            env=SimpleNamespace(walls=walls,config=SimpleNamespace(agent_radius=.05,agent_collision_margin=.01),instance=SimpleNamespace(geometry=SimpleNamespace(margin=margin)))
            # Include repeated exclusions and reservation paths; no plant steps.
            exclusions=tuple((j%horizon,*tuple(start if j==0 else np.rint(start/resolution)*resolution),*tuple(goal if j%3==0 else np.rint(goal/resolution)*resolution)) for j in range(i%7))
            reserved=[] if i%3 else [np.asarray([[-.5,0.],[0.,0.],[.5,0.]])]
            results=[]
            for library in (native,candidate):
                tiled_planning.CBS=library;grid=tiled_planning.Grid(env,resolution,.1);grid.max_time=horizon;grid.reserved=list(reserved)
                results.append(grid.search(start,goal,exclusions))
            assert (results[0] is None and results[1] is None) or (results[0] is not None and results[1] is not None and results[0].tobytes()==results[1].tobytes()),i
            edge_cases.append(dict(index=i,finite_grid_horizon=horizon,resolution=resolution,reservations=len(reserved),constraints=len(exclusions),path_present=results[0] is not None,bitwise_equal=True))
    finally:tiled_planning.CBS=native
    result=dict(utc=utc(),passed=True,saved_queries=rows,finite_grid_tests=edge_cases,new_research_rollouts=0,plant_steps=0,scope='Pure cache dictionary layout; same native path, search order and budgets',
        source_hashes={str(HERE/n):sha(HERE/n) for n in ('test_cbs_flat_cache_matrix.py','cbs_kernel_flat_cache.cpp','cbs_kernel_flat_cache.so','tiled_planning.py')})
    write(HERE/'CBS_FLAT_CACHE_MATRIX.json',result)
    print('PASS:',len(rows),'saved scene queries and',len(edge_cases),'finite native grids',flush=True)


if __name__=='__main__':main()
