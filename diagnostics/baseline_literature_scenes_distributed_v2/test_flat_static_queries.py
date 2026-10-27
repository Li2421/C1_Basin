"""Exercise static new modules directly; never replace any imported global."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import numpy as np
import time
from types import SimpleNamespace


def path_equal(a,b):
    return (a is None and b is None) or (a is not None and b is not None and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())


def main():
    verify_receipt('CBS_FLAT_CACHE_STATIC_SOURCE_CHECK.receipt.json')
    verify_receipt('CBS_FLAT_CACHE_CODE.receipt.json')
    from reservation_certificate_v2 import ReservationNominal as NativeNominal
    from reservation_certificate_v2_flat_static import ReservationNominal as StaticNominal
    from tiled_planning import Grid as NativeGrid
    from tiled_planning_flat_static import Grid as StaticGrid
    from flow_scene import LiteratureEnv
    from states import LiteratureConfig
    from controllers import bound
    import jax
    jax.config.update('jax_enable_x64',False)
    sources=read(HERE/'CBS_FLAT_CACHE_MATRIX.json')['saved_queries']
    profile=read(HERE/'CBS_SAVED_QUERY_PROFILE.json')
    sources += [dict(raw=profile['source_raw'],raw_sha256=profile['source_raw_sha256'],step=profile['step'])]
    rows=[]
    for source in sources:
        path=Path(source['raw']);assert sha(path)==source['raw_sha256']
        raw=read(path);assert sha(raw['trace_path'])==raw['trace_sha256'];case=raw['case'];step=source['step']
        outputs=[];paths=[];records=[];seconds=[]
        with np.load(raw['trace_path']) as trace:
            for actor_type in (NativeNominal,StaticNominal):
                env=LiteratureEnv(case['state'],LiteratureConfig(**case['scenario_payload']['config']))
                env.positions=trace['positions'][step].copy();env.velocities=trace['velocities'][step].copy();env.step_count=step
                actor=actor_type(case['controller_payload']['nominal']);key=jax.random.PRNGKey(0)
                started=time.monotonic();value=actor.action(env,key);seconds.append(time.monotonic()-started)
                value=bound(value,env.config.max_speed).astype(np.float64)
                value[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
                outputs.append(value);paths.append(actor.planner.paths);records.append(actor.planner.planning_records)
        assert outputs[0].dtype==outputs[1].dtype and outputs[0].tobytes()==outputs[1].tobytes(),source
        assert all(path_equal(paths[0][i],paths[1][i]) for i in paths[0]),source
        assert records[0]==records[1],source
        rows.append(dict(source=source,nominal_bitwise_equal=True,all_paths_bitwise_equal=True,planning_records_equal=True,native_seconds=seconds[0],static_seconds=seconds[1]))
        write(HERE/'CBS_FLAT_CACHE_STATIC_QUERY_PROGRESS.json',dict(utc=utc(),completed=len(rows),planned=len(sources)))
    rng=np.random.default_rng(2026102109);grids=[]
    for i in range(64):
        resolution=(.25,.5,1.)[i%3];horizon=(8,12,20,24)[i%4]
        start=rng.uniform(-1,1,2);goal=rng.uniform(-1,1,2)
        walls=np.asarray([] if i%4==0 else [[-2.,-2.,2.,-2.],[2.,-2.,2.,2.],[2.,2.,-2.,2.],[-2.,2.,-2.,-2.]],float).reshape(-1,4)
        margin=(.05,.2,.4)[i%3]
        env=SimpleNamespace(walls=walls,config=SimpleNamespace(agent_radius=.05,agent_collision_margin=.01),instance=SimpleNamespace(geometry=SimpleNamespace(margin=margin)))
        exclusions=tuple((j%horizon,*tuple(start if j==0 else np.rint(start/resolution)*resolution),*tuple(goal if j%3==0 else np.rint(goal/resolution)*resolution)) for j in range(i%7))
        reserved=[] if i%3 else [np.asarray([[-.5,0.],[0.,0.],[.5,0.]])]
        answers=[]
        for factory in (NativeGrid,StaticGrid):
            grid=factory(env,resolution,.1);grid.max_time=horizon;grid.reserved=list(reserved)
            answers.append(grid.search(start,goal,exclusions))
        assert path_equal(*answers),i
        grids.append(dict(index=i,bitwise_equal=True,path_present=answers[0] is not None))
    write(HERE/'CBS_FLAT_CACHE_STATIC_QUERY_MATRIX.json',dict(utc=utc(),passed=True,static_imports_only=True,no_runtime_module_patching=True,saved_queries=rows,finite_native_grids=grids,new_research_rollouts=0,plant_steps=0,source_hashes={str(HERE/n):sha(HERE/n) for n in ('test_flat_static_queries.py','CBS_FLAT_CACHE_STATIC_SOURCE_CHECK.receipt.json','cbs_kernel_flat_cache.so')}))
    print('PASS:17 static saved queries and64 native grids; no module patching',flush=True)


if __name__=='__main__':main()
