"""Compare identical native CBS queries on saved evidence; never step a plant."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import ctypes
import time
import numpy as np


def main():
    verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json')
    verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
    from states import LiteratureConfig
    from flow_scene import LiteratureEnv
    from reservation_certificate_v2 import ReservationNominal
    from controllers import bound
    import tiled_planning
    import jax
    jax.config.update('jax_enable_x64',False)
    native=tiled_planning.CBS
    candidate=ctypes.CDLL(str(HERE/'cbs_kernel_time_index.so'))
    candidate.astar_constrained.argtypes=native.astar_constrained.argtypes
    candidate.astar_constrained.restype=native.astar_constrained.restype
    profile=read(HERE/'CBS_SAVED_QUERY_PROFILE.json')
    path=Path(profile['source_raw']);assert sha(path)==profile['source_raw_sha256']
    raw=read(path);assert sha(raw['trace_path'])==raw['trace_sha256']
    case=raw['case'];step=profile['step']
    with np.load(raw['trace_path']) as trace:
        positions=trace['positions'][step].copy();velocities=trace['velocities'][step].copy();expected=trace['nominal'][step].copy()
    rows=[];outputs=[];plans=[];records=[]
    try:
        for name,library in [('native',native),('time_index',candidate)]:
            env=LiteratureEnv(case['state'],LiteratureConfig(**case['scenario_payload']['config']))
            env.positions=positions.copy();env.velocities=velocities.copy();env.step_count=step
            actor=ReservationNominal(case['controller_payload']['nominal']);key=jax.random.PRNGKey(0)
            tiled_planning.CBS=library
            started=time.monotonic();value=actor.action(env,key)
            value=bound(value,env.config.max_speed).astype(np.float64)
            value[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
            elapsed=time.monotonic()-started
            rows.append(dict(backend=name,elapsed_seconds=elapsed,saved_nominal_bitwise_equal=bool(np.array_equal(value,expected))))
            outputs.append(value);plans.append(actor.planner.paths);records.append(actor.planner.planning_records)
            print(name,round(elapsed,6),'seconds; saved nominal equal',rows[-1]['saved_nominal_bitwise_equal'],flush=True)
    finally:tiled_planning.CBS=native
    equal_paths=all((plans[0][i] is None and plans[1][i] is None) or (plans[0][i] is not None and plans[1][i] is not None and np.array_equal(plans[0][i],plans[1][i])) for i in plans[0])
    result=dict(utc=utc(),source_raw=str(path),source_raw_sha256=sha(path),trace_sha256=raw['trace_sha256'],step=step,
        queries=rows,nominal_bitwise_equal=bool(np.array_equal(*outputs)),all_paths_bitwise_equal=equal_paths,planning_records_equal=records[0]==records[1],
        speedup=rows[0]['elapsed_seconds']/rows[1]['elapsed_seconds'],new_rollouts=0,environment_steps=0,deployed=False,
        sources={str(HERE/n):sha(HERE/n) for n in ('cbs_kernel_tiled.cpp','cbs_kernel_time_index.cpp','cbs_kernel_time_index.so','test_cbs_time_index_saved.py')})
    write(HERE/'CBS_TIME_INDEX_SAVED_QUERY.json',result)
    assert all(r['saved_nominal_bitwise_equal'] for r in rows) and result['nominal_bitwise_equal'] and equal_paths and result['planning_records_equal'],result
    print('Paths, nominal and planning records exact; speedup',result['speedup'],flush=True)


if __name__=='__main__':main()
