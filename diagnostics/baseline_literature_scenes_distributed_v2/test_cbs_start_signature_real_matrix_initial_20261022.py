"""Static four-scene query and IEEE zero/constraint regression matrix."""
from experiment import *
from atomic_write import write
from reservation_certificate_v2_flat_static import ReservationNominal as Reference
from reservation_certificate_v2_start_signature_v2_20261022 import ReservationNominal as Candidate
from flow_scene import LiteratureEnv
from states import LiteratureConfig
from controllers import bound
import tiled_planning_flat_static as refgrid
import tiled_planning_start_signature_v2_20261022 as newgrid
from types import SimpleNamespace
import numpy as np,ctypes,time


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    box=np.asarray([[[-1,-1],[1,-1]],[[1,-1],[1,1]],[[1,1],[-1,1]],[[-1,1],[-1,-1]]],float)
    env=SimpleNamespace(walls=box,config=SimpleNamespace(agent_radius=.04,agent_collision_margin=0.),instance=SimpleNamespace(geometry=SimpleNamespace(margin=.04)))
    extra=[]
    cases=[(np.asarray(s),np.asarray(g),c) for s,g,c in [
        ([-0.,+0.],[.4,.4],[]),([+0.,-0.],[.4,.4],[]),([-0.,-0.],[.4,.4],[]),
        ([+0.,+0.],[-0.,.4],[]),([-0.,-0.],[+0.,.4],[]),
        ([.011,.017],[.4,.4],[]),([.011+1e-14,.017-1e-14],[.4,.4],[]),
        ([.011,.017],[.4,.4],[(5,.4,.4,.4,.4)]),
        ([.011,.017],[.4,.4],[(0,.4,.4,.4,.4)]),
        ([.011,.017],[.4,.4],[(0,.4,.4,.4,.4),(0,.4,.4,.4,.4)]),
        ([.011,.017],[.46,.46],[]),([.011,.017],[.45,.45],[])]]
    for index,(start,goal,constraints) in enumerate(cases):
        values=[]
        for module in (refgrid,newgrid):
            grid=module.Grid(env,.2,.1);grid.max_time=12;values.append(grid.search(start,goal,constraints))
        assert (values[0] is None)==(values[1] is None),index
        if values[0] is not None:assert values[0].tobytes()==values[1].tobytes(),index
        extra.append(dict(index=index,start_hex=[float(x).hex() for x in start],goal_hex=[float(x).hex() for x in goal],path_bytes_equal=True))
    source=read(HERE/'CBS_FLAT_CACHE_STATIC_QUERY_MATRIX.json');rows=[]
    for number,entry in enumerate(source['saved_queries']):
        origin=entry['source'];path=Path(origin['raw']);raw=read(path);assert sha(path)==origin['raw_sha256'];assert sha(raw['trace_path'])==origin['trace_sha256']
        c=raw['case'];step=origin['step'];cfg=LiteratureConfig(**c['scenario_payload']['config']);answers=[];paths=[];records=[];seconds=[]
        with np.load(raw['trace_path']) as z:
            for cls in (Reference,Candidate):
                env=LiteratureEnv(c['state'],cfg);env.positions=z['positions'][step].copy();env.velocities=z['velocities'][step].copy();env.step_count=step
                actor=cls(c['controller_payload']['nominal']);started=time.monotonic();answer=bound(actor.action(env,None),cfg.max_speed).astype(np.float64);answer[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0;seconds.append(time.monotonic()-started)
                answers.append(answer);paths.append(actor.planner.paths);records.append(actor.planner.planning_records)
            assert answers[0].tobytes()==answers[1].tobytes()==z['nominal'][step].tobytes(),('nominal',number,origin)
            assert records[0]==records[1],('records',number)
            for i in paths[0]:
                assert (paths[0][i] is None)==(paths[1][i] is None),(number,i)
                if paths[0][i] is not None:assert paths[0][i].tobytes()==paths[1][i].tobytes(),('path',number,i)
        rows.append(dict(source=origin,all_path_and_nominal_bytes_equal=True,planning_records_equal=True,reference_seconds=seconds[0],candidate_seconds=seconds[1]))
        write(HERE/'CBS_START_SIGNATURE_REAL_MATRIX_PROGRESS_20261022.json',dict(utc=utc(),passed_queries=len(rows),planned_queries=len(source['saved_queries']),queries=rows));print('PASS saved query',number+1,origin['scene'],origin['controller'],step,seconds,flush=True)
    result=dict(utc=utc(),passed=True,saved_queries=rows,ieee_and_time_constraints=extra,no_runtime_patching=True,new_research_rollouts=0,plant_steps=0,
        source_hashes={str(HERE/n):sha(HERE/n) for n in ('cbs_kernel_start_signature_v2_candidate_20261022.cpp','cbs_kernel_start_signature_v2_candidate_20261022.so','cbs_start_signature_cache_v2_20261022.hpp','tiled_planning_start_signature_v2_20261022.py',Path(__file__).name)})
    write(HERE/'CBS_START_SIGNATURE_REAL_MATRIX_20261022.json',result);print('Complete:17saved queries and12IEEE/constraint cases',flush=True)


if __name__=='__main__':main()
