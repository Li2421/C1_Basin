"""Independent static implementations on saved replanning states; no plant steps."""
from experiment import *
from atomic_write import write
from reservation_certificate_v2_flat_static import ReservationNominal as Reference
from reservation_certificate_v2_start_signature_v2_20261022 import ReservationNominal as Candidate
from flow_scene import LiteratureEnv
from states import LiteratureConfig
from controllers import bound
import tiled_planning_start_signature_v2_20261022 as backend
import ctypes,time,numpy as np


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    profile=read(HERE/'CBS_SAVED_QUERY_PROFILE.json');path=Path(profile['source_raw']);raw=read(path)
    assert sha(path)==profile['source_raw_sha256'];assert sha(raw['trace_path'])==raw['trace_sha256']
    c=raw['case'];cfg=LiteratureConfig(**c['scenario_payload']['config']);rows=[]
    backend.CBS.start_signature_cache_reset()
    backend.CBS.start_signature_cache_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    with np.load(raw['trace_path']) as z:
        for step in (1300,1310,1320):
            answers=[];paths=[];records=[];durations=[]
            for actor_class in (Reference,Candidate):
                env=LiteratureEnv(c['state'],cfg);env.positions=z['positions'][step].copy();env.velocities=z['velocities'][step].copy();env.step_count=step
                actor=actor_class(c['controller_payload']['nominal']);started=time.monotonic();answer=bound(actor.action(env,None),cfg.max_speed).astype(np.float64)
                answer[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
                answers.append(answer);durations.append(time.monotonic()-started)
                paths.append(actor.planner.paths);records.append(actor.planner.planning_records)
            assert answers[0].tobytes()==answers[1].tobytes(),('nominal',step)
            assert answers[0].tobytes()==z['nominal'][step].tobytes(),('saved nominal',step)
            assert records[0]==records[1],('planning record',step)
            assert paths[0].keys()==paths[1].keys()
            for i in paths[0]:
                assert (paths[0][i] is None)==(paths[1][i] is None),(step,i)
                if paths[0][i] is not None:assert paths[0][i].tobytes()==paths[1][i].tobytes(),('path',step,i)
            stats=np.zeros(5,np.uint64);backend.CBS.start_signature_cache_stats(stats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
            row=dict(step=step,reference_seconds=durations[0],candidate_seconds=durations[1],speedup=durations[0]/durations[1],nominal_and_all_path_bytes_equal=True,records_equal=True,cache_stats=stats.tolist());rows.append(row)
            write(HERE/'CBS_START_SIGNATURE_SAVED_QUERY_PROGRESS_20261022.json',dict(utc=utc(),queries=rows,new_rollouts=0));print(json.dumps(row),flush=True)
    write(HERE/'CBS_START_SIGNATURE_SAVED_QUERIES_20261022.json',dict(utc=utc(),passed=True,queries=rows,raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256'],no_runtime_patching=True,new_rollouts=0,environment_steps=0,
          source_hashes={str(HERE/n):sha(HERE/n) for n in ('cbs_kernel_start_signature_v2_candidate_20261022.cpp','cbs_start_signature_cache_v2_20261022.hpp','cbs_kernel_start_signature_v2_candidate_20261022.so','tiled_planning_start_signature_v2_20261022.py','reservation_certificate_start_signature_v2_20261022.py','reservation_certificate_v2_start_signature_v2_20261022.py',Path(__file__).name)}))


if __name__=='__main__':main()
