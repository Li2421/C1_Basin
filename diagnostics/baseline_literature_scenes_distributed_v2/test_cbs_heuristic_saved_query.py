"""Compare isolated current/candidate actors on a saved query, without patching."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import time,numpy as np

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});os.nice(10)
 verify_receipt('CBS_HEURISTIC_CACHE_CANDIDATE_CODE.receipt.json')
 from reservation_certificate_v2_flat_static import ReservationNominal as Current
 from reservation_certificate_v2_heuristic_candidate import ReservationNominal as Candidate
 from flow_scene import LiteratureEnv
 from states import LiteratureConfig
 from controllers import bound
 import jax
 jax.config.update('jax_enable_x64',False)
 profile=read(HERE/'CBS_SAVED_QUERY_PROFILE.json');path=Path(profile['source_raw']);assert sha(path)==profile['source_raw_sha256'];raw=read(path);assert sha(raw['trace_path'])==raw['trace_sha256'];case=raw['case'];step=profile['step'];records=[]
 with np.load(raw['trace_path']) as trace:
  reference=trace['nominal'][step].copy()
  for repetition in range(3):
   outputs=[];paths=[];plans=[];times={}
   factories=[('current',Current),('candidate',Candidate)]
   if repetition%2:factories.reverse()
   for name,factory in factories:
    env=LiteratureEnv(case['state'],LiteratureConfig(**case['scenario_payload']['config']));env.positions=trace['positions'][step].copy();env.velocities=trace['velocities'][step].copy();env.step_count=step
    actor=factory(case['controller_payload']['nominal']);key=jax.random.PRNGKey(0);started=time.monotonic();value=actor.action(env,key);seconds=time.monotonic()-started
    value=bound(value,env.config.max_speed).astype(np.float64);value[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
    assert value.dtype==reference.dtype and value.tobytes()==reference.tobytes(),(name,repetition)
    times[name]=seconds;outputs.append(value);paths.append(actor.planner.paths);plans.append(actor.planner.planning_records)
   assert all((paths[0][i] is None and paths[1][i] is None) or (paths[0][i] is not None and paths[1][i] is not None and paths[0][i].dtype==paths[1][i].dtype and paths[0][i].tobytes()==paths[1][i].tobytes()) for i in paths[0])
   assert plans[0]==plans[1]
   records.append(dict(repetition=repetition,current_seconds=times['current'],candidate_seconds=times['candidate'],speedup=times['current']/times['candidate'],nominal_bitwise_equal=True,paths_bitwise_equal=True,planning_records_equal=True))
   write(HERE/'CBS_HEURISTIC_CACHE_QUERY_PROGRESS.json',dict(utc=utc(),records=records))
 write(HERE/'CBS_HEURISTIC_CACHE_SAVED_QUERY.json',dict(utc=utc(),passed=True,records=records,source_raw=str(path),source_raw_sha256=sha(path),step=step,cpu_affinity=sorted(os.sched_getaffinity(0)),new_rollouts=0,environment_steps=0,production_changes=0))
 print(json.dumps(records),flush=True)
if __name__=='__main__':main()
