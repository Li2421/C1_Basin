"""Profile one saved-state nominal query; no environment stepping or DB writes."""
from experiment import *
from registry import verify_receipt
import cProfile
import pstats
import io
import time
import numpy as np


def main():
    verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json')
    history=read(HERE/'CBS_SAVED_TRACE_PROFILE_HISTORY.json')
    path=Path(history['raw_sources'][0]);raw=read(path);case=raw['case']
    assert sha(raw['trace_path'])==raw['trace_sha256']
    from states import LiteratureConfig
    from flow_scene import LiteratureEnv
    from reservation_certificate_v2 import ReservationNominal as Nominal
    from controllers import bound
    import jax
    jax.config.update('jax_enable_x64',False)
    with np.load(raw['trace_path']) as trace:
        step=1300
        env=LiteratureEnv(case['state'],LiteratureConfig(**case['scenario_payload']['config']))
        env.positions=trace['positions'][step].copy();env.velocities=trace['velocities'][step].copy();env.step_count=step
        expected=trace['nominal'][step].copy()
    actor=Nominal(case['controller_payload']['nominal']);key=jax.random.PRNGKey(0)
    profiler=cProfile.Profile();started=time.monotonic()
    value=profiler.runcall(actor.action,env,key)
    value=bound(value,env.config.max_speed).astype(np.float64)
    value[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
    elapsed=time.monotonic()-started
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(35)
    write(HERE/'CBS_SAVED_QUERY_PROFILE.json',dict(utc=utc(),source_raw=str(path),source_raw_sha256=sha(path),
        step=step,elapsed_seconds=elapsed,nominal_bitwise_equal=bool(np.array_equal(value,expected)),
        profile=stream.getvalue(),new_rollouts=0,environment_steps=0))
    print('saved CBS query seconds',elapsed,'nominal bitwise equal',np.array_equal(value,expected),flush=True)
    print(stream.getvalue(),flush=True)


if __name__=='__main__':main()
