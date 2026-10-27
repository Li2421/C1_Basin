"""One full engineering replay of a saved case; no research labels or journals."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import numpy as np
import ctypes
import time


def main():
    verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
    verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json')
    profile=read(HERE/'CBS_SAVED_QUERY_PROFILE.json');path=Path(profile['source_raw'])
    assert sha(path)==profile['source_raw_sha256'];raw=read(path)
    assert sha(raw['trace_path'])==raw['trace_sha256']
    with np.load(raw['trace_path']) as z:expected={k:z[k].copy() for k in z.files if k!='step_seconds'}
    import tiled_planning
    from controllers import bound
    from reservation_certificate_v2 import ReservationNominal
    import runtime_reservation_v2 as runtime
    library=ctypes.CDLL(str(HERE/'cbs_kernel_flat_cache.so'))
    library.astar_constrained.argtypes=tiled_planning.CBS.astar_constrained.argtypes
    library.astar_constrained.restype=tiled_planning.CBS.astar_constrained.restype
    tiled_planning.CBS=library
    started=time.monotonic()
    class CheckedNominal(ReservationNominal):
        def action(self,env,key):
            value=super().action(env,key)
            nominal=bound(value,env.config.max_speed).astype(np.float64)
            nominal[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
            assert nominal.tobytes()==expected['nominal'][env.step_count].tobytes(),('Nominal mismatch',env.step_count)
            if env.step_count%50==0:
                write(HERE/'CBS_FLAT_CACHE_REPLAY_PROGRESS.json',dict(utc=utc(),step=env.step_count,total_steps=raw['episode_length'],elapsed_seconds=time.monotonic()-started,new_research_rollouts=0,new_training=0))
            return value
    runtime.Nominal=CheckedNominal
    result,arrays=runtime.run(raw['case'])
    checks={}
    for key,original in expected.items():
        actual=np.asarray(arrays[key]);checks[key]=actual.dtype==original.dtype and actual.shape==original.shape and actual.tobytes()==original.tobytes()
    terminal={k:result[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length','switch_step','reservation_no_goal_certificates')}
    proof=dict(utc=utc(),passed=all(checks.values()) and all(terminal.values()),arrays=checks,terminal=terminal,
        source_raw=str(path),source_raw_sha256=sha(path),trace_sha256=raw['trace_sha256'],
        elapsed_seconds=time.monotonic()-started,saved_original_seconds=raw['elapsed_seconds'],new_research_rollouts=0,new_training=0,
        replayed_steps=len(arrays['nominal']),all_saved_nominals_checked=True,
        source_hashes={str(HERE/n):sha(HERE/n) for n in ('replay_flat_cache_saved.py','runtime_reservation_v2.py','cbs_kernel_flat_cache.cpp','cbs_kernel_flat_cache.so')})
    write(HERE/'CBS_FLAT_CACHE_FULL_REPLAY.json',proof)
    print(canonical(proof),flush=True)
    assert proof['passed'],proof


if __name__=='__main__':main()
