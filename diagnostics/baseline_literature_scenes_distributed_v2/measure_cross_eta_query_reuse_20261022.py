"""Saved-query inference only: upper bound on reuse between separate workers."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from reservation_certificate_v2_start_signature_v2_20261022 import ReservationNominal
from flow_scene import LiteratureEnv
from states import LiteratureConfig
from controllers import bound
import tiled_planning_start_signature_v2_20261022 as backend
import ctypes,time,numpy as np


def stats():
    out=np.zeros(5,np.uint64)
    backend.CBS.start_signature_cache_stats(out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
    return out.tolist()


def query(raw,step):
    c=raw['case'];cfg=LiteratureConfig(**c['scenario_payload']['config'])
    with np.load(raw['trace_path']) as z:
        env=LiteratureEnv(c['state'],cfg);env.positions=z['positions'][step].copy();env.velocities=z['velocities'][step].copy();env.step_count=step
        expected=z['nominal'][step].copy()
    actor=ReservationNominal(c['controller_payload']['nominal']);before=stats();start=time.monotonic()
    value=bound(actor.action(env,None),cfg.max_speed).astype(np.float64)
    value[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
    seconds=time.monotonic()-start;after=stats();assert value.dtype==expected.dtype and value.tobytes()==expected.tobytes(),(c['method'],step)
    digest=hashlib.sha256(value.tobytes())
    for index,path in sorted(actor.planner.paths.items()):
        digest.update(str(index).encode());digest.update(b'NONE' if path is None else path.dtype.str.encode()+canonical(path.shape).encode()+path.tobytes())
    digest.update(canonical(actor.planner.planning_records).encode())
    return dict(method=c['method'],step=step,seconds=seconds,lookups=after[0]-before[0],hits=after[1]-before[1],negative_hits=after[2]-before[2],clears=after[3]-before[3],answer_sha256=digest.hexdigest(),saved_nominal_bitwise_equal=True)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json')
    stage='distributed_revised_fixed_lit_circle_five';cases=[c for c in read(HERE/f'{stage}_CASES.json') if c['controller_payload']['nominal']['controller']=='conflict_based_search1' and c['state']['index']==13 and c['seed']==0]
    assert len(cases)==3
    raws=[read(HERE/stage/'raw'/f"{c['rollout_uid']}.json") for c in cases]
    for r in raws:assert sha(r['trace_path'])==r['trace_sha256'] and r['episode_length']==2000
    write(HERE/'CROSS_ETA_QUERY_REUSE_PLAN_20261022.json',dict(utc=utc(),history=['CBS_START_SIGNATURE_V2_FULL_REPLAY_20261022.json','CBS_START_SIGNATURE_SAVED_QUERIES_20261022.json','CBS_ACCELERATION_REVIEW.json'],gap='Cross-eta low-level query overlap was not measured; no shared-filesystem cache is implemented or deployed',sources=[dict(rollout_uid=c['rollout_uid'],raw_sha256=sha(HERE/stage/'raw'/f"{c['rollout_uid']}.json"),trace_sha256=r['trace_sha256']) for c,r in zip(cases,raws)],frames=[0,600,1200,1800],new_research_rollouts=0,plant_steps=0))
    backend.CBS.start_signature_cache_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    separate=[];shared=[]
    for raw in raws:
        backend.CBS.start_signature_cache_reset()
        for step in (0,600,1200,1800):
            row=query(raw,step);separate.append(row);print('separate',json.dumps(row),flush=True)
            write(HERE/'CROSS_ETA_QUERY_REUSE_PROGRESS_20261022.json',dict(utc=utc(),separate=separate,shared=shared))
    backend.CBS.start_signature_cache_reset()
    for raw in raws:
        for step in (0,600,1200,1800):
            row=query(raw,step);expected=next(r for r in separate if (r['method'],r['step'])==(row['method'],row['step']))
            assert row['answer_sha256']==expected['answer_sha256']
            assert row['lookups']==expected['lookups']
            shared.append(row);print('shared_upper_bound',json.dumps(row),flush=True)
            write(HERE/'CROSS_ETA_QUERY_REUSE_PROGRESS_20261022.json',dict(utc=utc(),separate=separate,shared=shared))
    a=sum(r['lookups']-r['hits'] for r in separate);b=sum(r['lookups']-r['hits'] for r in shared)
    write(HERE/'CROSS_ETA_QUERY_REUSE_20261022.json',dict(utc=utc(),passed=True,separate=separate,shared=shared,separate_native_misses=a,shared_native_misses=b,additional_native_misses_avoided=a-b,interpretation='Ideal sequential shared-memo upper bound only. Production already retains each process memo across cases; concurrent workers may compute the same cold query simultaneously,so this is not a production speedup claim or authorization to alter scientific labels.',shared_storage_cache_implemented=False,new_research_rollouts=0,new_training_runs=0,plant_steps=0,source_hashes={str(HERE/n):sha(HERE/n) for n in (Path(__file__).name,'CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json')}))
    print('PASS: saved-query reuse assessment complete; no production changes',flush=True)


if __name__=='__main__':main()
