"""Replay a saved full PA episode for equality, without new research labels."""
from experiment import *
from registry import verify_receipt
from runtime_correction_pa_memo_query_disk_candidate_20261022 import run
from pa_query_memo_candidate_20261022 import snapshot
import numpy as np
import resource
import time


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    os.nice(10)
    ctl=sys.argv[1]
    assert ctl in ('prioritized_astar0','prioritized_astar1')
    proof='PA_QUERY_MEMO_CANDIDATE_CODE_20261022.receipt.json'
    verify_receipt(proof);verify_receipt('PA_SAVED_QUERY_PROFILE_20261022.receipt.json')
    source=next(r for r in read(HERE/'PA_SAVED_QUERY_PROFILE_20261022.json')['sources'] if r['controller']==ctl)
    path=Path(source['raw_path']);assert sha(path)==source['raw_sha256']
    raw=read(path);assert sha(raw['trace_path'])==raw['trace_sha256']==source['trace_sha256']
    assert raw['episode_length']==2000 and not raw.get('deterministic_source')
    prefix='PA_QUERY_MEMO_FULL_REPLAY_'+ctl+'_20261022'
    assert not (HERE/(prefix+'.json')).exists()
    plan=dict(utc=utc(),controller=ctl,source_raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256'],source_elapsed_seconds=raw['elapsed_seconds'],candidate_code_receipt_sha256=sha(HERE/proof),new_research_rollouts=0,saved_episode_replays=1,new_labels=0,new_journals=0)
    write(HERE/(prefix+'_PLAN.json'),plan);print(json.dumps(plan),flush=True)
    usage=resource.getrusage(resource.RUSAGE_SELF);cpu_start=usage.ru_utime+usage.ru_stime
    started=time.monotonic();result,arrays=run(raw['case']);seconds=time.monotonic()-started
    usage=resource.getrusage(resource.RUSAGE_SELF);cpu_seconds=usage.ru_utime+usage.ru_stime-cpu_start
    checked={}
    with np.load(raw['trace_path']) as z:
        assert set(arrays)==set(z.files)
        for name,value in arrays.items():
            if name=='step_seconds':continue
            expected=z[name]
            if value.dtype!=expected.dtype or value.shape!=expected.shape or value.tobytes()!=expected.tobytes():
                np.savez_compressed(HERE/(prefix+'_MISMATCH.npz'),**arrays)
                raise AssertionError(('Saved physical array differs',name))
            checked[name]=dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest(),bitwise_equal=True)
    for key,value in result.items():
        if key=='elapsed_seconds':continue
        assert clean(value)==raw[key],('Non-timing result differs',key)
    verify_receipt(proof)
    report=dict(utc=utc(),passed=True,plan=plan,steps=result['episode_length'],outcome=result['outcome'],candidate_seconds=seconds,candidate_cpu_seconds=cpu_seconds,cpu_over_wall_ratio=cpu_seconds/seconds,source_elapsed_seconds=raw['elapsed_seconds'],observed_wall_ratio=raw['elapsed_seconds']/seconds,arrays=checked,all_non_timing_result_fields_equal=True,cache_stats=snapshot(),new_research_rollouts=0,new_labels=0,new_journals=0,saved_episode_replays=1)
    write(HERE/(prefix+'.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='arrays'}),flush=True)


if __name__=='__main__':main()
