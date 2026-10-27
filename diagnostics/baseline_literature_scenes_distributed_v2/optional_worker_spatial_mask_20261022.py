"""Bounded worker for optional versions, with the same official evidence path."""
from experiment import *
from registry import verify_receipt,save,save_alias,repair_journal
import time
import signal
STOP=False

def stop(*args):
    global STOP
    STOP=True

def query_disk_snapshot():
    import ctypes
    from tiled_planning_pa_memo_query_disk_candidate_spatial_mask_20261022 import CBS,QUERY_CACHE_NAMESPACE
    values=(ctypes.c_uint64*7)()
    CBS.query_disk_cache_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    CBS.query_disk_cache_stats.restype=None
    CBS.query_disk_cache_stats(values)
    return dict(namespace=QUERY_CACHE_NAMESPACE,values=list(values))


def pa_query_snapshot():
    from pa_query_memo_candidate_20261022 import snapshot
    return snapshot()


def spatial_mask_snapshot():
    from pa_spatial_mask_stats_20261022 import snapshot
    return snapshot()


def main(stage):
    verify_receipt('PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json');verify_receipt('PA_SPATIAL_MASK_PIPELINE_CODE_20261022.receipt.json')
    verify_receipt('PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json');verify_receipt('PA_QUERY_MEMO_PIPELINE_CODE_20261022.receipt.json')
    verify_receipt('CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_QUERY_DISK_PIPELINE_CODE_20261022.receipt.json')
    verify_receipt('CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
    verify_receipt('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json');verify_receipt('CORRECTION_REVISION.receipt.json')
    verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json');signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    verify_receipt('OPTIONAL_FINE_SCHEDULING.receipt.json');verify_receipt(f'{stage}/FINE_QUEUE.receipt.json')
    prereg();verify_receipt('IMPLEMENTATION_FROZEN.receipt.json');verify_receipt('OPTIONAL_CERTIFIED_BACKENDS_V2.receipt.json');verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json')
    index=read(HERE/stage/'FINE_GROUP_INDEX.json');claims=HERE/stage/'fine_claims';claims.mkdir(parents=True,exist_ok=True);groups=0
    for entry in index:
        if STOP:return
        claim=claims/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(utc=utc(),job=os.environ['SLURM_JOB_ID'],pid=os.getpid()))
        assert sha(entry['file'])==entry['sha256']
        for c in read(entry['file'])[entry['start']:entry['end']]:
            if STOP:return
            raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():repair_journal(stage,c);continue
            engine=c['execution_engine']
            if engine=='paper_dynamics':
                verify_receipt('DYNAMICS_IMPLEMENTATION.receipt.json')
                from dynamics_spatial_mask_20261022 import run
            elif engine=='local_progress':
                from runtime_local_progress_spatial_mask_20261022 import run
            else:
                from runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022 import run
            import jax
            jax.config.update('jax_enable_x64',False)
            kind=c['controller_payload']['nominal']['type_code']
            twin=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
            source=HERE/stage/'raw'/f'{twin}.json'
            if c['seed']==1 and kind in ('gradient_mpc','prioritized_astar','conflict_based_search','goal_reference') and source.exists():result=save_alias(stage,c,source)
            else:
                before_query=query_disk_snapshot() if True else None
                before_pa=pa_query_snapshot() if True else None
                before_spatial=spatial_mask_snapshot() if before_pa is not None else None
                result,arrays=run(c)
                if before_spatial is not None:
                    after_spatial=spatial_mask_snapshot()
                    result['spatial_mask_cache']=dict(counter_deltas={k:after_spatial[k]-before_spatial[k] for k in after_spatial if k!='maximum_entries'},maximum_cells=after_spatial['maximum_entries'])
                    result['spatial_mask_cache_equivalence_receipt_sha256']=sha(HERE/'PA_SPATIAL_MASK_EQUIVALENCE_20261022.receipt.json')
                if before_pa is not None:
                    after_pa=pa_query_snapshot()
                    names=('queries','hits','negative_hits','misses','stored','evictions','bypass','oversize','allocation_fallback')
                    result['pa_query_cache']=dict(counter_deltas={k:after_pa.get(k,0)-before_pa.get(k,0) for k in names},state_after=after_pa)
                    result['pa_query_cache_equivalence_receipt_sha256']=sha(HERE/'PA_QUERY_MEMO_EQUIVALENCE_20261022.receipt.json')
                if before_query is not None:
                    after_query=query_disk_snapshot()
                    result['native_query_cache']=dict(namespace=after_query['namespace'],counter_deltas=[b-a for a,b in zip(before_query['values'],after_query['values'])],counter_order=['reads','hits','collisions','invalid','writes','write_errors','skipped'])
                    result['query_disk_cache_equivalence_receipt_sha256']=sha(HERE/'CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json')
                result['computational_equivalence_receipt_sha256']=sha(HERE/'CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json')
                result['pure_geometry_cache_equivalence_receipt_sha256']=sha(HERE/'CBS_FLAT_CACHE_EQUIVALENCE.receipt.json')
                result['jax_enable_x64']=bool(jax.config.x64_enabled)
                result['correction_revision_receipt_sha256']=sha(HERE/'CORRECTION_REVISION.receipt.json')
                result['optional_revision_execution_receipt_sha256']=sha(HERE/'CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json')
                result['optional_implementation_receipt_sha256']=sha(HERE/'OPTIONAL_DELIVERY_READY.receipt.json')
                result['reservation_equivalence_receipt_sha256']=sha(HERE/'RESERVATION_V2_EQUIVALENCE.receipt.json')
                result['execution_backend']=dict(actual='dynamics_spatial_mask_20261022.py' if engine=='paper_dynamics' else 'runtime_local_progress_spatial_mask_20261022.py' if engine=='local_progress' else 'runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022.py')
                if engine=='paper_dynamics' and c['state']['scene'] in ('lit_narrow_passage','lit_crowded_random'):
                    import numpy as np
                    result['acceleration_bound_violation_steps']=int(np.any(np.linalg.norm(arrays['actions'],axis=-1)>1.5+1e-7,axis=-1).sum())
                save(stage,c,result,arrays)
            print(c['state']['scene'],c['controller_payload']['nominal']['controller'],c['method'],c['seed'],result['outcome'],round(result['elapsed_seconds'],3),flush=True)
        write(claim/'COMPLETE.json',dict(utc=utc()));groups+=1
        if groups>=512:os.execv(sys.executable,[sys.executable,'-B',str(HERE/'optional_worker_spatial_mask_20261022.py'),stage])

if __name__=='__main__':main(sys.argv[1])
