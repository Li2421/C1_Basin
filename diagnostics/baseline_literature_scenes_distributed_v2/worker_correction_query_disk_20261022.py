"""One CPU, bounded queue, persisted completed evidence, no SQLite writer."""
from experiment import *
from registry import verify_receipt,save,save_alias,repair_journal
import signal
import time
STOP=False

def stop(*args):
    global STOP
    STOP=True

def query_disk_snapshot():
    import ctypes
    from tiled_planning_query_disk_candidate_20261022 import CBS,QUERY_CACHE_NAMESPACE
    values=(ctypes.c_uint64*7)()
    CBS.query_disk_cache_stats.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    CBS.query_disk_cache_stats.restype=None
    CBS.query_disk_cache_stats(values)
    return dict(namespace=QUERY_CACHE_NAMESPACE,values=list(values))


def main(stage):
    verify_receipt('CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_QUERY_DISK_PIPELINE_CODE_20261022.receipt.json')
    verify_receipt('CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
    assert stage.startswith(('distributed_revised_fixed_','central_revised_fixed_','learning_labels_revised_20261021_','learning_eval_revised_20261021_')),stage
    verify_receipt('CORRECTION_REVISION.receipt.json');verify_receipt('PREREGISTRATION_ADDENDUM_20261021.receipt.json');verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json')
    verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json');prereg();verify_receipt('IMPLEMENTATION_FROZEN.receipt.json');verify_receipt('RESERVATION_V2_EQUIVALENCE.receipt.json');verify_receipt('WORKER_FULL_CERTIFIED_DEPLOYMENT.receipt.json')
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('PIPELINED_SCHEDULING.receipt.json');verify_receipt(f'{stage}/FINE_QUEUE.receipt.json')
    index=read(HERE/stage/'FINE_GROUP_INDEX.json')
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    claims=HERE/stage/'fine_claims';claims.mkdir(parents=True,exist_ok=True);count=0;groups=0
    from runtime_correction_query_disk_candidate_20261022 import run as distributed_run
    from central_correction_revised_20261021 import run as historical_run
    for entry in index:
        if STOP:break
        claim=claims/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(utc=utc(),job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),rollout_uids=entry['rollout_uids']))
        path=Path(entry['file']);assert path.resolve().is_relative_to(HERE/stage/'queue') and sha(path)==entry['sha256']
        for c in read(path)[entry['start']:entry['end']]:
            if STOP:break
            out=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if out.exists():repair_journal(stage,c);continue
            kind=c['controller_payload']['nominal']['type_code']
            twin=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
            source=HERE/stage/'raw'/f'{twin}.json'
            if c['seed']==1 and kind in ('gradient_mpc','prioritized_astar','conflict_based_search','goal_reference') and source.exists():
                result=save_alias(stage,c,source)
            else:
                run=historical_run if c['safety_layer']=='central' and c['state']['scene']!='lit_crowded_random' else distributed_run
                before_query=query_disk_snapshot() if run is distributed_run else None
                result,arrays=run(c)
                if before_query is not None:
                    after_query=query_disk_snapshot()
                    result['native_query_cache']=dict(namespace=after_query['namespace'],counter_deltas=[b-a for a,b in zip(before_query['values'],after_query['values'])],counter_order=['reads','hits','collisions','invalid','writes','write_errors','skipped'])
                    result['query_disk_cache_equivalence_receipt_sha256']=sha(HERE/'CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json')
                result['computational_equivalence_receipt_sha256']=sha(HERE/'CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json') if run is distributed_run else None
                result['correction_revision_receipt_sha256']=sha(HERE/'CORRECTION_REVISION.receipt.json')
                result['pure_geometry_cache_equivalence_receipt_sha256']=sha(HERE/'CBS_FLAT_CACHE_EQUIVALENCE.receipt.json')
                result['execution_backend']=dict(actual='central_correction_revised_20261021.py' if run is historical_run else 'runtime_correction_query_disk_candidate_20261022.py',implementation_receipt_sha256=sha(HERE/'IMPLEMENTATION_FROZEN.receipt.json'))
                result["reservation_equivalence_receipt_sha256"]=sha(HERE/"RESERVATION_V2_EQUIVALENCE.receipt.json");result["central_fixed_equivalence_receipt_sha256"]=sha(HERE/"CENTRAL_FIXED_CERTIFICATE_REPLAY.receipt.json");save(stage,c,result,arrays)
            count+=1
            print(c['controller_payload']['nominal']['controller'],c['state']['N'],c['state']['index'],c['method'],c['seed'],result['outcome'],round(result['elapsed_seconds'],4),flush=True)
        if STOP:break
        write(claim/'COMPLETE.json',dict(utc=utc()));groups+=1
        write(HERE/stage/f"worker_{os.environ['SLURM_JOB_ID']}.json",dict(utc=utc(),completed_in_process=count,groups=groups,last=entry['group']))
        if groups>=512:
            os.execv(sys.executable,[sys.executable,'-B',str(HERE/'worker_correction_query_disk_20261022.py'),stage])
    write(HERE/stage/f"worker_{os.environ['SLURM_JOB_ID']}_EXIT.json",dict(utc=utc(),graceful_stop=STOP,completed_in_process=count))

if __name__=='__main__':main(sys.argv[1])
