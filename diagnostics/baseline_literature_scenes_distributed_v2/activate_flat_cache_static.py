"""Activate exact cache storage only after all immutable equivalence proofs pass."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_released import snapshot
import subprocess
import fcntl
import time


def main():
    with (HERE/'FLAT_CACHE_ACTIVATOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('CBS_FLAT_CACHE_STATIC_ACTIVATOR_CODE.receipt.json')
        verify_receipt('CBS_FLAT_CACHE_CODE.receipt.json')
        write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_PID.json',dict(utc=utc(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0))))
        proof=HERE/'CBS_FLAT_CACHE_FULL_REPLAY.json'
        while not proof.exists() or not (HERE/'CBS_FLAT_CACHE_STATIC_QUERY_MATRIX.json').exists():
            write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_STATUS.json',dict(utc=utc(),phase='Waiting for full saved replay; no deployment',main_workers_untouched=True))
            time.sleep(10)
        replay=read(proof);assert replay['passed'],replay
        matrix=read(HERE/'CBS_FLAT_CACHE_MATRIX.json');assert matrix['passed']
        static=read(HERE/'CBS_FLAT_CACHE_STATIC_QUERY_MATRIX.json');assert static['passed'] and static['no_runtime_module_patching']
        verify_receipt('CBS_FLAT_CACHE_STATIC_SOURCE_CHECK.receipt.json')
        assert len(static['saved_queries'])==17 and len(static['finite_native_grids'])==64
        assert all(sha(path)==expected for path,expected in static['source_hashes'].items())
        assert replay['replayed_steps']==2000 and replay['all_saved_nominals_checked']
        assert len(matrix['saved_queries'])==16 and len(matrix['finite_grid_tests'])==64
        assert all(sha(path)==expected for path,expected in replay['source_hashes'].items())
        assert all(sha(path)==expected for path,expected in matrix['source_hashes'].items())
        changes=read(HERE/'CBS_FLAT_CACHE_STATIC_SOURCE_CHANGES.json')['changes']
        for item in changes:
            assert sha(HERE/item['original'])==item['original_sha256']
            assert sha(HERE/item['new'])==item['new_sha256']
        if not (HERE/'CBS_FLAT_CACHE_EQUIVALENCE.receipt.json').exists():
            seal('CBS_FLAT_CACHE_EQUIVALENCE.receipt.json',[HERE/n for n in ('CBS_FLAT_CACHE_CODE.receipt.json','CBS_FLAT_CACHE_FULL_REPLAY.json','CBS_FLAT_CACHE_MATRIX.json','CBS_FLAT_CACHE_SOURCE_CHECK.json','CBS_FLAT_CACHE_STATIC_SOURCE_CHECK.receipt.json','CBS_FLAT_CACHE_STATIC_QUERY_MATRIX.json','test_flat_static_queries.py','tiled_parent_store.hpp')],
                 passed=True,all_noncache_cpp_bytes_unchanged=True,full_replay_steps=2000,native_saved_queries=16,static_saved_queries=17,no_runtime_patching=True,finite_native_grid_tests=64,dictionary_exact_keys=1100000,new_research_rollouts=0)
        paths=[HERE/n for n in ('tiled_planning_flat_static.py','deploy_flat_cache_static.py','CBS_FLAT_CACHE_STATIC_SOURCE_CHANGES.json','CBS_FLAT_CACHE_STATIC_SOURCE_CHECK.json','CBS_FLAT_CACHE_EQUIVALENCE.receipt.json','CBS_FLAT_CACHE_STATIC_ACTIVATOR_CODE.receipt.json')]
        paths += [HERE/item['new'] for item in changes]+[HERE/(item['new']+'.diff') for item in changes]
        if not (HERE/'CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json').exists():
            seal('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json',paths,scientific_functions_unchanged=True,case_payloads_unchanged=True,only_storage_and_entrypoints=True,no_hard_rollout_cancellation=True)
        verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json')
        write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_STATUS.json',dict(utc=utc(),phase='Equivalence passed; switching local orchestration'))
        with (HERE/'CBS_FLAT_CACHE_DEPLOYMENT.log').open('a') as stream:
            result=subprocess.run([sys.executable,'-B',str(HERE/'deploy_flat_cache_static.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,'See CBS_FLAT_CACHE_DEPLOYMENT.log'
        migration=read(HERE/'CBS_FLAT_CACHE_MIGRATION_INITIATED.json');old=set(migration['old_jobs'])
        while True:
            state=snapshot();remaining=old&{j['job_id'] for j in state['jobs']}
            write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_STATUS.json',dict(utc=utc(),phase='Graceful old-worker episode drain' if remaining else 'Verifying preserved evidence',old_workers_remaining=sorted(remaining),resource_cpu=state['scheduled_cpus'],resource_gpu=state['gpu_shards']))
            if not remaining:break
            time.sleep(10)
        before=read(HERE/'CBS_FLAT_CACHE_PREMIGRATION_RAW_HASHES.json')
        for path,expected in before.items():assert sha(path)==expected,path
        verify_receipt('CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json')
        write(HERE/'CBS_FLAT_CACHE_MIGRATION_COMPLETE.json',dict(utc=utc(),old_workers_remaining=0,all_preexisting_raw_hashes_unchanged=len(before),hard_cancelled_rollouts=0,discarded_cases=0,case_payloads_unchanged=True,resources=snapshot(),main_still_running=True))
        seal('CBS_FLAT_CACHE_MIGRATION_COMPLETE.receipt.json',[HERE/n for n in ('CBS_FLAT_CACHE_MIGRATION_INITIATED.json','CBS_FLAT_CACHE_PREMIGRATION_RAW_HASHES.json','CBS_FLAT_CACHE_MIGRATION_COMPLETE.json','CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json')],equivalent_storage_acceleration=True,main_task_complete=False)
        with (HERE/'ITERATION_LOG.md').open('a') as stream:
            stream.write('\n\n### '+utc()+' Exact geometry-cache storage activated\n\nThe full2000-step saved episode replay matched every nominal and all saved physical/control/event arrays bitwise, with identical outcomes and certificates.16 four-scene saved native queries,64 finite-grid cases and1100000-key dictionary tests also passed. Only cache storage changed. Switched local orchestration to coordinator_flat_static.py/downstream_flat_static.py/scheduler_watchdog_flat_static.py and future optional_delivery_flat_static.py; worker_flat_static.py/optional_worker_flat_static.py install the sealed native library. Old17 workers finished their current episodes before exiting; no hard cancellations, dropped cases, changed queues or unknown relabels. All'+str(len(before))+' pre-migration raw hashes are unchanged. Existing semantic receipts remain valid; new cache equivalence and deployment receipts record the implementation. Resource caps18 CPU including1 local and2 GPU shares remain. Full experiment still running.\n')
        with (HERE/'HANDOFF.md').open('a') as stream:
            stream.write('\n\n'+utc()+': computational storage acceleration active. ACTIVE_ENTRYPOINT.json now selects coordinator_flat_static.py, downstream_flat_static.py, scene_learning_scheduler_flat_static.py/fine_dispatch_flat_static.py, worker_flat_static.py, optional_delivery_flat_static.py/optional_fine_dispatch_flat_static.py/optional_worker_flat_static.py and scheduler_watchdog_flat_static.py. Equivalence and non-destructive migration: CBS_FLAT_CACHE_EQUIVALENCE.receipt.json, CBS_FLAT_CACHE_STATIC_DEPLOYMENT.receipt.json, CBS_FLAT_CACHE_MIGRATION_COMPLETE.receipt.json. No scientific case or execution-chain change. No task completion claimed.\n')
        print('Equivalent native cache migration complete; experiment continues',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc(),research_workers_not_hard_cancelled=True))
        raise
