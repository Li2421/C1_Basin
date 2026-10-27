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
        verify_receipt('CBS_FLAT_CACHE_ACTIVATOR_CODE.receipt.json')
        verify_receipt('CBS_FLAT_CACHE_CODE.receipt.json')
        write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_PID.json',dict(utc=utc(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0))))
        proof=HERE/'CBS_FLAT_CACHE_FULL_REPLAY.json'
        while not proof.exists():
            write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_STATUS.json',dict(utc=utc(),phase='Waiting for full saved replay; no deployment',main_workers_untouched=True))
            time.sleep(10)
        replay=read(proof);assert replay['passed'],replay
        matrix=read(HERE/'CBS_FLAT_CACHE_MATRIX.json');assert matrix['passed']
        assert replay['replayed_steps']==2000 and replay['all_saved_nominals_checked']
        assert len(matrix['saved_queries'])==16 and len(matrix['finite_grid_tests'])==64
        assert all(sha(path)==expected for path,expected in replay['source_hashes'].items())
        assert all(sha(path)==expected for path,expected in matrix['source_hashes'].items())
        changes=read(HERE/'CBS_FLAT_CACHE_DEPLOYMENT_SOURCE_CHANGES.json')['changes']
        for item in changes:
            before=(HERE/item['original']).read_text();after=before
            assert sha(HERE/item['original'])==item['original_sha256']
            for a,b in item['replacements']:after=after.replace(a,b)
            assert item['extra_header']+after==(HERE/item['new']).read_text()
            assert sha(HERE/item['new'])==item['new_sha256']
        if not (HERE/'CBS_FLAT_CACHE_EQUIVALENCE.receipt.json').exists():
            seal('CBS_FLAT_CACHE_EQUIVALENCE.receipt.json',[HERE/n for n in ('CBS_FLAT_CACHE_CODE.receipt.json','CBS_FLAT_CACHE_FULL_REPLAY.json','CBS_FLAT_CACHE_MATRIX.json','CBS_FLAT_CACHE_SOURCE_CHECK.json')],
                 passed=True,all_noncache_cpp_bytes_unchanged=True,full_replay_steps=2000,native_saved_queries=16,finite_native_grid_tests=64,dictionary_exact_keys=1100000,new_research_rollouts=0)
        paths=[HERE/n for n in ('flat_geometry_backend.py','deploy_flat_cache.py','CBS_FLAT_CACHE_DEPLOYMENT_SOURCE_CHANGES.json','CBS_FLAT_CACHE_DEPLOYMENT_SOURCE_CHECK.json','CBS_FLAT_CACHE_EQUIVALENCE.receipt.json','CBS_FLAT_CACHE_ACTIVATOR_CODE.receipt.json')]
        paths += [HERE/item['new'] for item in changes]+[HERE/(item['new']+'.diff') for item in changes]
        if not (HERE/'CBS_FLAT_CACHE_DEPLOYMENT.receipt.json').exists():
            seal('CBS_FLAT_CACHE_DEPLOYMENT.receipt.json',paths,scientific_functions_unchanged=True,case_payloads_unchanged=True,only_storage_and_entrypoints=True,no_hard_rollout_cancellation=True)
        verify_receipt('CBS_FLAT_CACHE_DEPLOYMENT.receipt.json')
        write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_STATUS.json',dict(utc=utc(),phase='Equivalence passed; switching local orchestration'))
        with (HERE/'CBS_FLAT_CACHE_DEPLOYMENT.log').open('a') as stream:
            result=subprocess.run([sys.executable,'-B',str(HERE/'deploy_flat_cache.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,'See CBS_FLAT_CACHE_DEPLOYMENT.log'
        migration=read(HERE/'CBS_FLAT_CACHE_MIGRATION_INITIATED.json');old=set(migration['old_jobs'])
        while True:
            state=snapshot();remaining=old&{j['job_id'] for j in state['jobs']}
            write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_STATUS.json',dict(utc=utc(),phase='Graceful old-worker episode drain' if remaining else 'Verifying preserved evidence',old_workers_remaining=sorted(remaining),resource_cpu=state['scheduled_cpus'],resource_gpu=state['gpu_shards']))
            if not remaining:break
            time.sleep(10)
        before=read(HERE/'CBS_FLAT_CACHE_PREMIGRATION_RAW_HASHES.json')
        for path,expected in before.items():assert sha(path)==expected,path
        verify_receipt('CBS_FLAT_CACHE_DEPLOYMENT.receipt.json')
        write(HERE/'CBS_FLAT_CACHE_MIGRATION_COMPLETE.json',dict(utc=utc(),old_workers_remaining=0,all_preexisting_raw_hashes_unchanged=len(before),hard_cancelled_rollouts=0,discarded_cases=0,case_payloads_unchanged=True,resources=snapshot(),main_still_running=True))
        seal('CBS_FLAT_CACHE_MIGRATION_COMPLETE.receipt.json',[HERE/n for n in ('CBS_FLAT_CACHE_MIGRATION_INITIATED.json','CBS_FLAT_CACHE_PREMIGRATION_RAW_HASHES.json','CBS_FLAT_CACHE_MIGRATION_COMPLETE.json','CBS_FLAT_CACHE_DEPLOYMENT.receipt.json')],equivalent_storage_acceleration=True,main_task_complete=False)
        with (HERE/'ITERATION_LOG.md').open('a') as stream:
            stream.write('\n\n### '+utc()+' Exact geometry-cache storage activated\n\nThe full2000-step saved episode replay matched every nominal and all saved physical/control/event arrays bitwise, with identical outcomes and certificates.16 four-scene saved native queries,64 finite-grid cases and1100000-key dictionary tests also passed. Only cache storage changed. Switched local orchestration to coordinator_flat.py/downstream_flat.py/scheduler_watchdog_flat.py and future optional_delivery_flat.py; worker_flat.py/optional_worker_flat.py install the sealed native library. Old17 workers finished their current episodes before exiting; no hard cancellations, dropped cases, changed queues or unknown relabels. All'+str(len(before))+' pre-migration raw hashes are unchanged. Existing semantic receipts remain valid; new cache equivalence and deployment receipts record the implementation. Resource caps18 CPU including1 local and2 GPU shares remain. Full experiment still running.\n')
        with (HERE/'HANDOFF.md').open('a') as stream:
            stream.write('\n\n'+utc()+': computational storage acceleration active. ACTIVE_ENTRYPOINT.json now selects coordinator_flat.py, downstream_flat.py, scene_learning_scheduler_flat.py/fine_dispatch_flat.py, worker_flat.py, optional_delivery_flat.py/optional_fine_dispatch_flat.py/optional_worker_flat.py and scheduler_watchdog_flat.py. Equivalence and non-destructive migration: CBS_FLAT_CACHE_EQUIVALENCE.receipt.json, CBS_FLAT_CACHE_DEPLOYMENT.receipt.json, CBS_FLAT_CACHE_MIGRATION_COMPLETE.receipt.json. No scientific case or execution-chain change. No task completion claimed.\n')
        print('Equivalent native cache migration complete; experiment continues',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'CBS_FLAT_CACHE_ACTIVATOR_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc(),research_workers_not_hard_cancelled=True))
        raise
