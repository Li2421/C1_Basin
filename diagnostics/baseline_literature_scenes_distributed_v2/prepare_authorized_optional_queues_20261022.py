"""Prepare existing authorized future queues on the management CPU;submit nothing."""
from experiment import *
from registry import verify_receipt
from optional_cases_correction_revised_20261021 import build
from work_queue import build as queue
import optional_fine_dispatch_spatial_mask_20261022 as fine
import fcntl


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    with (HERE/'OPTIONAL_FUTURE_PREPARATION.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('MIGRATION_NUMBERS.receipt.json')
        verify_receipt('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json')
        assert read(HERE/'STATUS.json')['phase'] == 'paper_dynamics_optional'
        scopes = [('full_circle','lit_circle_five')]
        scopes += [('local_progress',s) for s in NAMES]
        scopes += [('nominal','lit_crowded_random')]
        names = ['optional_'+k+'_'+s for k,s in scopes]
        history = HERE/'OPTIONAL_FUTURE_PREPARATION_HISTORY_20261022.json'
        if not history.exists():
            write(history, dict(utc=utc(), stages=names,
                historical_paths=['MIGRATION_NUMBERS.receipt.json','COMPACT_REVISED_MAIN_RESULTS_20261021.json','OPTIONAL_SCOPE.json','CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json','CORRECTION_REVISION.receipt.json'],
                historical_compatibility={
                    'full_circle':'Existing reciprocal circle and paper-dynamics evidence has a different responsibility or engine fingerprint;identical full-responsibility entries,if any,are reused only by official preflight.',
                    'local_progress':'Main broadcast-progress evidence uses a different switch signal;factory preserves the previously sealed per-robot-progress fingerprint and contact gate. Exact entries,if any,are reused by official preflight.',
                    'nominal':'Main no-wall crowded states are reused;executing without safety differs from main safety arms. Exact nominal entries,if any,are reused by official preflight.'},
                evidence_gap='These exact previously authorized optional scopes have no completed own stage;official preflight determines all genuinely missing execution identities before later submission.',
                current_paper_dynamics_overlap='Different engine/responsibility/switch/no-safety fingerprints;future preparation cannot duplicate current in-flight dynamics executions.',
                requested_scope_unchanged=True, new_training=0, rollout_submissions_by_this_helper=0,
                source_hashes={p:sha(HERE/p) for p in ['optional_cases_correction_revised_20261021.py','optional_cases.py','work_queue.py','optional_fine_dispatch_spatial_mask_20261022.py']}))
            seal('OPTIONAL_FUTURE_PREPARATION_HISTORY_20261022.receipt.json', [history, Path(__file__), HERE/'OPTIONAL_SCOPE.json', HERE/'CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json'], before_preflight=True, before_future_rollouts=True)
        else:
            verify_receipt('OPTIONAL_FUTURE_PREPARATION_HISTORY_20261022.receipt.json')
        results = []
        for (kind,scene), stage in zip(scopes,names):
            assert read(HERE/'STATUS.json')['phase'] == 'paper_dynamics_optional'
            if not (HERE/(stage+'_PREFLIGHT.receipt.json')).exists():
                build(kind,scene)
            verify_receipt(stage+'_PREFLIGHT.receipt.json')
            if not (HERE/stage/'WORK_QUEUE.receipt.json').exists():
                queue(stage)
            fine.prepare(stage)
            cases = read(HERE/(stage+'_CASES.json'))
            results.append(dict(stage=stage, planned=len(cases), missing=len(read(HERE/(stage+'_MISSING_CASES.json'))), reused=len(read(HERE/(stage+'_REUSED_ROWS.json'))), canonical_preflight=True, immutable_queue_ready=True))
            print('PREPARED',stage,results[-1],flush=True)
        output = HERE/'OPTIONAL_FUTURE_PREPARATION_20261022.json'
        assert not output.exists()
        write(output,dict(utc=utc(),stages=results,total_planned=sum(r['planned'] for r in results),new_training=0,rollout_submissions_by_this_helper=0,scientific_worker_signals=0,purpose='Overlap only official planning/preflight/queue preparation with current science;ordinary sole coordinator retains all execution ordering and admission.'))
        seal('OPTIONAL_FUTURE_PREPARATION_20261022.receipt.json', [Path(__file__),history,output,*[HERE/(s+'_PREFLIGHT.receipt.json') for s in names],*[HERE/s/'WORK_QUEUE.receipt.json' for s in names],*[HERE/s/'FINE_QUEUE.receipt.json' for s in names]],before_future_rollout_submission=True,no_science_or_statistics_changes=True,rollout_submissions_by_this_helper=0)
        with (HERE/'ITERATION_LOG.md').open('a') as f:
            f.write('\n'+utc()+'：在论文动力学worker计算期间，管理核提前完成原授权圆形全责、四场景逐机器人进度和拥挤无安全背景共36416逻辑请求的官方缓存预检及原队列封存。历史兼容性记录先于preflight；只提交缺口的既有协调器与阶段顺序不改。本helper提交rollout=0，训练=0，科学worker信号=0。见OPTIONAL_FUTURE_PREPARATION_20261022.receipt.json。\n')
        print('PASS all authorized future queues ready;no rollout submitted',flush=True)


if __name__ == '__main__':
    main()
