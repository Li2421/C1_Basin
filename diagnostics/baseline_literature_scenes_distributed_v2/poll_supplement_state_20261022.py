"""Read-only foreground monitor for the complete revised task."""
from experiment import *
import collections


def main():
    management=read(HERE/'LOCAL_CPU_GUARD_STATUS.json')
    if isinstance(management.get('local_cpu'),int):os.sched_setaffinity(0,{management['local_cpu']})
    p=read(HERE/'LIVE_PROGRESS_PERCENT.json');w=read(HERE/'SCHEDULER_WATCHDOG.json');r=p['resources']
    issues=[str(path.relative_to(HERE)) for pattern in ('COORDINATOR_ACTION_ERROR.json','*/NEEDS_IMPLEMENTATION_REVIEW.json','LEARNING_GAP_REVISED_*.json','PARALLEL_NUMERIC_REVIEW_NEEDED_*.json','OPTIONAL_GAPS.json') for path in HERE.glob(pattern)]
    out=dict(utc=p['utc'],phase=p['phase'],main_done=p['main_registered_or_preflight_verified_completed'],main_plan=p['main_planned_logical_requests'],main_remaining=p['main_remaining'],main_percent=p['main_request_percent'],circle_labels=p['revised_learning_label_stages'][2]['completed'],circle_evaluation=p['revised_learning_eval_stages'][2]['completed'],primary_fixed={v['stage'].removeprefix('learning_eval_revised_20261021_trainfixed_'):v['completed'] for v in p.get('training_selected_fixed_eval_stages',[])},worker_cpus=r['scheduled_cpus'],cpu_cap=r['cpu_limit'],gpu_shares=r['gpu_shards'],coordinator_alive=w['coordinator_alive'],downstream_alive=w['downstream_alive'],release_guard_alive=read(HERE/'CPU_RELEASE_GUARD_WATCHDOG_20261023.json')['guard_alive'],issues=issues,complete=(HERE/'V2_COMPLETE.json').exists())
    out.update(statistics_fragments=len(list((HERE/'parallel_numeric_fragments_bootstrap_memo_20261022').glob('*.receipt.json'))),statistics_total=44,main_numbers_sealed=(HERE/'MIGRATION_NUMBERS.receipt.json').exists(),optional_done=sum(v['completed'] for v in p['second_stage_and_other_optional']),optional_plan=sum(v['planned'] for v in p['second_stage_and_other_optional']),all_v2_request_percent=p['all_v2_request_percent'],management_cpu=management.get('local_cpu'),management_shared_with_own_worker=management.get('shared_with_verified_own_slurm_worker',False),management_cpu_error=management.get('error'))
    out.update(slurm_job_states=dict(collections.Counter(j['state'] for j in r['jobs'])),slurm_job_names=dict(collections.Counter(j['name'] for j in r['jobs'])),optional_stage_counts={v['stage']:[v['completed'],v['planned']] for v in p['second_stage_and_other_optional']})
    print(json.dumps(out,ensure_ascii=False),flush=True)
    return out


if __name__=='__main__':main()
