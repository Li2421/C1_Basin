"""Read-only foreground check of progress, scheduling and outstanding reviews."""
import collections
import json
import os
from datetime import datetime,timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    here=Path(__file__).resolve().parent
    os.sched_setaffinity(0,{read(here/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    progress=read(here/'LIVE_PROGRESS_PERCENT.json')
    watcher=read(here/'SCHEDULER_WATCHDOG.json')
    resources=progress['resources']
    issues=[str(p.relative_to(here)) for pattern in (
        'LEARNING_GAP_REVISED_*.json','*/NEEDS_IMPLEMENTATION_REVIEW.json',
        'COORDINATOR_ACTION_ERROR.json','PARALLEL_NUMERIC_REVIEW_NEEDED_20261023.json','OPTIONAL_GAPS.json',
        'COMPLETED_*_COMPARISON_ERROR.json') for p in here.glob(pattern)]
    out=dict(utc=progress['utc'],watchdog_utc=watcher['utc'],
             progress_age_seconds=round((datetime.now(timezone.utc)-datetime.fromisoformat(progress['utc'])).total_seconds()),
             phase=progress['phase'],
             main_done=progress['main_registered_or_preflight_verified_completed'],
             main_remaining=progress['main_remaining'],percent=round(progress['main_request_percent'],4),
             remaining_fixed={r['stage']:r['remaining'] for r in progress['revised_fixed_stages'] if r['remaining']},
             circle_learning=[{k:r[k] for k in ('stage','completed','remaining')} for key in (
                 'revised_learning_label_stages','revised_learning_eval_stages')
                 for r in progress[key] if 'circle_five' in r['stage']],
             worker_cpus=resources['scheduled_cpus'],gpu_shares=resources['gpu_shards'],
             job_states=dict(collections.Counter(j['state'] for j in resources['jobs'])),
             job_names=dict(collections.Counter(j['name'] for j in resources['jobs'])),
             coordinator_alive=watcher['coordinator_alive'],downstream_alive=watcher['downstream_alive'],
             action_error=watcher['action_error'],issues=issues,
             release_guard_alive=read(here/'CPU_RELEASE_GUARD_WATCHDOG_20261023.json')['guard_alive'],
             complete=(here/'V2_COMPLETE.json').exists())
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
