"""Read critic failures before retry; preserve completed fits and other scenes."""
from experiment import *
from coordinator import jobs_by_stage
from resources import snapshot
from infrastructure_recovery_v3 import classify
from critic_jobs import submit
import time

def train(scenes):
    from status import main as status
    blocked=set();reviewed=set()
    while True:
        state=snapshot();mapping=jobs_by_stage(state)
        for path in (HERE/'submissions').glob('*.json'):
            row=read(path);stage=row['stage'];job=row['job_id']
            if not stage.startswith('critic_') or job in mapping or job in reviewed:continue
            scene=stage.removeprefix('critic_')
            if scene not in scenes:continue
            reviewed.add(job)
            if (HERE/f'FROZEN_SCENE_CRITIC_{scene}.receipt.json').exists():continue
            review=classify(stage,job);write(HERE/'critic_failure_reviews'/f'{job}.json',review)
            if review['infrastructure']:
                if any(w in review['tail'].lower() for w in ('resource_exhausted','cuda_error_out_of_memory','out of memory')):
                    write(HERE/'CRITIC_CONCURRENCY.json',dict(utc=utc(),maximum_simultaneous_jobs=1,reason='Observed GPU memory failure; parameters and seeds unchanged'))
                continue
            blocked.add(scene)
            write(HERE/f'LEARNING_GAP_{scene}.json',dict(utc=utc(),required_by_protocol=True,reason='Critic implementation failed; source labels retained and other scenes continue',review=str(HERE/'critic_failure_reviews'/f'{job}.json')))
        pending=[s for s in scenes if s not in blocked and not (HERE/f'FROZEN_SCENE_CRITIC_{s}.receipt.json').exists()]
        if not pending:return [s for s in scenes if s not in blocked]
        cap=read(HERE/'CRITIC_CONCURRENCY.json')['maximum_simultaneous_jobs'] if (HERE/'CRITIC_CONCURRENCY.json').exists() else 5
        for scene in pending:
            current=snapshot();live=jobs_by_stage(current)
            if 'critic_'+scene in live.values():continue
            if sum(s.startswith('critic_') for s in live.values())>=cap:break
            if current['free_cpu_shares'] and current['free_gpu_shards']:submit(scene)
        status('scene_critics_training');time.sleep(20)
