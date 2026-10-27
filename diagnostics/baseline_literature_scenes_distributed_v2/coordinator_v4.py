"""Quota-aware unattended progression; only this experiment's jobs are owned."""
from experiment import *
import subprocess
import time
import fcntl
from resources import snapshot
from infrastructure_recovery_v3 import recover

def action(label,*arguments):
    path=HERE/'coordinator_actions'/f'{label}_{time.time_ns()}.log';path.parent.mkdir(exist_ok=True)
    with path.open('w') as stream:
        result=subprocess.run([sys.executable,'-B',str(HERE/arguments[0]),*arguments[1:]],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode:
        write(HERE/'COORDINATOR_ACTION_ERROR.json',dict(utc=utc(),label=label,arguments=arguments,log=str(path),returncode=result.returncode))
        raise RuntimeError(str(path))

def stage_ready(stage):return (HERE/stage/'WORK_QUEUE.receipt.json').exists()

def stage_done(stage):
    if not stage_ready(stage):return False
    return all((HERE/stage/'claims'/r['group']/'COMPLETE.json').exists() for r in read(HERE/stage/'GROUP_INDEX.json'))

def jobs_by_stage(state):
    live={j['job_id'] for j in state['jobs']};mapping={}
    for path in (HERE/'submissions').glob('*.json'):
        row=read(path)
        if row['job_id'] in live:mapping[row['job_id']]=row['stage']
    return mapping

def run_stages(stages,phase):
    from status import main as status
    while True:
        state=snapshot();mapping=jobs_by_stage(state);live=set(mapping)
        for stage in stages:recover(stage,live)
        pending=[s for s in stages if not stage_done(s)]
        status(phase)
        if not pending:
            # Do not advance while an exiting worker still holds a claim/job.
            if not any(s in stages for s in mapping.values()):return
        free=state['free_cpu_shares']
        if pending and free:
            from submit_reservation_v2 import submit
            from analysis_jobs import submit as submit_analysis
            counts={s:sum(v==s for v in mapping.values()) for s in pending}
            available={s:sum(not (HERE/s/'claims'/r['group']).exists() for r in read(HERE/s/'GROUP_INDEX.json')) for s in pending}
            for _ in range(min(free,sum(available.values()))):
                targets=[s for s in pending if available[s]>0]
                target=min(targets,key=lambda s:counts[s]);(submit_analysis if target.startswith('analysis_') else submit)(target);counts[target]+=1;available[target]-=1
        time.sleep(20)

def prepare(layer,kind,scene):
    stage=f'{layer}_{kind}_{scene}'
    if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('plan_'+stage,'study_cases.py',layer,kind,scene)
    if not stage_ready(stage):action('queue_'+stage,'work_queue.py',stage)
    return stage

def main():
    with (HERE/'COORDINATOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(HERE/'COORDINATOR_PID.json',dict(pid=os.getpid(),utc=utc()))
        cap='capability_lit_crowded_random'
        run_stages([cap],'crowded_single_agent_capability')
        if not (HERE/f'{cap}_QUALIFICATION.receipt.json').exists():action('qualify_crowded','stage_results.py','capability')
        stages=[prepare('distributed','baseline',s) for s in NAMES]
        stages += [prepare('distributed','whitebox',s) for s in NAMES]
        run_stages(stages,'distributed_safety_only')
        for scene in NAMES:
            if not (HERE/f'BASELINE_USABILITY_{scene}.json').exists():action('baseline_'+scene,'stage_results.py','distributed',scene)
        stages=[prepare('central','baseline',s) for s in NAMES]
        run_stages(stages,'central_comparison_safety_only')
        # Downstream implementation is sealed independently before use.
        while not (HERE/'DOWNSTREAM_READY.receipt.json').exists():
            from status import main as status
            status('central_complete_awaiting_downstream_implementation');time.sleep(30)
        from registry import verify_receipt
        verify_receipt('DOWNSTREAM_READY.receipt.json')
        action('downstream','downstream_v4.py')

if __name__=='__main__':main()
