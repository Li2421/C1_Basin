"""Optional work follows sealed main numbers; failures become explicit gaps."""
from experiment import *
from coordinator_pipelined import action,stage_ready,stage_done,jobs_by_stage
from infrastructure_recovery_active import recover
from resources_released import snapshot,require_capacity
from registry import verify_receipt
import time
import subprocess
import fcntl

def submit(stage):
    verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json');verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
    analysis=stage.startswith('analysis_')
    if not analysis:verify_receipt(f'{stage}_PREFLIGHT.receipt.json')
    script='optional_analysis_cached.sbatch' if analysis else 'optional.sbatch'
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);before=require_capacity()
        memory=read(HERE/stage/'MEMORY_OVERRIDE_GB.json')['memory_gb'] if (HERE/stage/'MEMORY_OVERRIDE_GB.json').exists() else 4
        job=subprocess.check_output(['sbatch','--parsable',f'--mem={memory}G',f'--export=ALL,BLV2_STAGE={stage}',str(HERE/script)],text=True).strip()
        write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),stage=stage,job_id=job,resource_snapshot=before,script_sha256=sha(HERE/script)))
        return job

def gap(stage,reason):
    path=HERE/'OPTIONAL_GAPS.json';rows=read(path) if path.exists() else []
    if not any(r['stage']==stage for r in rows):rows.append(dict(utc=utc(),stage=stage,reason=reason,main_numbers_unchanged=True))
    write(path,rows)

def run(stages,phase):
    from status_released import main as status
    blocked=set()
    while True:
        state=snapshot();mapping=jobs_by_stage(state)
        for stage in stages:
            recover(stage,set(mapping))
            issue=HERE/stage/'NEEDS_IMPLEMENTATION_REVIEW.json'
            if issue.exists():
                blocked.add(stage);gap(stage,dict(kind='Optional implementation failure',review=read(issue)))
                for job,s in mapping.items():
                    if s==stage:subprocess.run(['scancel',job],check=True)
        pending=[s for s in stages if s not in blocked and not stage_done(s)]
        status(phase)
        if not pending and not any(s in stages for s in mapping.values()):return [s for s in stages if s not in blocked]
        available={s:sum(not (HERE/s/'claims'/r['group']).exists() for r in read(HERE/s/'GROUP_INDEX.json')) for s in pending}
        counts={s:sum(v==s for v in mapping.values()) for s in pending}
        for _ in range(min(state['free_cpu_shares'],sum(available.values()))):
            target=min([s for s in pending if available[s]],key=lambda s:counts[s]);submit(target);counts[target]+=1;available[target]-=1
        time.sleep(20)

def prepare(kind,scene):
    stage=f'optional_{kind}_{scene}'
    try:
        if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('plan_'+stage,'optional_cases.py',kind,scene)
        if not stage_ready(stage):action('queue_'+stage,'work_queue.py',stage)
        return stage
    except Exception as exc:
        gap(stage,str(exc));return None

def main():
    verify_receipt('MIGRATION_NUMBERS.receipt.json');verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json')
    if (HERE/'V2_COMPLETE.json').exists():return
    if not (HERE/'MAIN_ADDITIONAL_METRICS.receipt.json').exists():action('additional_main_metrics','derived_metrics.py')
    stages=[s for scene in NAMES if (s:=prepare('dynamics',scene))]
    complete=run(stages,'paper_dynamics_optional')
    other=[]
    for kind,scenes in [('full_circle',['lit_circle_five']),('local_progress',list(NAMES)),('nominal',['lit_crowded_random'])]:
        planned=[s for scene in scenes if (s:=prepare(kind,scene))]
        other += run(planned,'optional_'+kind)
    from analysis_jobs_current import prepare as prepare_analysis
    analysis=[]
    for stage in complete+other:
        # No safety layer was executed in the nominal background arm.
        if stage.startswith('optional_nominal_'):continue
        try:analysis.append(prepare_analysis(stage))
        except Exception as exc:gap('analysis_'+stage,str(exc))
    run(analysis,'optional_saved_trace_diagnosis')
    if not (HERE/'OPTIONAL_NUMBERS.receipt.json').exists():action('optional_numbers','optional_results.py')
    action('official_final_merge','db_merge.py')
    if not (HERE/'DB_POSTFLIGHT.receipt.json').exists():action('postflight','postflight.py')
    # No scientific image may change a controller or physical evidence.
    plots=HERE/'FINAL_PLOTS.log'
    if not (HERE/'FIGURES.receipt.json').exists():
        with plots.open('w') as stream:
            result=subprocess.run(['/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python','-B',str(HERE/'plot_results.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:gap('plots',dict(log=str(plots),returncode=result.returncode))
    # Every experiment worker has ended before the final source audit.
    state=snapshot();assert not any(j['name'].startswith('blv2_') for j in state['jobs']),state
    if not (HERE/'INTEGRITY_END.receipt.json').exists():action('integrity_end','integrity.py','end')
    if not (HERE/'FINAL_REPORTS.receipt.json').exists():action('final_report','final_reports_pipelined.py')
    from status_released import main as status
    status('complete_with_explicit_optional_gaps' if (HERE/'OPTIONAL_GAPS.json').exists() else 'complete')
    write(HERE/'V2_COMPLETE.json',dict(utc=utc(),main_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),source_audit=sha(HERE/'INTEGRITY_END.json'),v1_readonly=True))

if __name__=='__main__':main()
