"""Foreground-supervised sequential main line; non-main output never gates science."""
from common import *
from submit_aux import submit,PYTHON
from resource_guard import queue,limits
import subprocess,time,shlex,re

def cmd(script,*args,plot=False):
    python='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python' if plot else PYTHON
    return ' '.join(shlex.quote(str(x)) for x in (python,'-B',HERE/script,*args))

def active(job):
    r=subprocess.run(['squeue','-h','-j',job.split(';')[0],'-o','%i|%T'],capture_output=True,text=True)
    return bool(r.stdout.strip())

def diagnosis(receipt):
    job=receipt['job_id'].split(';')[0];r=subprocess.run(['scontrol','show','job',job,'--oneliner'],capture_output=True,text=True);text=r.stdout+'\n'
    for p in HERE.glob(f"{receipt['name']}_{job}_*.log"):
        # Logs are our own outputs. Keep diagnostic tails bounded.
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));text+=f.read().decode(errors='replace')+'\n'
    infrastructure=bool(re.search(r'JobState=(OUT_OF_MEMORY|NODE_FAIL|TIMEOUT|PREEMPTED|CANCELLED)\b',text)) or any(x in text.lower() for x in ('out-of-memory','oom-kill','oom_kill','due to time limit','launch failed','failed to launch'))
    return dict(utc=utc(),job_id=job,infrastructure=infrastructure,tail=text[-24000:])

def ensure(name,commands,done,gpu=False,workers=1,memory_gb=None,optional=False,pool=None):
    # pool: original worker IDs and per-ID completion filename; retries only unfinished IDs.
    def finished():return all((HERE/p).exists() for p in done)
    if finished():return True
    versions=[HERE/f'{name}_SUBMISSION.json',*[HERE/f'{name}_retry{i}_SUBMISSION.json' for i in range(1,4)]]
    receipts=[read(p) for p in versions if p.exists()]
    if receipts and active(receipts[-1]['job_id']):return False
    if receipts:
        failed=diagnosis(receipts[-1]);write(HERE/f'{name}_LAST_FAILURE.json',failed)
        if optional:
            write(HERE/f'{name}_OPTIONAL_GAP.json',dict(**failed,main_line_continues=True));return True
        if not failed['infrastructure']:
            write(HERE/f'{name}_NEEDS_IMPLEMENTATION_FIX.json',dict(**failed,independent_work_continues=True,completed_records_preserved=True));return False
        if len(receipts)>=4:
            write(HERE/f'{name}_INFRASTRUCTURE_GAP.json',dict(**failed,retries=3,completed_records_preserved=True));return False
    attempt=len(receipts);actual_name=name if not attempt else f'{name}_retry{attempt}';actual_workers=workers;actual_commands=commands
    submit_failures=HERE/f'{name}_SUBMIT_FAILURES.json'
    if submit_failures.exists() and len(read(submit_failures))>=4:return False
    if pool is not None:
        ids=[i for i in range(workers) if not (HERE/pool['marker'].format(worker=i)).exists()]
        if not ids:return True
        actual_workers=len(ids);array=' '.join(str(i) for i in ids);prefix=f'worker_ids=({array})';worker='"${worker_ids[${SLURM_ARRAY_TASK_ID:-0}]}"';actual_commands=[prefix,*[c.replace('WORKER_ID',worker) for c in commands]]
    q=queue();cpu_cap,gpu_cap=limits()
    if q['cpu_shards']+actual_workers>cpu_cap or q['gpu_shards']+(actual_workers if gpu else 0)>gpu_cap:return False
    if attempt and any(x in failed['tail'].lower() for x in ('out_of_memory','out-of-memory','oom-kill','oom_kill')):memory_gb=min(32,(memory_gb or (12 if gpu else 4))*2**attempt)
    try:submit(actual_name,actual_commands,gpu=gpu,workers=actual_workers,memory_gb=memory_gb)
    except subprocess.CalledProcessError as exc:
        failures=read(submit_failures) if submit_failures.exists() else [];failures.append(dict(utc=utc(),returncode=exc.returncode,infrastructure=True,commands_unchanged=True));write(submit_failures,failures)
    return False

def stage():
    # Independent diagnostics are retried/recorded separately, never gate new rollout submission.
    if (HERE/'CONTROLLER_FEATURES_V2.receipt.json').exists() and not (HERE/'FEATURE_DIAGNOSIS.receipt.json').exists():
        ensure('feature_diagnosis',[cmd('feature_diagnosis_c6.py')],['FEATURE_DIAGNOSIS.receipt.json'],memory_gb=4)
    if not (HERE/'CONTROLLER_FEATURES_V2.receipt.json').exists():return 'offline probes / feature assembly'
    if not ensure('training',['unset XLA_FLAGS',cmd('worker_c6.py','training','WORKER_ID',18)],[f'training_WORKER_{i}_COMPLETE.json' for i in range(18)],workers=18,memory_gb=4,pool=dict(marker='training_WORKER_{worker}_COMPLETE.json')):return 'training rollout labels'
    if not ensure('finish_training',[cmd('resume_steps_c6.py','finish_training')],['CRITIC_TRAINING_PLAN.receipt.json'],memory_gb=12):return 'training merge / bundle / model gates'
    if not ensure('critic_training',[cmd('train_critic.py','run','WORKER_ID')],[f'CRITIC_WORKER_{i}_COMPLETE.json' for i in range(2)],gpu=True,workers=2,memory_gb=16,pool=dict(marker='CRITIC_WORKER_{worker}_COMPLETE.json')):return 'source-only nested critic fitting'
    if not ensure('select_models',[cmd('select_c6_v2.py')],['SELECTED_MODELS.receipt.json'],memory_gb=12):return 'source-only checkpoint, blend and global trigger selection'
    if not ensure('freeze_choices',[cmd('resume_steps_c6.py','freeze_choices')],['NEW_EVAL_CHOICES.receipt.json'],gpu=True,memory_gb=20):return 'frozen inference replay / freeze / sealed fresh choices'
    if not ensure('design_evaluation',[cmd('evaluation_c6.py','design')],['evaluation_EXECUTION_FROZEN.receipt.json'],memory_gb=12):return 'fresh exact evaluation preflight'
    if not ensure('evaluation_safety',['unset XLA_FLAGS',cmd('worker_evaluation_c6.py','WORKER_ID',18,'safety')],[f'evaluation_safety_WORKER_{i}_COMPLETE.json' for i in range(18)],workers=18,memory_gb=4,pool=dict(marker='evaluation_safety_WORKER_{worker}_COMPLETE.json')):return 'frozen fresh Safety actions first'
    if not ensure('merge_safety',[cmd('resume_steps_c6.py','merge_safety')],['evaluation_SAFETY_MERGED.receipt.json'],memory_gb=8):return 'merge actual Safety traces for certified local reuse'
    if not ensure('evaluation_remaining',['unset XLA_FLAGS',cmd('worker_evaluation_c6.py','WORKER_ID',18,'remaining')],[f'evaluation_remaining_WORKER_{i}_COMPLETE.json' for i in range(18)],workers=18,memory_gb=4,pool=dict(marker='evaluation_remaining_WORKER_{worker}_COMPLETE.json')):return 'frozen fresh remaining actions / actual-trace local reuse'
    if not ensure('numeric_T1_T7',[cmd('resume_steps_c6.py','numeric_T1_T7')],['T1_T7_STATISTICS.receipt.json'],memory_gb=12):return 'numeric T1--T7 paired bootstrap'
    if not ensure('type_oracle',[cmd('oracle_c6.py','run','WORKER_ID',18)],[f'ORACLE_WORKER_{i}_COMPLETE.json' for i in range(18)],workers=18,memory_gb=4,pool=dict(marker='ORACLE_WORKER_{worker}_COMPLETE.json')):return 'type-information oracle donor-refit paired bootstrap'
    if not ensure('seal_numeric',[cmd('oracle_c6.py','collect')],['NUMERIC_RESULTS.receipt.json'],memory_gb=4):return 'seal numeric information-value table'
    ensure('plots',[cmd('plots_c6.py',plot=True)],['PLOTS.receipt.json'],memory_gb=4,optional=True)
    if not (HERE/'PLOTS.receipt.json').exists() and not (HERE/'plots_OPTIONAL_GAP.json').exists():return 'numeric work complete; optional plotting'
    if not ensure('closing_audit',[cmd('closing_audit_c6.py')],['INTEGRITY_COMPARISON.receipt.json'],memory_gb=12):return 'closing integrity audit'
    if not ensure('deliver',[cmd('deliver_c6.py')],['DELIVERY.receipt.json'],memory_gb=4):return 'delivery documents'
    return 'COMPLETE'

def main():
    while True:
        try:phase=stage()
        except Exception as exc:
            write(HERE/'COORDINATOR_EXCEPTION.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),main_agent_must_inspect=True));phase='exception recorded; inspect while preserving completed work'
        q=queue();progress={p:len(list((HERE/p/'raw').glob('*.json'))) for p in ('probe','training','evaluation')};
        if phase=='training rollout labels' and q['cpu_shards']>=limits()[0] and not any(r['name'].startswith('crosstype6_training') and r['counts'] for r in q['rows']):phase='training unfinished retained; waiting for global CPU capacity'
        status=dict(utc=utc(),phase=phase,progress=progress,queue=q,foreground_primary_supervision=True);write(HERE/'MONITOR_STATUS.json',status)
        (HERE/'STATUS.md').write_text('# 跨类型6运行状态\n\n'+status['utc']+'\n\n当前阶段：'+phase+'。\n\n实际原始记录：'+json.dumps(progress,ensure_ascii=False)+'。\n\n'+f"当前全用户CPU/GPU占用{q['cpu_shards']}/{q['gpu_shards']}，上限18/2。完成/unknown保留；非主线输出失败不阻断数值；基础设施只重提未完成同案例同种子，最多3次。主agent保持当前回合前台监督。\n")
        print(json.dumps(dict(utc=utc(),phase=phase,progress=progress,cpu=q['cpu_shards'],gpu=q['gpu_shards'])),flush=True)
        if phase=='COMPLETE':write(HERE/'COORDINATOR_COMPLETE.json',dict(utc=utc(),completed=True));return
        time.sleep(20)

if __name__=='__main__':main()
