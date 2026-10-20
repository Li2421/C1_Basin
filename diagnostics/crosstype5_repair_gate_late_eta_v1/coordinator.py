"""Durable sequential stages, no retries of claimed research work or unknowns."""
from common import *
from submit_aux import submit,failure,PYTHON
from resource_guard import queue,limits
import subprocess,time,shlex
from status import update as update_status

def admit(name,cpu,gpu=0):
    if (HERE/f'{name}_SUBMISSION.json').exists():
        r=read(HERE/f'{name}_SUBMISSION.json');return r['workers'],r['workers'] if r['gpu'] else 0
    while True:
        r=queue()
        cpu_limit,gpu_limit=limits()
        requested_gpu=min(gpu,gpu_limit);requested_cpu=requested_gpu if gpu else min(cpu,cpu_limit)
        if r['cpu_shards']+requested_cpu<=cpu_limit and r['gpu_shards']+requested_gpu<=gpu_limit:return requested_cpu,requested_gpu
        write(HERE/'COORDINATOR_STATUS.json',dict(utc=utc(),stage='waiting_for_resources',next_stage=name,requested_cpu=requested_cpu,requested_gpu=requested_gpu,queue=r));update_status('waiting_for_resources: '+name);time.sleep(30)

def command(script,*args):return ' '.join([shlex.quote(PYTHON),'-B',shlex.quote(str(HERE/script)),*[shlex.quote(str(a)) for a in args]])
def queued(job):
    result=subprocess.run(['squeue','-h','-r','-j',str(job),'-o','%i'],text=True,capture_output=True)
    if result.returncode and 'Invalid job id' not in result.stderr:raise RuntimeError(result.stderr)
    return bool(result.stdout.strip())
def successful(name):
    problem=failure(name)
    if not problem:return True
    # A leased slot may expire while still pending. Verify every physical
    # request in phase_complete; never infer missing evidence from job state.
    if (HERE/'RESOURCE_RELEASE_POLICY.json').exists() and datetime.now(timezone.utc)>=datetime.fromisoformat(read(HERE/'RESOURCE_RELEASE_POLICY.json')['deadline_utc']):
        if set(problem)<=set(('CANCELLED','TIMEOUT','DEADLINE')):return True
    if problem!=['Job unavailable; require completion artifact']:return False
    job=read(HERE/f'{name}_SUBMISSION.json')['job_id'].split(';')[0]
    result=subprocess.run(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State'],capture_output=True,text=True)
    if result.returncode or 'accounting storage is disabled' in result.stderr.lower():
        stage=name.removesuffix('_rollouts')
        # Official phase_complete independently verifies every genuine request,
        # raw trace and DB/local provenance; no rollout is retried here.
        count=read(HERE/f'{name}_SUBMISSION.json')['workers']
        return all((HERE/f'{stage}_WORKER_{i}_COMPLETE.json').exists() for i in range(count))
    states=[line.split('|')[1] for line in result.stdout.splitlines() if line.strip()]
    return bool(states) and all(s=='COMPLETED' for s in states)
def wait_job(name,artifact):
    job=read(HERE/f'{name}_SUBMISSION.json')['job_id'].split(';')[0]
    while queued(job):
        write(HERE/'COORDINATOR_STATUS.json',dict(utc=utc(),stage=name,waiting_for=str(artifact),queue=queue()));update_status(name);time.sleep(30)
    assert Path(artifact).exists(),dict(stage=name,failure=failure(name),missing_artifact=str(artifact))
def local(script,*args):
    subprocess.run([PYTHON,'-B',str(HERE/script),*map(str,args)],cwd=ROOT,check=True)
def cpu(name,script,args=(),artifact=None):
    if artifact and Path(artifact).exists():return
    admit(name,1)
    submit(name,[command(script,*args)],memory_gb=4)
    wait_job(name,artifact or HERE/f'{name}_COMPLETE.json')
def auxiliary_cpu(name,script,args=(),artifact=None):
    """Noncritical reporting failures become explicit gaps, never block rollouts."""
    try:
        cpu(name,script,args,artifact)
        return True
    except Exception as exc:
        path=HERE/'NONCRITICAL_STAGE_GAPS.json';rows=read(path) if path.exists() else []
        rows.append(dict(utc=utc(),stage=name,script=script,error=repr(exc),remaining_compute_continues=True,no_rollout_retries=True))
        write(path,rows);print('Noncritical stage gap; continuing:',name,repr(exc),flush=True)
        return False
def gpu(name,script,args=(),artifact=None,workers=1):
    if artifact and Path(artifact).exists():return
    workers,_=admit(name,workers,workers)
    suffix=' ${SLURM_ARRAY_TASK_ID}' if workers>1 else ''
    submit(name,[command(script,*args)+suffix],gpu=True,workers=workers)
    wait_job(name,artifact or HERE/f'{name}_COMPLETE.json')
def rollout(stage):
    if (HERE/f'{stage}_COMPLETE.receipt.json').exists():return
    if read(HERE/f'{stage}_EXECUTION_CASES.json'):
        name=f'{stage}_rollouts';count,_=admit(name,24);submit(name,['unset XLA_FLAGS',command('worker.py',stage)+' ${SLURM_ARRAY_TASK_ID}'],workers=count,memory_gb=4)
        job=read(HERE/f'{name}_SUBMISSION.json')['job_id'].split(';')[0]
        while queued(job):
            write(HERE/'COORDINATOR_STATUS.json',dict(utc=utc(),stage=stage,queue=queue()));update_status(stage);time.sleep(30)
        assert successful(name),dict(stage=stage,failure=failure(name))
    local('phase_complete.py',stage)

def main():
    # One coordinator owns stage transitions; official merger also has its own lock.
    import fcntl
    with (HERE/'COORDINATOR.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(HERE/'MODEL_TESTS.json')['passed']
        rollout('training')
        if not (HERE/'TRAINING_BUNDLE.receipt.json').exists():local('assemble_training.py')
        if not (HERE/'GENERATOR_TRAINING_PLAN.receipt.json').exists():local('schedule_generator.py','plan')
        gpu('generator_training','schedule_generator.py',('run',),HERE/'GENERATOR_WORKER_0_COMPLETE.json',5)
        config=read(HERE/'GENERATOR_TRAINING_PLAN.json');assert all((HERE/'schedule_generator_fits'/t['source_key']/'COMPLETE.receipt.json').exists() for t in config['tasks'])
        if not (HERE/'CRITIC_TRAINING_PLAN.receipt.json').exists():local('train_critic.py','plan')
        gpu('critic_training','train_critic.py',('run',),HERE/'CRITIC_WORKER_0_COMPLETE.json',5)
        config=read(HERE/'CRITIC_TRAINING_PLAN.json');assert all((HERE/'critic_fits'/t['mode']/t['source_key']/'COMPLETE.receipt.json').exists() for t in config['tasks'])
        cpu('select_models','select_models.py',artifact=HERE/'SELECTED_MODELS.receipt.json')
        gpu('inference_tests','inference.py',artifact=HERE/'FROZEN_INFERENCE_TESTS.receipt.json')
        gpu('gate_selection','gate_selection.py',artifact=HERE/'SELECTED_GATES.receipt.json')
        gpu('coverage_design','coverage.py',('design',),HERE/'coverage_EXECUTION_FROZEN.receipt.json')
        rollout('coverage')
        if not (HERE/'COVERAGE_BUNDLE.receipt.json').exists():local('coverage.py','assemble')
        auxiliary_cpu('coverage_statistics','coverage_statistics.py',artifact=HERE/'COVERAGE_STATISTICS.receipt.json')
        cpu('fresh_states','fresh_states.py',artifact=HERE/'NEW_EVAL_INPUTS.receipt.json')
        if not (HERE/'FROZEN_PIPELINE.receipt.json').exists():local('evaluation.py','freeze')
        gpu('evaluation_choices','evaluation.py',('choices',),HERE/'NEW_EVAL_CHOICES.receipt.json')
        if not (HERE/'evaluation_EXECUTION_FROZEN.receipt.json').exists():local('evaluation.py','design')
        rollout('evaluation')
        if not (HERE/'NEW_EVAL_BUNDLE.receipt.json').exists():local('evaluation.py','assemble')
        cpu('final_statistics','final_statistics.py',artifact=HERE/'FINAL_STATISTICS.receipt.json')
        local('close_and_deliver.py')
        write(HERE/'COORDINATOR_COMPLETE.json',dict(utc=utc(),all_stages_complete=True))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'COORDINATOR_BLOCKED.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc(),automatic_retries=False))
        write(HERE/'COORDINATOR_STATUS.json',dict(utc=utc(),stage='blocked',error=repr(exc),no_claim_of_continued_computation=True))
        update_status('blocked: '+repr(exc))
        raise
