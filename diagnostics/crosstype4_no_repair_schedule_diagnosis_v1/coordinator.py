"""Serial scientific stages, admission-checked Slurm, no implicit reruns."""
from common import *
from submit_aux import submit,PYTHON,failure
import time,fcntl,traceback,shlex
from resource_guard import queue

def command(script,*args):return ' '.join([PYTHON,'-B',shlex.quote(str(HERE/script)),*map(str,args)])
def exists(name):return (HERE/name).exists()
def active(name):
    path=HERE/f'{name}_SUBMISSION.json'
    if not path.exists():return False
    job=read(path)['job_id'].split(';')[0]
    return any(r['job'].split('_')[0]==job for r in queue()['rows'])
def ensure(name,commands,artifact,gpu=False,desired=1):
    if exists(artifact):return True
    path=HERE/f'{name}_SUBMISSION.json'
    if path.exists():
        if active(name):return False
        raise RuntimeError(f'{name} ended without {artifact}; no automatic retry; inspect logs and claims')
    q=queue();available=min(desired,18-q['cpu_shards'],2-q['gpu_shards'] if gpu else desired)
    if available<=0:return False
    submit(name,commands,gpu=gpu,workers=available);return False

def rollout(stage,name):
    artifact=f'{stage}_COMPLETE.receipt.json'
    if exists(artifact):return True
    if not exists(f'{name}_SUBMISSION.json'):
        q=queue();n=min(18,18-q['cpu_shards'])
        if n<=0:return False
        submit(name,[command('rollout_worker_v3.py',stage,'${SLURM_ARRAY_TASK_ID:-0}')],workers=n);return False
    if active(name):return False
    return ensure(stage+'_collect',[command('phase_complete.py',stage)],artifact)

def status(stage):
    q=queue();counts={}
    for s in ('A','B_training','B_evaluation'):
        actual_dir=HERE/s/'raw';local_dir=HERE/s/'local_equivalence'
        counts[s]=dict(actual=len(list(actual_dir.glob('*.json'))) if actual_dir.exists() else 0,dynamic_local=len(list(local_dir.glob('*.json'))) if local_dir.exists() else 0)
    fits=HERE/'critic_fits';counts['critic_fits_complete']=len(list(fits.glob('*/*/COMPLETE.receipt.json'))) if fits.exists() else 0
    write(HERE/'MONITOR_STATUS.json',dict(utc=utc(),pid=os.getpid(),stage=stage,counts=counts,queue=q,not_a_completion_claim=True))

def step():
    # A was already submitted with its sealed version2 worker.
    if not exists('A_COMPLETE.receipt.json'):
        status('A 补跑')
        if active('A_diagnosis'):return False
        ensure('A_collect',[command('phase_complete.py','A')],'A_COMPLETE.receipt.json');return False
    if not exists('A_DIAGNOSIS.receipt.json'):
        status('A 分解与5000次bootstrap');ensure('A_analysis',[command('a_statistics.py')],'A_DIAGNOSIS.receipt.json');return False
    if not exists('B_training_EXECUTION_FROZEN.receipt.json'):
        status('B候选/新状态/训练标签preflight')
        cmds=[]
        if not exists('NEW_EVAL_INPUTS.receipt.json'):cmds.append(command('b_fresh.py'))
        cmds.append(command('b_design.py','training'));ensure('B_prepare',cmds,'B_training_EXECUTION_FROZEN.receipt.json');return False
    if not exists('B_training_COMPLETE.receipt.json'):
        status('B训练标签补跑');rollout('B_training','B_training_rollouts');return False
    if not exists('CRITIC_TRAINING_PLAN.receipt.json'):
        status('B训练标签装配')
        cmds=[]
        if not exists('TRAINING_BUNDLE.receipt.json'):cmds.append(command('b_design.py','assemble'))
        if not exists('B_IMPLEMENTATION_FINAL.receipt.json'):cmds.append(command('implementation_manifest.py'))
        cmds.append(command('train_critic.py','plan'));ensure('B_training_plan',cmds,'CRITIC_TRAINING_PLAN.receipt.json');return False
    fits=HERE/'critic_fits';completed=len(list(fits.glob('*/*/COMPLETE.receipt.json')))
    if completed!=382:
        status(f'B critic训练 {completed}/382')
        if not exists('B_critic_SUBMISSION.json'):
            q=queue();n=min(2,2-q['gpu_shards'],18-q['cpu_shards'])
            if n>0:submit('B_critic',[command('train_critic.py','run','${SLURM_ARRAY_TASK_ID:-0}')],gpu=True,workers=n)
        elif not active('B_critic'):raise RuntimeError(f'Critic workers ended at {completed}/382; preserve claims, no automatic retraining')
        return False
    if not exists('DEVELOPMENT_RESULTS.receipt.json'):
        status('B源内选择及19折开发评分');ensure('B_selection',[command('select_models.py'),command('b_development.py')],'DEVELOPMENT_RESULTS.receipt.json');return False
    if not exists('FROZEN_INFERENCE_TESTS.receipt.json'):
        status('B冻结推理逐位复现');ensure('B_inference_check',[command('inference.py')],'FROZEN_INFERENCE_TESTS.receipt.json',gpu=True);return False
    if not exists('FROZEN_PIPELINE.receipt.json'):
        status('B模型冻结');ensure('B_freeze',[command('b_freeze.py')],'FROZEN_PIPELINE.receipt.json');return False
    if not exists('NEW_EVAL_CHOICES.receipt.json'):
        status('B新状态开局选择密封');ensure('B_choices',[command('b_choices.py')],'NEW_EVAL_CHOICES.receipt.json',gpu=True);return False
    if not exists('B_evaluation_EXECUTION_FROZEN.receipt.json'):
        status('B新状态精确preflight');ensure('B_eval_design',[command('b_evaluation.py','design')],'B_evaluation_EXECUTION_FROZEN.receipt.json');return False
    if not exists('B_evaluation_COMPLETE.receipt.json'):
        status('B新状态闭环评估');rollout('B_evaluation','B_evaluation_rollouts');return False
    if not exists('B_RESULTS.receipt.json'):
        status('B1–B5配对统计')
        cmds=[]
        if not exists('NEW_EVAL_BUNDLE.receipt.json'):cmds.append(command('b_evaluation.py','assemble'))
        cmds.append(command('b_statistics.py'));ensure('B_analysis',cmds,'B_RESULTS.receipt.json');return False
    if not exists('DELIVERY.receipt.json'):
        status('结束完整性审计与交付')
        cmds=[]
        if not exists('INTEGRITY_COMPARISON.receipt.json'):cmds.append(command('close_audit.py'))
        cmds.append(command('deliver.py'));ensure('delivery',cmds,'DELIVERY.receipt.json');return False
    if active('delivery'):return False
    status('完成' if read(HERE/'DELIVERY.receipt.json')['complete'] else '审计差异待复核');write(HERE/'FINAL_RESOURCE_RELEASE.json',queue());write(HERE/'MONITOR_COMPLETION.json',dict(utc=utc(),complete=read(HERE/'DELIVERY.receipt.json')['complete'],delivery_receipt=str(HERE/'DELIVERY.receipt.json')));return True

def main():
    with (HERE/'COORDINATOR.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(HERE/'COORDINATOR_PID.json',dict(pid=os.getpid(),utc=utc()))
        try:
            while not step():time.sleep(30)
        except Exception:
            write(HERE/'MONITOR_FAILURE.json',dict(utc=utc(),traceback=traceback.format_exc(),no_automatic_reruns=True));raise
if __name__=='__main__':main()
