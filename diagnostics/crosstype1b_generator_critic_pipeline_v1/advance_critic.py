"""Wait for model-free evidence, fit source models, and report development.

No path in this supervisor freezes a model or submits new-evaluation rollouts.
"""
from core import *
from submit_aux import submit,failure
from resource_guard import queue
import subprocess,time,fcntl,traceback

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def run(name,*args):subprocess.run([PYTHON,'-B',str(HERE/name),*map(str,args)],cwd=ROOT,check=True)
def exists(name):return (HERE/name).exists()


def stage():
    if exists('COLLECTION_ERROR.json'):raise RuntimeError('Collection error requires review')
    if not exists('SELECTABLE_SPACE.receipt.json'):return 'waiting_for_full_model_free_space_report'
    if not exists('CRITIC_TRAINING_PLAN.receipt.json'):run('train_critic.py','plan')
    plan=read(HERE/'CRITIC_TRAINING_PLAN.json')
    if not exists('critic_training_SUBMISSION.json'):
        usage=queue()
        if usage['reserved_cpu_shards']+2>18 or usage['reserved_gpu_shards']+2>2:return 'waiting_for_two_gpu_shards'
        submit('critic_training',[f'{PYTHON} -B {HERE}/train_critic.py run $SLURM_ARRAY_TASK_ID'],gpu=True,array=True)
    from train_critic import folder_for
    done=sum((folder_for(t)/'COMPLETE.receipt.json').exists() for t in plan['tasks'])
    if done<len(plan['tasks']):
        if failure('critic_training'):raise RuntimeError('GPU critic training job failed; inspect preserved claims and logs')
        return f'critic_training_{done}_of_{len(plan["tasks"])}'
    if not exists('DEVELOPMENT_CHOICES.receipt.json'):run('select_models.py')
    if not exists('DEVELOPMENT_RESULTS.receipt.json'):
        if not exists('development_statistics_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
            submit('development_statistics',[f'{PYTHON} -B {HERE}/development_statistics.py'])
        if exists('development_statistics_SUBMISSION.json') and failure('development_statistics'):
            raise RuntimeError('Development paired-statistics job failed')
        return 'development_statistics_running'
    return 'development_ready_for_review_and_freeze'


def main():
    with (HERE/'ADVANCE_CRITIC.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(HERE/'ADVANCE_CRITIC_PID.json',dict(pid=os.getpid(),utc=utc()))
        while True:
            try:status=stage()
            except Exception as e:
                write(HERE/'CRITIC_ADVANCE_ERROR.json',dict(utc=utc(),error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
                status='blocked_critic_error'
            write(HERE/'CRITIC_LIVE_STATUS.json',dict(utc=utc(),stage=status,new_evaluation_rollouts_submitted=False))
            if status in ('blocked_critic_error','development_ready_for_review_and_freeze'):return
            time.sleep(30)


if __name__=='__main__':main()
