"""Advance gates and training only. Deliberately cannot open locked arm labels."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time
import sys
import os
import fcntl
from resource_guard import require,queue

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'

def read(name):return json.loads((HERE/name).read_text())
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def failure(name):
    path=HERE/f'{name}_SUBMISSION.json'
    if not path.exists():return None
    job=read(path.name)['job_id'].split(';')[0]
    # This workstation has accounting storage disabled. Active/recent jobs are
    # available from slurmctld even when sacct cannot query the accounting DB.
    result=subprocess.run(['scontrol','show','job',job,'--oneliner'],text=True,capture_output=True)
    if result.returncode:
        active=subprocess.check_output(['squeue','-h','-r','-j',job,'-o','%T'],text=True).splitlines()
        if active:
            states=active
        else:
            return ['Job absent from controller and required completion evidence missing; inspect retained log']
    else:
        states=[token.split('=',1)[1] for token in result.stdout.split() if token.startswith('JobState=')]
    bad=[s.strip() for s in states if any(s.strip().startswith(v) for v in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'))]
    return bad or None

def submit(name,commands,gpu=False,array=None):
    receipt=HERE/f'{name}_SUBMISSION.json'
    if receipt.exists():return
    count=2 if array else 1
    before=require(cpu=count,gpu=count if gpu else 0)
    path=HERE/f'{name}.sbatch'
    script=f'''#!/bin/bash
#SBATCH --job-name=crosstype1_{name}
#SBATCH --partition={'gpu' if gpu else 'cpu'}
#SBATCH --cpus-per-task=1
#SBATCH --mem={12 if gpu else 4}G
#SBATCH --time=24:00:00
#SBATCH --output={HERE}/{name}_%A_%a.log
'''
    if gpu:script+='#SBATCH --gres=shard:1\n'
    if array:script+='#SBATCH --array=0-1%2\n'
    script+=f'''set -euo pipefail
cd {ROOT}
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS={'cuda' if gpu else 'cpu'}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
'''
    script+='\n'.join(commands)+'\n';path.write_text(script)
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    data=dict(utc=utc(),queue_before=before,files=files,script_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(f'{name}_SUBMISSION_PREFLIGHT.json',data)
    data['job_id']=subprocess.check_output(['sbatch','--parsable',str(path)],text=True).strip()
    write(receipt.name,data)


def stage():
    if not (HERE/'REPRODUCTION_SUBMISSION.json').exists():return 'waiting_reproduction_resources'
    checks=list((HERE/'reproduction/checks').glob('*.json'))
    if len(checks)!=32:return 'waiting_32_reproduction_checks'
    if not (HERE/'REPRODUCTION_GATE.json').exists():
        subprocess.run([PYTHON,'-B',str(HERE/'reproduction.py'),'collect'],check=True)
    gate=read('REPRODUCTION_GATE.json')
    if not any(gate['type_passed'][k] for k in ('deterministic','mixture','diffusion','G')):
        write('TRAINING_BLOCKED.json',dict(utc=utc(),reason='No TT historical type passed reproduction'))
        return 'blocked_all_TT_reproduction'
    if not (HERE/'ZOO_UNIT_TESTS.json').exists():
        if failure('unit_gpu'):
            write('TRAINING_BLOCKED.json',dict(utc=utc(),reason='GPU unit-test job failed; inspect retained Slurm log',states=failure('unit_gpu')))
            return 'blocked_gpu_unit_job'
        submit('unit_gpu',[f'{PYTHON} -B {HERE}/test_selector.py',f'{PYTHON} -B {HERE}/test_frames.py',
                           f'{PYTHON} -B {HERE}/frozen_expanded.py',f'{PYTHON} -B {HERE}/test_zoo.py'],gpu=True)
        return 'waiting_gpu_unit_tests'
    if not (HERE/'TRAINING_ENV_FLOAT32.json').exists():
        if failure('training_env_equivalence'):
            write('TRAINING_BLOCKED.json',dict(utc=utc(),reason='Equivalence test process failed; inspect retained Slurm log',states=failure('training_env_equivalence')))
            return 'blocked_training_env_test_process'
        submit('training_env_equivalence',[f'{PYTHON} -B {HERE}/test_training_env.py float64',
                                          f'{PYTHON} -B {HERE}/test_training_env.py float32'])
        return 'waiting_training_environment_equivalence'
    if not read('ZOO_UNIT_TESTS.json')['passed'] or not read('FRAME_TESTS_NEW.json')['passed']:
        write('TRAINING_BLOCKED.json',dict(utc=utc(),reason='New policy unit or coordinate test failed'))
        return 'blocked_unit_tests'
    if not (HERE/'TRAINING_ENV_GATE.json').exists():
        if failure('training_observation'):
            return 'blocked_training_observation_job'
        submit('training_observation',[f'{PYTHON} -B {HERE}/test_training_observation.py'])
        return 'waiting_same_state_observation_check'
    if not (HERE/'TRAINING_CONFIGS.json').exists():
        (HERE/'TRAINING_CONFIGS.json').write_bytes((HERE/'TRAINING_CONFIGS_DRAFT.json').read_bytes())
        if read('TRAINING_ENV_GATE.json')['passed']:
            (HERE/'IPPO_TRAINING_CONFIGS.json').write_bytes((HERE/'IPPO_TRAINING_CONFIGS_DRAFT.json').read_bytes())
        else:
            write('IPPO_IMPLEMENTATION_GAP.json',dict(utc=utc(),reason='Ring training environment equivalence gate failed; PPO not trained'))
        paths=list(HERE.glob('*TRAINING_CONFIGS.json'))
        write('TRAINING_CONFIGS_FROZEN.receipt.json',dict(utc=utc(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},before_training=True))
    if not (HERE/'TRAINING_WORKER_0_COMPLETE.json').exists() or not (HERE/'TRAINING_WORKER_1_COMPLETE.json').exists():
        failed=failure('training')
        if failed:
            write('TRAINING_BLOCKED.json',dict(utc=utc(),reason='Training worker job failed; inspect retained attempts and Slurm log',states=failed))
            return 'blocked_training_worker_job'
        submit('training',[f'{PYTHON} -B {HERE}/train_batch.py "$SLURM_ARRAY_TASK_ID"'],gpu=True,array=True)
        return 'waiting_new_policy_training'
    if (HERE/'IPPO_PARALLEL_RESERVATION.json').exists():
        if not all((HERE/f'IPPO_EXTRA_WORKER_{i}_COMPLETE.json').exists() for i in range(3)):
            if failure('ippo_extra'):
                return 'blocked_extra_ippo_training_job'
            return 'waiting_extra_ippo_training'
    if (HERE/'BC_PARALLEL_RESERVATION.json').exists():
        if not all((HERE/f'BC_EXTRA_WORKER_{i}_COMPLETE.json').exists() for i in range(3)):
            if failure('bc_extra'):
                return 'blocked_extra_bc_training_job'
            return 'waiting_extra_bc_training'
    write('TRAINING_STAGE_READY.json',dict(utc=utc(),status='Training phase ended; review attempts and select PPO checkpoints before capability gates',
          locked_arm_labels_opened=False,frozen_selector_exists=(HERE/'FROZEN_PIPELINE.json').exists()))
    return 'training_stage_ready'


def main():
    lock=(HERE/'ADVANCE_TRAINING.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write('ADVANCE_TRAINING_PID.json',dict(pid=os.getpid(),utc=utc()))
    while True:
        try:
            status=stage()
        except RuntimeError as exc:
            status='waiting_resources: '+str(exc)
        except Exception as exc:
            import traceback
            write('SUPERVISOR_ERROR.json',dict(utc=utc(),error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            status='blocked_supervisor_error: '+str(exc)
        write('LIVE_STATUS.json',dict(utc=utc(),stage=status))
        if status.startswith('blocked_') or status=='training_stage_ready':return
        time.sleep(30)

if __name__=='__main__':main()
