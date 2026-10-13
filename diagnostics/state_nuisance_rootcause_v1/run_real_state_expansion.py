"""Resumable Slurm coordinator for controller-input-free Q data and evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=HERE/'real_state_expansion'
PY=Path('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python')
PLOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python')
sys.path.insert(0,str(REPO))
from diagnostics.state_nuisance_rootcause_v1.real_state_expansion import read,write


def run(command,name,env=None):
    print('START',name,flush=True)
    write(OUT/'batch_status.json',dict(phase=name,updated=time.time(),research_complete=(OUT/'research_results.json').exists(),batch_complete=False,video_required=False))
    with (OUT/'logs'/f'{name}.log').open('a') as f:
        result=subprocess.run([str(c) for c in command],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        write(OUT/'batch_status.json',dict(phase=name,exit_code=result.returncode,updated=time.time(),
              research_complete=(OUT/'research_results.json').exists(),batch_complete=False,video_required=False,log=str(OUT/'logs'/f'{name}.log')))
        raise RuntimeError(f'{name} exited {result.returncode}; see its log')
    print('DONE',name,flush=True)


def submit(stage):
    n=len(read(OUT/f'missing_{stage}.json'));chunks=math.ceil(n/192)
    cmd=['sbatch','--parsable','-p','cpu',f'--array=0-{chunks-1}%22','--cpus-per-task=1','--mem=3G','--time=00:45:00',
         f'--job-name=ring_q_{stage}',f'--output={OUT}/logs/{stage}_%A_%a.log',
         str(HERE/'real_state_expansion_worker.sh'),stage,str(chunks)]
    job=subprocess.check_output(cmd,cwd=REPO,text=True).strip().split(';')[0]
    write(OUT/f'{stage}_submission.json',dict(job_id=int(job),array_tasks=chunks,max_concurrent=22,chunk_size=192,submitted=time.time(),stage=stage))
    return job


def wait_job(job,stage):
    write(OUT/'batch_status.json',dict(phase=f'{stage}_rollouts',job_id=str(job),updated=time.time(),research_complete=False,batch_complete=False,video_required=False))
    while True:
        r=subprocess.run(['squeue','-h','-j',str(job)],capture_output=True,text=True)
        if not r.stdout.strip():break
        time.sleep(15)
    accounting=subprocess.run(['sacct','-j',str(job),'-X','-n','-P','--format=JobID,State,ExitCode'],capture_output=True,text=True)
    # This cluster disables accounting storage. Complete per-shard receipts and
    # the following exhaustive case-by-case collect are the authoritative gate.
    count=len(read(OUT/f'missing_{stage}.json'));chunks=math.ceil(count/192)
    receipts=[]
    for i in range(chunks):
        r=read(OUT/'progress'/f'{stage}_{i}.json');expected=min(192,count-i*192)
        assert r['complete'] and not r['limited_pilot'] and r['workers']==chunks
        assert r['stage']==stage and r['worker']==i and r['total']==r['completed']==expected
        receipts.append(r)
    write(OUT/f'{stage}_slurm_accounting.json',dict(job_id=str(job),stdout=accounting.stdout,stderr=accounting.stderr,
        returncode=accounting.returncode,completed_shards=len(receipts),complete_case_count=sum(r['completed'] for r in receipts)))


def gpu(action):
    env=os.environ.copy();env.update(JAX_PLATFORMS='cuda',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    run(['srun','-p','gpu','--gres=shard:1','--cpus-per-task=2','--mem=8G','--time=01:00:00',f'--job-name=ring_q_{action}',
         PY,HERE/'real_state_learning.py',action],action,env)


def main(source_job=None):
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    if not (OUT/'protocol.json').exists():run([PY,HERE/'real_state_expansion.py','prepare'],'prepare')
    if not (OUT/'physical_inputs.npz').exists():run([PY,HERE/'real_state_learning.py','features'],'features')
    if not (OUT/'source_dataset.npz').exists():
        job=source_job or (read(OUT/'source_submission.json')['job_id'] if (OUT/'source_submission.json').exists() else submit('source'))
        wait_job(job,'source')
        run([PY,HERE/'real_state_expansion.py','merge'],'merge_source')
        run([PY,HERE/'real_state_expansion.py','collect','--stage','source'],'collect_source')
    if not (OUT/'models_frozen.json').exists():gpu('fit')
    if not (OUT/'test_dataset.npz').exists():
        job=read(OUT/'test_submission.json')['job_id'] if (OUT/'test_submission.json').exists() else submit('test')
        wait_job(job,'test')
        run([PY,HERE/'real_state_expansion.py','merge'],'merge_test')
        run([PY,HERE/'real_state_expansion.py','collect','--stage','test'],'collect_test')
    if not (OUT/'evaluation_audit.json').exists():gpu('evaluate')
    if not (OUT/'research_results.json').exists():run([PY,HERE/'real_state_learning.py','analyze'],'analyze')
    run([PLOT,HERE/'real_state_report.py'],'report')
    write(OUT/'batch_status.json',dict(phase='complete',research_complete=True,batch_complete=True,
        video_required=False,updated=time.time()))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source-job');q=a.parse_args()
    main(q.source_job)
