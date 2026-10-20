"""Wait without consuming a shard, then submit one guarded CPU task."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def queue():
    return subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%P|%j|%T|%C|%b'],text=True)
def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2)+'\n')
while True:
    q=queue(); running=[line.split('|') for line in q.splitlines() if line.split('|')[3]=='RUNNING']
    used=sum(int(v[4]) for v in running if v[1]=='cpu')
    save('RESOURCE_WAIT.json',dict(utc=datetime.now(timezone.utc).isoformat(),queue=q,CPU_shards_running=used,required=1))
    if used<18:break
    time.sleep(5)
cmd=' '.join(['env','PYTHONDONTWRITEBYTECODE=1','JAX_PLATFORMS=cpu','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',shlex.quote(PY),'-B',shlex.quote(str(HERE/'reproduction.py')),'execute'])
args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=4G','--time=00:40:00','-J','r39_reproduction','-o',str(HERE/'reproduction_%j.log'),'--wrap',cmd]
job=subprocess.check_output(args,text=True).strip()
save('REPRODUCTION_SUBMISSION.json',dict(utc=datetime.now(timezone.utc).isoformat(),job=job,args=args,pre_submission_queue=q))
print('Submitted '+job,flush=True)
paused=False
while True:
    q=queue(); rows=[line.split('|') for line in q.splitlines()];own=next((v for v in rows if v[0]==job),None)
    if own is None:break
    external=sum(int(v[4]) for v in rows if v[3]=='RUNNING' and v[1]=='cpu' and v[0]!=job)
    if external>=18 and own[3]=='RUNNING':
        subprocess.run(['scontrol','suspend',job],check=True);paused=True
    elif external<18 and own[3]=='SUSPENDED' and paused:
        subprocess.run(['scontrol','resume',job],check=True);paused=False
    save('RESOURCE_STATUS.json',dict(utc=datetime.now(timezone.utc).isoformat(),queue=q,external_CPU_shards=external,own_job=job,paused=paused))
    time.sleep(3)
print('Job left queue; inspect gate files',flush=True)
