"""Finalize after the one numerical coordinator exits; no simulation or solver use."""
from pathlib import Path
import time,subprocess,os,json,datetime

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'

def main():
    observed_active=False;quiet=0
    recovery_file=HERE/'COORDINATOR_RECOVERY.json'
    recovery=json.loads(recovery_file.read_text())if recovery_file.exists()else dict(restarts=0,events=[])
    while True:
        done=HERE/'COMPLETE.json'
        jobs=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%j|%T|%C'],text=True)
        active=[line for line in jobs.splitlines()if line.startswith('stb3_')]
        if done.exists()and not active:break
        if active:observed_active=True;quiet=0
        elif observed_active and not done.exists():
            quiet+=1
            if quiet>=2:
                if recovery['restarts']>=3:
                    print('Coordinator infrastructure recovery limit reached; completed raw records retained.',flush=True);return
                recovery['restarts']+=1
                recovery['events'].append(dict(time=datetime.datetime.now().astimezone().isoformat(),reason='No live v3 Slurm coordinator/worker and no completion marker'))
                recovery_file.write_text(json.dumps(recovery,indent=2)+'\n')
                with(HERE/'main.log').open('a')as log:
                    subprocess.Popen(['bash',str(HERE/'resume.sh')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                quiet=0
        time.sleep(30)
    if not(HERE/'COVERAGE.json').exists():
        print('No coverage artifact exists; numerical pipeline gap remains explicit.',flush=True);return
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([PYTHON,str(HERE/'finalize.py')],cwd=ROOT,env=env,check=True)
    print('Delivery analysis finished',datetime.datetime.now().astimezone().isoformat(),flush=True)

if __name__=='__main__':main()
