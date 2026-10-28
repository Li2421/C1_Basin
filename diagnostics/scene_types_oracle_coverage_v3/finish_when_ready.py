"""Finalize after the one numerical coordinator exits; no simulation or solver use."""
from pathlib import Path
import time,subprocess,os,json,datetime

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'

def main():
    while True:
        done=HERE/'COMPLETE.json'
        jobs=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%j|%T|%C'],text=True)
        active=[line for line in jobs.splitlines()if line.startswith('stb3_')]
        if done.exists()and not active:break
        time.sleep(30)
    if not(HERE/'COVERAGE.json').exists():
        print('No coverage artifact exists; numerical pipeline gap remains explicit.',flush=True);return
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([PYTHON,str(HERE/'finalize.py')],cwd=ROOT,env=env,check=True)
    print('Delivery analysis finished',datetime.datetime.now().astimezone().isoformat(),flush=True)

if __name__=='__main__':main()
