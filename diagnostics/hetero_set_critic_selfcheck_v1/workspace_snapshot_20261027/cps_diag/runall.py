import json,subprocess,sys,os
from concurrent.futures import ThreadPoolExecutor
jobs=[json.loads(l) for l in open(sys.argv[1])]
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def run(j):
 r=subprocess.run([PY,'fit.py',json.dumps(j)],capture_output=True,text=True)
 if r.returncode:print('FAIL',j['out'],r.stderr[-500:],flush=True)
with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv)>2 else 14) as ex:list(ex.map(run,jobs))
print('done',len(jobs))
