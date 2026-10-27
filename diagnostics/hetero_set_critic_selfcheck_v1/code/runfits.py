import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
jobs=[l for l in open(sys.argv[1])]
def r(j):
 p=subprocess.run([PY,'critic.py',j],capture_output=True,text=True)
 if p.returncode:print('FAIL',p.stderr[-400:],flush=True)
with ThreadPoolExecutor(int(sys.argv[2])) as ex:list(ex.map(r,jobs))
print('done',len(jobs))
