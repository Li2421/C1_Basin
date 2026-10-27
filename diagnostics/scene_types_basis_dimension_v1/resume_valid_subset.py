import pathlib,time,json,os,signal,subprocess
p=pathlib.Path(__file__).resolve().parent
while True:
 q=p/'impc_v2_development_other15_execution.json'
 if q.exists() and json.loads(q.read_text()).get('complete'):break
 time.sleep(10)
old=int((p/'main.pid').read_text())
try:
 command=pathlib.Path(f'/proc/{old}/cmdline').read_bytes()
 if b'scene_types_basis_dimension_v1/main.py' in command:os.kill(old,signal.SIGKILL)
except FileNotFoundError:pass
log=(p/'main.log').open('ab')
child=subprocess.Popen(['/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python',str(p/'main.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ|{'PYTHONDONTWRITEBYTECODE':'1'})
(p/'main.pid').write_text(str(child.pid)+'\n')
print('Resumed valid15 coordinator',child.pid,flush=True)
