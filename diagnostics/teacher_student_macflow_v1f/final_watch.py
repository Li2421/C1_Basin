"""Finish review after coordinator success; never retrain or rerun a trajectory."""
import sys,os,pathlib,json,subprocess,time
H=pathlib.Path(__file__).resolve().parent;R=H.parents[1];P='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python';sys.dont_write_bytecode=True;sys.path.insert(0,str(R))
import shared_rollout_db
import shared_rollout_db.src.cache_writer as writer
assert pathlib.Path(shared_rollout_db.__file__).resolve().is_relative_to(R/'shared_rollout_db')
assert (writer.ROOT/'journals').resolve()==R/'shared_rollout_db/journals'
from quota import census
os.environ.update(PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu')
(H/'FINAL_WATCH_PID').write_text(str(os.getpid()))
while True:
 p=H/'RUN_STATUS.json';s=json.loads(p.read_text()) if p.exists() else {}
 if s.get('status','').startswith('COMPLETE'):break
 if s.get('status')=='INCOMPLETE':
  while census()['running_cpu']>=18:time.sleep(30)
  if not (H/'GAP_DELIVERY.receipt.json').exists():subprocess.run([P,'-B',str(H/'failure_delivery.py')],cwd=R,env=os.environ,check=True)
  raise SystemExit('Coordinator incomplete; evidence gaps reported in GAP_DELIVERY.receipt.json')
 time.sleep(30)
while census()['running_cpu']>=18:time.sleep(30)
if not (H/'FINAL_REVIEW.receipt.json').exists():
 for script,args in [('integrity.py',['END']),('final_review.py',[])]:
  subprocess.run([P,'-B',str(H/script),*args],cwd=R,env=os.environ,check=True)
(H/'FINAL_WATCH_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',receipt='FINAL_REVIEW.receipt.json',no_new_rollouts=True),indent=2))
