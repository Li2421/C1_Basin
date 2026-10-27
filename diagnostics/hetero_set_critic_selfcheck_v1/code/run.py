"""Isolated rollout runner (no shared-DB writes). tasks: (scene, ctl, state, action_index)."""
import sys,os,json,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
H='/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness'
KEYS=['safety']+[f'eta{i}' for i in range(16)]+[f'sE{i:02d}' for i in range(32)]
_W={}
def _init(scenefile):
 os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
 os.environ['OMP_NUM_THREADS']='1';sys.path.insert(0,H)
 import bootstrap  # noqa
 from common import actions
 from design import controllers
 from scenes import make_new
 from execution import execute
 if os.environ.get('HS_FAST')=='1':
  sys.path.insert(0,'/home/zhihan/.claude/jobs/8061f3d8/tmp/hetset');import accel;accel.enable()
 _W.update(actions=actions(),controllers=controllers(),make_new=make_new,execute=execute,scenes={s['name']:s for s in json.load(open(scenefile))})
def _run(t):
 name,ci,si,ai=t;sc=_W['scenes'][name];st=sc['states'][si]
 c=_W['make_new'](sc,_W['controllers'][ci],st,_W['actions'][ai],0);t0=time.time();r,_=_W['execute'](c)
 return dict(scene=name,controller=ci,state=si,action=KEYS[ai],success=int(r['success']),outcome=r['outcome'],length=int(r['episode_length']),seconds=round(time.time()-t0,3))
def run(tasks,scenefile,out,workers=16):
 done=set()
 if os.path.exists(out):
  for l in open(out):d=json.loads(l);done.add((d['scene'],d['controller'],d['state'],d['action']))
 todo=[t for t in tasks if (t[0],t[1],t[2],KEYS[t[3]]) not in done]
 # longest-first: circle scenes first
 pass
 print(len(todo),'todo',len(tasks)-len(todo),'done',flush=True);t0=time.time();n=0
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_init,initargs=(scenefile,)) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):
   f.write(json.dumps(fu.result())+'\n');f.flush();n+=1
   if n%1000==0:print(n,round(time.time()-t0),'s',flush=True)
 print('done',n,round(time.time()-t0),'s',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('scenefile');a.add_argument('out');a.add_argument('--states',default='0-7');a.add_argument('--scenes',default='');a.add_argument('--actions',default='all');a.add_argument('--workers',type=int,default=16);a=a.parse_args()
 S=json.load(open(a.scenefile));lo,hi=map(int,a.states.split('-'))
 acts=range(49) if a.actions=='all' else [KEYS.index(k) for k in a.actions.split(',')]
 tasks=[(s['name'],ci,si,ai) for s in S if not a.scenes or s['name'] in a.scenes.split(',') for ci in range(3) for si in range(lo,hi+1) for ai in acts]
 run(tasks,a.scenefile,a.out,a.workers)
