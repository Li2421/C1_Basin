"""Unified isolated runner. Task = (out_file, scene, controller, state, key, eta). key 'menu:<KEY>' or 'p<k>'."""
import os,sys,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run as R
_W={}
def _init():
 R._init('scenes_all.json');_W.update(R._W)
def _run(t):
 out,v,ci,si,key,eta=t;sc=_W['scenes'][v]
 if key.startswith('menu:'):act=_W['actions'][R.KEYS.index(key[5:])]
 else:act=dict(key='gen'+key,eta_a=list(eta),eta_b=list(eta),theta=None)
 c=_W['make_new'](sc,_W['controllers'][ci],sc['states'][si],act,0);r,_=_W['execute'](c)
 return out,dict(scene=v,key=key,controller=ci,state=si,success=int(r['success']),outcome=r['outcome'])
def go(tasks,workers):
 done={}
 for f in {t[0] for t in tasks}:
  done[f]=set()
  if os.path.exists(f):
   for l in open(f):d=json.loads(l);done[f].add((d['scene'],d['key'],d['controller'],d['state']))
 todo=[t for t in tasks if (t[1],t[4],t[2],t[3]) not in done[t[0]]];todo.sort(key=lambda t:0 if t[1].startswith('hs_obst') else 1)
 print(len(todo),'todo',flush=True);fh={f:open(f,'a') for f in done};t0=time.time();n=0
 with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),initializer=_init) as pool:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):
   f,d=fu.result();fh[f].write(json.dumps(d)+'\n');fh[f].flush();n+=1
   if n%5000==0:print(n,round(time.time()-t0),flush=True)
 print('done',n,round(time.time()-t0),flush=True)
