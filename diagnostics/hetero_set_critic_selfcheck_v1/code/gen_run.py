"""Roll out frozen generator proposals on validation fresh states 8-23 (isolated harness, accelerated)."""
import sys,os,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run as R
_W={}
def _init():
 R._init('scenes_set.json');_W.update(R._W)
def _run(t):
 v,k,eta,ci,si=t;sc=_W['scenes'][v];act=dict(key=f'gen{k:02d}',eta_a=list(eta),eta_b=list(eta),theta=None)
 c=_W['make_new'](sc,_W['controllers'][ci],sc['states'][si],act,0);r,_=_W['execute'](c)
 return dict(scene=v,proposal=k,controller=ci,state=si,success=int(r['success']),outcome=r['outcome'])
if __name__=='__main__':
 G=json.load(open('GEN_FROZEN.json'));done=set()
 if os.path.exists('gen.jsonl'):
  for l in open('gen.jsonl'):d=json.loads(l);done.add((d['scene'],d['proposal'],d['controller'],d['state']))
 tasks=[(v,k,e,ci,si) for v,g in G.items() for k,e in enumerate(g['proposals']) for ci in range(3) for si in range(8,24) if (v,k,ci,si) not in done]
 tasks.sort(key=lambda t:0 if t[0].startswith('hs_obst') else 1);print(len(tasks),'todo',flush=True);t0=time.time();n=0
 with ProcessPoolExecutor(int(os.environ.get("HS_WORKERS","16")),mp_context=mp.get_context('spawn'),initializer=_init) as pool,open('gen.jsonl','a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in tasks]):
   f.write(json.dumps(fu.result())+'\n');f.flush();n+=1
   if n%1000==0:print(n,round(time.time()-t0),flush=True)
 print('done',n,round(time.time()-t0),flush=True)
