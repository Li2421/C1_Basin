"""Roll out generator proposals on TRAINING scenes, states 0-15 (labels for critic training)."""
import os,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import gen_run as GR
if __name__=='__main__':
 P=json.load(open('GEN_TRAIN_PROPOSALS.json'));done=set()
 if os.path.exists('gen_train.jsonl'):
  for l in open('gen_train.jsonl'):d=json.loads(l);done.add((d['scene'],d['proposal'],d['controller'],d['state']))
 tasks=[(v,k,e,ci,si) for v,p in P.items() for k,e in enumerate(p) for ci in range(3) for si in range(16) if (v,k,ci,si) not in done]
 tasks.sort(key=lambda t:0 if t[0].startswith('hs_obst') else 1);print(len(tasks),'todo',flush=True);t0=time.time();n=0
 with ProcessPoolExecutor(int(os.environ.get('HS_WORKERS','16')),mp_context=mp.get_context('spawn'),initializer=GR._init) as pool,open('gen_train.jsonl','a') as f:
  for fu in as_completed([pool.submit(GR._run,t) for t in tasks]):
   f.write(json.dumps(fu.result())+'\n');f.flush();n+=1
   if n%2000==0:print(n,round(time.time()-t0),flush=True)
 print('done',n,round(time.time()-t0),flush=True)
