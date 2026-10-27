"""Validation scenes (frozen 8): five more controllers (pipeline 16-proposal choice, sE22, safety) + right-hand rule for all 8; states 8-23."""
import os,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run8,rhr
def _run(t):
 if t[3]=='rhr':return rhr._run((t[0],t[1],t[2],rhr.RULE['speed']))
 return run8._run(t)
if __name__=='__main__':
 Z=json.load(open('FINAL_FROZEN.json'));V=Z['val'];out='val8.jsonl';tasks=[]
 for v in V:
  k=Z['choices'][v]['ens16'];e=Z['choices'][v]['proposals'][k]
  for si in range(8,24):
   for ci in range(3,8):tasks+=[(v,ci,si,f'p{k}',e),(v,ci,si,'menu:sE22',None),(v,ci,si,'menu:safety',None)]
   for ci in range(8):tasks.append((v,ci,si,'rhr',None))
 done=set()
 if os.path.exists(out):done={(d['scene'],d['controller'],d['state'],d['key']) for d in map(json.loads,open(out))}
 todo=[t for t in tasks if t[:4] not in done];print(len(todo),'todo',flush=True);t0=time.time()
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=run8._init) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):f.write(json.dumps(fu.result())+'\n');f.flush()
 print('done',round(time.time()-t0),flush=True)
