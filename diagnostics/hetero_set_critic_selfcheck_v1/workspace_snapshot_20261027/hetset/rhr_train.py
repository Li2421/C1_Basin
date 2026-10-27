"""Right-hand rule (same fixed rule as rhr.py) on the 28 training scenes, 3 original controllers, states 8-15."""
import os,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run8,rhr
if __name__=='__main__':
 S=[s['name'] for s in json.load(open('scenes_all.json'))];Z=json.load(open('FINAL_FROZEN.json'))
 TR=[s for s in S if s not in Z['val'] and not s.endswith('_te')];out='train_rhr.jsonl'
 tasks=[(v,ci,si,rhr.RULE['speed']) for v in TR for ci in range(3) for si in range(8,16)]
 done=set()
 if os.path.exists(out):done={(d['scene'],d['controller'],d['state']) for d in map(json.loads,open(out))}
 todo=[t for t in tasks if t[:3] not in done];print(len(todo),'todo',flush=True);t0=time.time()
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=run8._init) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(rhr._run,t) for t in todo]):f.write(json.dumps(fu.result())+'\n');f.flush()
 print('done',round(time.time()-t0),flush=True)
