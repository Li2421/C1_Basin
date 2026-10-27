"""Goal-relative right-hand rule (push to the right of each robot's own goal direction; same trigger/speed) on all 48 scenes:
train 3 controllers states 8-15; val/test 7 capable controllers states 8-23."""
import os,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run as R,run8,rhr
def _run(t):
 v,ci,si=t;W=run8._W;sc=W['scenes'][v]
 c=W['make_new'](sc,W['controllers'][ci],sc['states'][si],W['actions'][R.KEYS.index('safety')],0);r=rhr.run_rhr(c,rhr.RULE['speed'],'goal')
 return dict(scene=v,key='rhr_goal',controller=ci,state=si,**r)
if __name__=='__main__':
 S=[s['name'] for s in json.load(open('scenes_all.json'))];Z=json.load(open('FINAL_FROZEN.json'))
 TR=[s for s in S if s not in Z['val'] and not s.endswith('_te')];out='rhr_goal.jsonl'
 tasks=[(v,ci,si) for v in TR for ci in range(3) for si in range(8,16)]+[(v,ci,si) for v in Z['val']+Z['test'] for ci in [0,1,2,4,5,6,7] for si in range(8,24)]
 done=set()
 if os.path.exists(out):done={(d['scene'],d['controller'],d['state']) for d in map(json.loads,open(out))}
 todo=[t for t in tasks if t not in done];todo.sort(key=lambda t:0 if t[0].startswith('hs_obst') else 1);print(len(todo),'todo',flush=True);t0=time.time()
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=run8._init) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):f.write(json.dumps(fu.result())+'\n');f.flush()
 print('done',round(time.time()-t0),flush=True)
