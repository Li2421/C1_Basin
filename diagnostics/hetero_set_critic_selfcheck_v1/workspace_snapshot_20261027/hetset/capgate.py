"""Capability gate on the 12 test scenes: each robot alone (others removed), safety only, seed 0, states 8-23; pass >= 95%."""
import os,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run as R,run8
def _run(t):
 v,ci,si,j=t;W=run8._W;sc=W['scenes'][v];st=dict(sc['states'][si]);st['N']=1
 for k in ('positions','goals','velocities'):st[k]=[st[k][j]]
 st['index']=f"{si}_solo{j}";c=W['make_new'](sc,W['controllers'][ci],st,W['actions'][R.KEYS.index('safety')],0);r,_=W['execute'](c)
 return dict(scene=v,controller=ci,state=si,robot=j,success=int(r['success']),outcome=r['outcome'])
if __name__=='__main__':
 TE=json.load(open('FINAL_FROZEN.json'))['test'];S={s['name']:s for s in json.load(open('scenes_all.json'))};out='test_capgate.jsonl'
 tasks=[(v,ci,si,j) for v in TE for ci in range(8) for si in range(8,24) for j in range(S[v]['states'][si]['N'])]
 done=set()
 if os.path.exists(out):done={(d['scene'],d['controller'],d['state'],d['robot']) for d in map(json.loads,open(out))}
 todo=[t for t in tasks if t not in done];print(len(todo),'todo',flush=True);t0=time.time()
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=run8._init) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):f.write(json.dumps(fu.result())+'\n');f.flush()
 print('done',round(time.time()-t0),flush=True)
