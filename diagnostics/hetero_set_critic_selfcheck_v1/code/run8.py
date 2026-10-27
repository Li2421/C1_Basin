"""Five more controller types (index 3-7, one representative each) on the 12 test scenes, states 8-23.
Pipeline choice stays frozen (FINAL_FROZEN.json, 16 proposals, chosen without looking at controller)."""
import os,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run as R,nominal as NM
H='/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness'
NEW=['centralized_mppi0','cross_entropy0','distributed_cross_entropy0','conflict_based_search0','prioritized_astar0']
_W={}
def _init():
 R._init('scenes_all.json');M={m['controller']:m for m in json.load(open(H+'/CONTROLLER_MANIFEST.json'))}
 _W.update(R._W);_W['controllers']=list(_W['controllers'])+[M[c] for c in NEW]
def _run(t):
 v,ci,si,key,eta=t;sc=_W['scenes'][v]
 if key=='nominal':
  c=_W['make_new'](sc,_W['controllers'][ci],sc['states'][si],_W['actions'][R.KEYS.index('safety')],0);r=NM.run_nominal(c)
 else:
  act=_W['actions'][R.KEYS.index(key[5:])] if key.startswith('menu:') else dict(key='gen'+key,eta_a=list(eta),eta_b=list(eta),theta=None)
  c=_W['make_new'](sc,_W['controllers'][ci],sc['states'][si],act,0);r,_=_W['execute'](c)
 return dict(scene=v,key=key,controller=ci,controller_id=_W['controllers'][ci]['controller'],state=si,success=int(r['success']),outcome=r['outcome'])
if __name__=='__main__':
 Z=json.load(open('FINAL_FROZEN.json'));TE=Z['test'];out='test8.jsonl'
 tasks=[]
 for v in TE:
  k=Z['choices'][v]['ens16'];e=Z['choices'][v]['proposals'][k]
  for ci in range(3,8):
   for si in range(8,24):
    tasks+=[(v,ci,si,f'p{k}',e),(v,ci,si,'menu:sE22',None),(v,ci,si,'menu:eta13',None),(v,ci,si,'menu:safety',None),(v,ci,si,'nominal',None)]
 done=set()
 if os.path.exists(out):
  for l in open(out):d=json.loads(l);done.add((d['scene'],d['controller'],d['state'],d['key']))
 todo=[t for t in tasks if (t[0],t[1],t[2],t[3]) not in done];todo.sort(key=lambda t:0 if t[0].startswith('hs_obst') else 1)
 print(len(todo),'todo',flush=True);t0=time.time();n=0
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=_init) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):
   f.write(json.dumps(fu.result())+'\n');f.flush();n+=1
   if n%500==0:print(n,round(time.time()-t0),flush=True)
 print('done',n,round(time.time()-t0),flush=True)
