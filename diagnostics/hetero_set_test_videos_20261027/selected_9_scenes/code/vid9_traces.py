"""Rerun the 18 episodes of VID9_PLAN.json (safety only / pipeline, same start state) and save trajectories."""
import os,sys,json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(HERE);sys.path.insert(0,ROOT);sys.path.insert(0,HERE)
import run as R,run8,lrun
OUT=sys.argv[1]
def job(t):
 p,role=t;W=run8._W;v=p['scene'];sc=W['scenes'][v];st=sc['states'][p['state']]
 act=W['actions'][R.KEYS.index('safety')] if role=='A' else dict(key='gen'+p['pipeline_key'],eta_a=list(p['pipeline_eta']),eta_b=list(p['pipeline_eta']),theta=None)
 c=W['make_new'](sc,W['controllers'][p['controller']],st,act,0);r,a=W['execute'](c)
 np.savez_compressed(f"{OUT}/traces/{p['order']}{role}.npz",positions=a['positions'],goals=a['goals'],walls=np.asarray(sc['walls'],float).reshape(-1,2,2),
   radius=sc['config']['agent_radius'],meta=np.asarray(json.dumps(dict(p,role=role,eta=list(act['eta_a']),outcome=r['outcome'],success=int(r['success'])),ensure_ascii=False)))
 return p['order'],role,r['outcome']
if __name__=='__main__':
 plan=json.load(open(os.path.join(HERE,'VID9_PLAN.json')))
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=lrun._init) as ex:
  res=sorted(ex.map(job,[(p,role) for p in plan for role in 'AB']))
 bad=[x for x in res if (x[1]=='A')==(x[2]=='success')]
 print('episodes',len(res),'unexpected outcomes',bad);print(res)
