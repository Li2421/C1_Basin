import os,sys,json,numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import run as R,run8
OUT=sys.argv[1]
def job(p):
 W=run8._W;sc=W['scenes'][p['scene']];st=sc['states'][p['state']]
 act=W['actions'][R.KEYS.index(p['key'][5:])] if p['key'].startswith('menu:') else dict(key='gen'+p['key'],eta_a=list(p['eta']),eta_b=list(p['eta']),theta=None)
 c=W['make_new'](sc,W['controllers'][p['controller']],st,act,0);r,a=W['execute'](c)
 name=f"{p['scene']}__{p['controller_type']}__{p['role']}";eta=list(act['eta_a'])
 np.savez_compressed(f"{OUT}/traces/{name}.npz",positions=a['positions'],goals=a['goals'],walls=np.asarray(sc['walls'],float).reshape(-1,2,2),radius=sc['config']['agent_radius'],dt=sc['config']['dt'],
  meta=np.asarray(json.dumps(dict(p,eta=eta,outcome=r['outcome'],success=int(r['success']),steps=int(r['episode_length'])))))
 return name,r['outcome'],p['expected_outcome']
if __name__=='__main__':
 plan=json.load(open('VIDEO_PLAN.json'))['plan']
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=run8._init) as ex:
  res=list(ex.map(job,plan))
 bad=[x for x in res if x[1]!=x[2]];print(len(res),'traces; mismatches',len(bad),bad[:5])
