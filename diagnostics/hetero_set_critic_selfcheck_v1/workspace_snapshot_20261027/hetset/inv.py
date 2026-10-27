"""Inventory: every scene with full-menu labels -> (3,S,49) table."""
import sys,json,numpy as np
sys.path.insert(0,'/home/zhihan/.claude/jobs/8061f3d8/tmp/cps_diag')
from load import *
KEYS_=KEYS
SS='/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/'
def tables():
 T={}
 for v in VAL:T[v]=('hetero',val_y(v,False))
 for n in POOL:
  t=cell_table(n)
  if np.isfinite(t).mean((0,1)).min()>=.9 or (np.isfinite(t).all(-1).mean()>.9):T[n]=('pool:'+POOL[n]['group'],t)
 for f in ['sweep1.jsonl','sweep2.jsonl','sweep3.jsonl']:
  acc={}
  for l in open(SS+f):
   d=json.loads(l)
   if d['solo'] is not None:continue
   acc.setdefault(d['scene'],[]).append(d)
  for s,rs in acc.items():
   S=max(r['state'] for r in rs)+1;t=np.full((3,S,49),np.nan)
   for r in rs:t[r['controller'],r['state'],KEYS_.index(r['action'])]=r['success']
   if np.isfinite(t).all():T['explore:'+s]=('explore',t)
 return T
if __name__=='__main__':
 T=tables();i29=KEYS.index('sE29')
 for n,(src,t) in sorted(T.items(),key=lambda x:x[1][0]):
  q=np.nanmean(t,(0,1));S=t.shape[1]
  # leave-one-state-out own best
  loso=[]
  for s in range(S):
   o=np.nanmean(np.delete(t,s,1),(0,1));loso.append(np.nanmean(t[:,s,int(np.nanargmax(o))]))
  top=np.argsort(-np.nan_to_num(q,nan=-1))[:3]
  print(f"{n[:46]:46s} {src:9s} S={S:2d} mean {np.nanmean(q):.2f} best {KEYS[top[0]]}:{q[top[0]]:.2f} 2nd {KEYS[top[1]]}:{q[top[1]]:.2f} 3rd {KEYS[top[2]]}:{q[top[2]]:.2f} | sE29 {q[i29]:.2f} | LOSO-own {np.mean(loso):.2f}")
