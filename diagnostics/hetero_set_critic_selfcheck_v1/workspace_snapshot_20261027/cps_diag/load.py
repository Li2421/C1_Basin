"""Read-only loader for critic_pool_supplement_v1 data (no writes outside this dir)."""
import json,struct,numpy as np
from functools import lru_cache
R='/home/zhihan/research/Basin_C1/diagnostics/'
D=R+'critic_pool_supplement_v1/'
CENTER=np.array([1.,0.,.75]);RADIUS=np.array([1.,1.5,.75])
def rd(p):return json.load(open(p))
def etahex(e):return ':'.join(struct.pack('>d',float(v)).hex() for v in e)
POOL={r['name']:r for r in rd(D+'TRAINING_POOL.json')}
VIN=rd(D+'VALIDATION_INPUTS.json')
MENU=np.array(VIN['etas'],float);KEYS=VIN['eta_keys']
LOOK={etahex(e):i for i,e in enumerate(MENU)}
VAL=list(VIN['scene_inputs'])
@lru_cache(None)
def rows(n):
 z=np.load(POOL[n]['rows_path']);return {k:z[k] for k in z.files}
@lru_cache(None)
def curve(n,ctl=None):
 """49-menu rate per candidate (scene, optionally per controller); nan when coverage<.5"""
 d=rows(n);S=POOL[n]['state_count'];s=np.zeros(49);c=np.zeros(49);seen=[set() for _ in range(49)]
 for i,e in enumerate(d['eta']):
  a=LOOK.get(etahex(e))
  if a is None or (ctl is not None and d['ctl'][i]!=ctl):continue
  s[a]+=d['y'][i];c[a]+=1;seen[a].add((int(d['state'][i]),int(d['ctl'][i])))
 cov=np.array([len(v) for v in seen])/(S*(3 if ctl is None else 1))
 q=np.divide(s,c,out=np.full(49,np.nan),where=c>0);q[cov<.5]=np.nan;return q
@lru_cache(None)
def cell_table(n):
 """(3,S,49) success with nan where missing"""
 d=rows(n);S=POOL[n]['state_count'];t=np.full((3,S,49),np.nan)
 for i,e in enumerate(d['eta']):
  a=LOOK.get(etahex(e))
  if a is not None:t[d['ctl'][i],d['state'][i],a]=d['y'][i]
 return t
LT=rd(R+'hetero_scene_pipeline_v1/LABEL_TABLES.json')
def val_y(n,eval_only=True):
 r=LT[n];y=np.asarray(r['y'],float)
 if eval_only:
  pos=[r['state_indices'].index(i) for i in VIN['evaluation'][n]];y=y[:,pos]
 return y
def cprime(n):return np.asarray(VIN['scene_inputs'][n]['c_prime'] if n in VIN['scene_inputs'] else POOL[n]['c_prime'],float)
def vh(n):return np.asarray(VIN['scene_inputs'][n]['h'],float)
