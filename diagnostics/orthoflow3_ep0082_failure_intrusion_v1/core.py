#!/usr/bin/env python3
import csv,json,hashlib,sys,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;G=D/'orthoflow3_general_basin_geometry_v1'
SID='T0_WIDE_perm00_ep0082';SHA='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
AFF=np.array([.875,0,.375]);SCALE=np.array([.75,1.,.75])
HS=np.unique(np.array(json.load(open(D/'orthoflow3_t0_multiball_basin_learning_v1/geometry_constants.json'))['halfspaces']),axis=0)
def read(p):return list(csv.DictReader(open(p)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(eta):return np.asarray(eta,dtype='<f8').tobytes().hex()
def eta(r):return np.array([float(r[f'eta{i}']) for i in (1,2,3)])
def norm(r):return (eta(r)-AFF)/SCALE
def inside(z):return np.all(np.atleast_2d(z)@HS[:,:3].T+HS[:,3]<=1e-10,axis=1)
def write(name,rows,fields=None):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def frame():
 r=json.load(open(H/'frozen_frame.json'));return np.array(r['center']),np.array(r['R'])
def provenance_categories(r):
 s=(r.get('phases','')+';'+r.get('sources','')).lower();out=[]
 for cat,words in [('Sobol',['coverage','outward','common_stage','initial_q64']),('interpolation',['interpol','star','detour']),('boundary_probe',['ray','bisect','bis1','bis2','bis3','limit','boundary']),('ball_validation',['inside','union','ball']),('analytic_family_validation',['retained_validation','bounded_validation','polyhedral_validation','minimal_validation']),('cross_transfer',['cross_transfer','common_core','val_neighbor'])]:
  if any(w in s for w in words):out.append(cat)
 return out or ['other']
def plan(name,requests,question):
 assert not (H/f'plans/{name}').exists(),'Frozen plan already exists'
 existing={r['eta_key']:r for r in read(H/'exact_q64_ep0082.csv')};unique={}
 for r in requests:
  e=np.asarray(r['eta'],float);assert inside((e-AFF)/SCALE)[0];k=key(e)
  if k not in unique:unique[k]=dict(state_id=SID,eta=e.tolist(),eta_key=k,phase=name,roles=[])
  unique[k]['roles'].append(r['role'])
 rows=[];new=[]
 for r in unique.values():
  r['cached_Q64']=r['eta_key'] in existing;rows.append(dict(r,eta=json.dumps(r['eta']),roles=json.dumps(r['roles'])))
  if not r['cached_Q64']:new.append(r)
 write(f'rounds/{name}/manifest.csv',rows,['state_id','eta','eta_key','phase','roles','cached_Q64'])
 for sh in range(2):
  p=H/f'plans/{name}/shard{sh}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('w') as f:
   for j,r in enumerate(new):
    if j%2==sh:
     for seed in range(64):f.write(json.dumps(dict(r,future_index=seed))+'\n')
 cost=dict(question=question,new_exact_Q64=len(new),cached_Q64=len(rows)-len(new),maximum_new_continuations=64*len(new),GPU_shards=2,CPU_threads=4,estimated_wall_seconds=120+len(new)*42/2,reference='Prior64seed-query cost roughly42worker-seconds;2shards, same frozen wrapper. Actual >3x estimate requires diagnosis.',created_local=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
 dump(f'rounds/{name}/cost_estimate.json',cost)
 allrows=[]
 for p in sorted(H.glob('rounds/*/manifest.csv')):allrows+=read(p)
 write('adaptive_probe_manifest.csv',allrows,['state_id','eta','eta_key','phase','roles','cached_Q64'])
 ledger=read(H/'experiment_ledger.csv') if (H/'experiment_ledger.csv').exists() else []
 ledger.append(dict(experiment=name,question=question,new_eta=len(new),cached_eta=len(rows)-len(new),planned_continuations=64*len(new),status='FROZEN_NOT_SUBMITTED',job_id='',result=''))
 write('experiment_ledger.csv',ledger)
 return cost
