#!/usr/bin/env python3
"""Freeze/execute/aggregate fresh VAL inner-set exact-Q64 validation."""
from __future__ import annotations
import argparse,csv,importlib.util,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import qmc
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1';SHARED=D/'orthoflow3_shared_conservative_family_v1'
sys.path.insert(0,str(SHARED));from shared_families import contains,domain_contains
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75])
def read(p):return list(csv.DictReader(open(p)))
def write(n,rows,fields=None):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(n,x):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def key(z):return np.asarray(z,dtype='<f8').tobytes().hex()
def sample(m,sid,known):
 inner=json.loads(json.dumps(m));inner['gamma']*=.80; rng=qmc.Sobol(3,scramble=True,seed=int.from_bytes(sid.encode()[:8].ljust(8,b'0'),'little')%(2**32));cand=[]
 for block in range(8):
  u=2*rng.random(128)-1;u=u[np.sum(abs(u)**4,axis=1)<=1]
  z=np.asarray(inner['center'])+(u*np.asarray(inner['axes']))@np.asarray(inner['R']).T
  good=contains(inner,z,True)&domain_contains(z)
  for x in z[good]:
   if key(x) not in known:cand.append(x)
  if len(cand)>=80:break
 if len(cand)<6:raise RuntimeError(('insufficient fresh candidates',sid,len(cand)))
 a=np.asarray(inner['anchor']);selected=[min(cand,key=lambda x:np.linalg.norm(x-a))]
 while len(selected)<6:selected.append(max(cand,key=lambda x:min(np.linalg.norm(x-y) for y in selected)))
 return selected
def prepare():
 models=json.load(open(H/'fitted_models.json'));states={x['state_id']:x for x in json.load(open(POINT/'final_source_split.json'))['states']};evid=read(H/'geometry_evidence_inventory.csv');known={sid:set() for sid in states}
 for r in evid:known[r['state_id']].add(r['eta_key'])
 rows=[];tasks=[]
 for sid,st in sorted(states.items()):
  if st['split']!='val':continue
  pts=sample(models[sid],sid,known[sid])
  for j,z in enumerate(pts):
   role='central' if j<2 else 'separated_interior' if j<4 else 'near_retained_boundary';eta=(AFF+SCALE*z).tolist();r=dict(state_id=sid,split='val',point_index=j,role=role,eta1=eta[0],eta2=eta[1],eta3=eta[2],z1=z[0],z2=z[1],z3=z[2],eta_key=key(z),fresh=True)
   rows.append(r)
   for fi in range(64):tasks.append(dict(state_id=sid,controller='fresh_retained',eta=eta,future_index=fi,phase='fresh_retained_q64',point_index=j,eta_key=key(z)))
 write('fresh_retained_manifest.csv',rows);d=H/'plans/fresh_retained';d.mkdir(parents=True,exist_ok=True)
 for sh in range(2):
  with open(d/f'shard{sh}.jsonl','w') as f:
   for t in tasks:
    if (hash((t['state_id'],t['point_index']))&1)==sh:f.write(json.dumps(t,sort_keys=True)+'\n')
 dump('fresh_retained_cost.json',dict(new_exact_Q64=len(rows),new_continuations=len(tasks),GPU_shards=2,question='Are six frozen inner-set points per VAL state robust under exact Q64?'))
 ledger=read(H/'experiment_ledger.csv');ledger.append(dict(experiment='fresh_retained_validation',new_eta=len(rows),new_continuations=len(tasks),status='FROZEN_NOT_SUBMITTED',result=''));write('experiment_ledger.csv',ledger)
 print(json.dumps({'points':len(rows),'tasks':len(tasks)}))
def run(shard):
 tasks=[json.loads(x) for x in open(H/f'plans/fresh_retained/shard{shard}.jsonl') if x.strip()]
 states={x['state_id']:dict(x,feature_index=x['dataset_index']) for x in json.load(open(POINT/'final_source_split.json'))['states'] if x['state_id'] in {t['state_id'] for t in tasks}}
 features=np.load(POINT/'point_learning_arrays.npz')['features'];spec=importlib.util.spec_from_file_location('point_driver',POINT/'run_eval_shard.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);out=H/f'runs/fresh_retained/shard{shard}';out.mkdir(parents=True,exist_ok=True);m,O=mod.loadmod(out,states);o=O(states,features,np.zeros(3),np.ones(3));start=time.time();res=o.ensure(tasks,'fresh_retained');dest=H/'raw/fresh_retained';dest.mkdir(parents=True,exist_ok=True)
 with open(dest/f'shard{shard}.jsonl','w') as f:
  for r in res:f.write(json.dumps({k:v for k,v in r.items() if k!='_source'},sort_keys=True)+'\n')
 dump(f'raw/fresh_retained/shard{shard}_runtime.json',dict(tasks=len(tasks),new_continuations=o.new,reused_continuations=len(tasks)-o.new,physical_steps=o.steps,wall_seconds=time.time()-start));print(json.dumps({'shard':shard,'tasks':len(tasks),'new':o.new,'steps':o.steps}))
def aggregate():
 man=read(H/'fresh_retained_manifest.csv');raw=defaultdict(list)
 for p in (H/'raw/fresh_retained').glob('shard*.jsonl'):
  for line in open(p):
   r=json.loads(line);raw[(r['state_id'],int(r['point_index']))].append(r)
 rows=[]
 for r in man:
  x=raw[(r['state_id'],int(r['point_index']))];assert len(x)==64 and {int(q['future_index']) for q in x}==set(range(64));s=sum(bool(q['success']) for q in x)
  rows.append(dict(**r,successes=s,trials=64,Q64=s/64,B63=s>=63,deadlock=sum(q['outcome']=='safe_deadlock' for q in x),timeout=sum(q['outcome']=='timeout' for q in x),collision=sum(q['outcome']=='collision' for q in x)))
 write('fresh_retained_validation.csv',rows);bad=sum(not bool(r['B63']) for r in rows);dump('fresh_retained_summary.json',dict(points=len(rows),non_B63=bad,false_inclusion_rate=bad/len(rows),quality='A' if bad==0 else ('B' if bad/len(rows)<=.02 else 'C')));print(json.dumps(json.load(open(H/'fresh_retained_summary.json'))))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','aggregate']);ap.add_argument('--shard',type=int);a=ap.parse_args(); {'prepare':prepare,'run':lambda:run(a.shard),'aggregate':aggregate}[a.stage]()
if __name__=='__main__':main()
