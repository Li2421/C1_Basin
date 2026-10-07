#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent
SH=D/'orthoflow3_shared_eta_codebook_v1';POINT=D/'orthoflow3_true_t0_point_learning_v1';GEN=D/'orthoflow3_general_basin_geometry_v1';XFER=D/'orthoflow3_t0_eta_continuity_cross_transfer_v1'
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75])
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ekey(e):return np.asarray(e,dtype='<f8').tobytes().hex()
class MLP(nn.Module):
 M:int
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(64)(x));x=nn.silu(nn.Dense(64)(x));return nn.Dense(self.M)(x)
def selected_modes():
 st=json.load(open(SH/'state_split.json'))['states'];a=np.load(SH/'state_features.npz');X=a['x'];cb=read(SH/'codebook_eta.csv');M=len(cb);sel=json.load(open(SH/'selected_model.json'));model=MLP(M);tmp=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));p=serialization.from_bytes(tmp,Path(sel['checkpoint']).read_bytes());log=np.asarray(model.apply(p,jnp.asarray(X)));cal=json.load(open(SH/'calibration.json'));T=cal['temperature'] if cal['used'] else 1.;prob=1/(1+np.exp(-np.clip(log/T,-30,30)));tau=json.load(open(SH/'selected_threshold.json'))['tau'];out=[]
 for q,pr in zip(st,prob):
  m=int(np.argmax(pr));out.append(dict(state_id=q['state_id'],split=q['split'],dataset_index=q['dataset_index'],source_group=q['source_group'],raw_argmax_mode=m,selected_mode=-1 if pr[m]<tau else m,confidence=float(pr[m]),abstained=bool(pr[m]<tau)))
 return st,out
def status(k,n):
 if (n,k) in ((64,63),(64,64),(32,31),(32,32),(16,16),(8,8)):return 'positive'
 if (n==64 and k<=60) or (n==32 and k<=29) or (n==16 and k<=14) or (n==8 and k<=6):return 'negative'
 return 'ambiguous'
def add(E,sid,e,k,n,src):
 if sid not in E or n not in (8,16,32,64):return
 e=np.asarray(e,float);E[sid].append(dict(state_id=sid,eta=e,successes=int(k),trials=int(n),provenance=src))
def collect(states,sel):
 E={q['state_id']:[] for q in states if q['split'] in ('train','val')};cb=read(SH/'codebook_eta.csv');A=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in cb]);
 for split,fn in [('train','train_mode_counts.csv'),('val','val_mode_counts.csv')]:
  for r in read(SH/fn):add(E,r['state_id'],A[int(r['mode_id'])],r['successes'],r['trials'],'aligned_codebook_anchor')
 for r in read(GEN/'exact_q64_inventory.csv'):
  if r['scenario']=='ToyGiveWay_2A':add(E,r['state_id'],[float(r[f'eta{i}']) for i in (1,2,3)],r['successes'],r['trials'],'general_exact_q64')
 for fn in ('cached_cross_transfer.csv','cross_transfer_results.csv'):
  for r in read(XFER/fn):add(E,r['destination_state'],[float(r[f'eta{i}']) for i in (1,2,3)],r['successes'],64,'cross_transfer_q64')
 for r in read(POINT/'local_robustness_probes.csv'):add(E,r['state_id'],[float(r[f'eta{i}']) for i in (1,2,3)],r['successes'],r['trials'],'point_local_q8')
 # Exact-coordinate deduplication keeps the largest trial count, then successes.
 out=[]
 for sid,v in E.items():
  best={}
  for r in v:
   k=ekey(r['eta']);rank=(r['trials'],r['successes'])
   if k not in best or rank>(best[k]['trials'],best[k]['successes']):best[k]=r
  for r in best.values():
   z=(r['eta']-AFF)/SCALE;dist=np.linalg.norm((A-AFF)/SCALE-z,axis=1);m=int(np.argmin(dist))
   if dist[m]<=.20+1e-12:out.append(dict(state_id=sid,assigned_mode=m,normalized_distance_to_anchor=dist[m],eta1=r['eta'][0],eta2=r['eta'][1],eta3=r['eta'][2],successes=r['successes'],trials=r['trials'],label_status=status(r['successes'],r['trials']),provenance=r['provenance']))
 return out,A
def usable(rows,sid,m):
 p=[np.array([float(r[f'eta{i}']) for i in (1,2,3)]) for r in rows if r['state_id']==sid and int(r['assigned_mode'])==m and r['label_status']=='positive']
 if len(p)<2:return False
 z=np.array([(x-AFF)/SCALE for x in p]);return max(np.linalg.norm(z[i]-z[j]) for i in range(len(z)) for j in range(i))>=.05
def offsets_for(anchor_z,Ahs,bhs):
 u=qmc.Sobol(3,scramble=False).random_base2(12)[1:];ans=[]
 for x in u:
  d=.15*(2*x-1)/np.sqrt(3)
  if np.linalg.norm(d)<.05:continue
  z=anchor_z+d
  if np.max(Ahs@z+bhs)<=1e-10 and all(np.linalg.norm(d-q)>.025 for q in ans):ans.append(d)
  if len(ans)==16:return ans
 raise RuntimeError(('not enough feasible offsets',anchor_z,len(ans)))
def main():
 states,sel=selected_modes();smap={r['state_id']:r for r in sel};rows,A=collect(states,sel);write('local_mode_dataset_cached.csv',rows)
 hs=read(POINT/'ebridge_halfspaces.csv');Ah=np.array([[float(r[f'n{i}']) for i in (1,2,3)] for r in hs]);bh=np.array([float(r['b']) for r in hs]);off={};probes=[]
 targets=[r for r in sel if r['split'] in ('train','val') and int(r['selected_mode'])>=0]
 for r in targets:
  sid=r['state_id'];m=int(r['selected_mode'])
  if usable(rows,sid,m):continue
  if m not in off:off[m]=offsets_for((A[m]-AFF)/SCALE,Ah,bh)
  for j,d in enumerate(off[m]):
   e=A[m]+SCALE*d;probes.append(dict(state_id=sid,split=r['split'],mode_id=m,candidate_index=j,eta1=e[0],eta2=e[1],eta3=e[2],offset_z1=d[0],offset_z2=d[1],offset_z3=d[2],normalized_radius=np.linalg.norm(d),trials=8 if r['split']=='train' else 16))
 write('local_probe_manifest.csv',probes)
 tasks=[]
 for r in probes:
  e=[r[f'eta{i}'] for i in (1,2,3)]
  for fi in range(int(r['trials'])):tasks.append(dict(state_id=r['state_id'],controller=f"local_m{int(r['mode_id']):02d}_c{int(r['candidate_index']):02d}",mode_id=int(r['mode_id']),candidate_index=int(r['candidate_index']),eta=e,future_index=fi,phase='local_probe',split=r['split']))
 groups=defaultdict(list)
 for t in tasks:groups[(t['state_id'],t['controller'])].append(t)
 load=[0]*6;owner={}
 for g in sorted(groups):j=int(np.argmin(load));owner[g]=j;load[j]+=len(groups[g])
 p=H/'local_plans';p.mkdir(exist_ok=True)
 for j in range(6):
  with (p/f'shard{j}.jsonl').open('w') as f:
   for t in tasks:
    if owner[(t['state_id'],t['controller'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
 np.savez_compressed(H/'frozen_arrays.npz',features=np.load(SH/'state_features.npz')['features'],x=np.load(SH/'state_features.npz')['x'],state_ids=np.array([q['state_id'] for q in states]),splits=np.array([q['split'] for q in states]),anchors=A,halfspace_A=Ah,halfspace_b=bh)
 dump('frozen_manifest.json',{'codebook_sha256':sha(SH/'codebook_eta.csv'),'checkpoint_sha256':sha(Path(json.load(open(SH/'selected_model.json'))['checkpoint'])),'temperature':json.load(open(SH/'calibration.json')),'threshold':json.load(open(SH/'selected_threshold.json')),'state_split_sha256':sha(SH/'state_split.json'),'normalization_sha256':sha(SH/'normalization.json'),'selected_pairs':len(targets),'cached_usable_pairs':len(targets)-len({(r['state_id'],r['mode_id']) for r in probes}),'probed_pairs':len({(r['state_id'],r['mode_id']) for r in probes}),'probe_candidates':len(probes),'continuations':len(tasks),'shards':load})
 dump('working_state.json',{'status':'LOCAL_PROBES_FROZEN','completed':['frozen_inputs','cached_local_evidence','selected_mode_pairs','local_probe_manifest'],'next_action':'run TRAIN/VAL local probes'})
 print(json.dumps(json.load(open(H/'frozen_manifest.json')),indent=2))
if __name__=='__main__':main()

