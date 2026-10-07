#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
H=Path(__file__).parent
class MLP(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(64)(x));x=nn.silu(nn.Dense(64)(x));return nn.Dense(12)(x)
def write(name,rows):
 with open(H/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def agg(v):
 o=Counter(str(r['outcome']) for r in v);n=len(v);k=sum(bool(r['success']) for r in v)
 return dict(successes=k,trials=n,Q64=k/n,B63=k>=63,deadlock=sum(o[x] for x in ('strict_deadlock','safe_deadlock','deadlock')),timeout=o['timeout'],collision=sum('collision' in str(r['outcome']) for r in v),J_def=float(np.mean([r['J_def'] for r in v])),episode_length=float(np.mean([r['episode_steps'] for r in v])))
def main():
 states=json.load(open(H/'hard_state_manifest.json'))['states'];assets=json.load(open(H/'frozen_assets.json'));raw=[]
 for p in (H/'raw').glob('discrete*.jsonl'):raw += [json.loads(x) for x in open(p) if x.strip()]
 bad=[r for r in raw if r.get('scientific_outcome_valid') is not True]
 if bad:raise RuntimeError(f'{len(bad)} invalid rollout records require matched-block repair before aggregation')
 by=defaultdict(dict)
 for r in raw:by[(r['state_id'],r['controller'],int(r['mode_id']))][int(r['future_index'])]=r
 # Exact frozen h schema and prior TRAIN-only normalization/model.
 norm=assets['feature_normalization'];mean=np.asarray(norm['mean']);std=np.asarray(norm['std']);X=[]
 for s in states:
  c=json.load(open(H/'raw'/f"conditioning_{s['state_id']}.json"));x=np.asarray(c['h_feature'],float);assert x.shape==(80,);X.append((x-mean)/std)
 model=MLP();tmp=model.init(jax.random.PRNGKey(0),jnp.zeros((1,80)));params=serialization.from_bytes(tmp,Path(assets['selector']['checkpoint']).read_bytes());log=np.asarray(model.apply(params,jnp.asarray(np.asarray(X),dtype=jnp.float32)));T=assets['calibration']['temperature'] if assets['calibration']['used'] else 1.;P=1/(1+np.exp(-np.clip(log/T,-30,30)));tau=float(assets['threshold']['tau']);fixed=int(assets['best_fixed_mode'])
 mode_rows=[];state_rows=[];selections=Counter()
 for i,s in enumerate(states):
  sid=s['state_id'];sv=[by[(sid,'safety',-1)][j] for j in range(64)];safe=agg(sv);modes=[]
  for m in range(12):
   v=[by[(sid,'transformed',m)][j] for j in range(64)];q=agg(v);q.update(state_id=sid,source_group=s['source_group'],hard_rank=s['hard_rank'],hard_score=s['hard_score'],regime=s['regime'],mode_id=m,eta1=v[0]['eta'][0],eta2=v[0]['eta'][1],eta3=v[0]['eta'][2]);mode_rows.append(q);modes.append(q)
  oracle=max(modes,key=lambda q:(q['Q64'],-q['mode_id']));m=int(np.argmax(P[i]));conf=float(P[i,m]);ab=conf<tau;sel=safe if ab else modes[m];selections['safety' if ab else str(m)]+=1
  fixedq=modes[fixed]
  state_rows.append(dict(state_id=sid,source_group=s['source_group'],hard_rank=s['hard_rank'],hard_score=s['hard_score'],regime=s['regime'],
   safety_B63=safe['B63'],safety_Q64=safe['Q64'],safety_success=safe['successes'],safety_deadlock=safe['deadlock'],safety_timeout=safe['timeout'],safety_collision=safe['collision'],safety_J_def=safe['J_def'],safety_episode_length=safe['episode_length'],
   fixed_mode=fixed,fixed_B63=fixedq['B63'],fixed_Q64=fixedq['Q64'],fixed_success=fixedq['successes'],fixed_deadlock=fixedq['deadlock'],fixed_timeout=fixedq['timeout'],fixed_collision=fixedq['collision'],fixed_J_def=fixedq['J_def'],fixed_episode_length=fixedq['episode_length'],
   selector_mode=m,selector_confidence=conf,selector_abstained=ab,selector_B63=sel['B63'],selector_Q64=sel['Q64'],selector_success=sel['successes'],selector_deadlock=sel['deadlock'],selector_timeout=sel['timeout'],selector_collision=sel['collision'],selector_J_def=sel['J_def'],selector_episode_length=sel['episode_length'],
   oracle_mode=oracle['mode_id'],oracle_B63=oracle['B63'],oracle_Q64=oracle['Q64'],oracle_success=oracle['successes'],oracle_deadlock=oracle['deadlock'],oracle_timeout=oracle['timeout'],oracle_collision=oracle['collision'],oracle_J_def=oracle['J_def'],oracle_episode_length=oracle['episode_length'],feasible_modes=sum(q['B63'] for q in modes),mean_mode_Q64=float(np.mean([q['Q64'] for q in modes])),selector_regret=oracle['Q64']-sel['Q64']))
 write('transformed_mode_q64.csv',mode_rows);write('discrete_baselines.csv',state_rows)
 def summ(prefix):
  return dict(states=len(state_rows),B63_states=sum(r[prefix+'_B63'] in (True,'True') for r in state_rows),coverage=sum(r[prefix+'_B63'] in (True,'True') for r in state_rows)/len(state_rows),mean_Q64=float(np.mean([r[prefix+'_Q64'] for r in state_rows])),success=sum(r[prefix+'_success'] for r in state_rows),trials=64*len(state_rows),deadlock=sum(r[prefix+'_deadlock'] for r in state_rows),timeout=sum(r[prefix+'_timeout'] for r in state_rows),collision=sum(r[prefix+'_collision'] for r in state_rows),J_def=float(np.mean([r[prefix+'_J_def'] for r in state_rows])),episode_length=float(np.mean([r[prefix+'_episode_length'] for r in state_rows])))
 cnt=np.array([selections.get(str(m),0) for m in range(12)],float);cnt=cnt[cnt>0];entropy=float(-np.sum((cnt/cnt.sum())*np.log(cnt/cnt.sum()))/np.log(12)) if len(cnt)>1 else 0.
 out={'safety':summ('safety'),'fixed':summ('fixed'),'selector':summ('selector'),'codebook_oracle':summ('oracle'),'mean_feasible_modes':float(np.mean([r['feasible_modes'] for r in state_rows])),'median_feasible_modes':float(np.median([r['feasible_modes'] for r in state_rows])),'selector_mode_counts':dict(selections),'selector_entropy':entropy,'codebook_fail_states':[r['state_id'] for r in state_rows if not r['oracle_B63']],'codebook_fail_count':sum(not r['oracle_B63'] for r in state_rows),'fixed_asset_hashes_verified':all(Path(assets['prior_dir'],k).exists() and __import__('hashlib').sha256(Path(assets['prior_dir'],k).read_bytes()).hexdigest()==v for k,v in assets['asset_hashes'].items())}
 dump('discrete_summary.json',out);dump('continuous_search_targets.json',{'selection':'all and only hard states with no transformed-codebook B63 mode; no manual filtering','states':out['codebook_fail_states']});print(json.dumps(out,indent=2))
if __name__=='__main__':main()
