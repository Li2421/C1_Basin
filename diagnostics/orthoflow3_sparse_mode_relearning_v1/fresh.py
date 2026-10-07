#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
import sys;sys.path.insert(0,str(Path(__file__).parent))
from train import MLP

H=Path(__file__).parent;D=H.parent;S=D/'orthoflow3_shared_eta_codebook_v1';F=D/'orthoflow3_fresh_selector_vs_fixed_v1'
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def frozen_predictions():
 cfg=json.load(open(H/'sparse_codebook.json'));sel=json.load(open(H/'selected_model.json'));cal=json.load(open(H/'calibration.json'));thr=json.load(open(H/'selected_threshold.json'));modes=cfg['selected_original_mode_ids'];fa=np.load(F/'fresh_state_features.npz');norm=json.load(open(S/'normalization.json'));x=((fa['features']-np.array(norm['h_mean']))/np.array(norm['h_std'])).astype(np.float32);model=MLP(len(modes));tmp=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));p=serialization.from_bytes(tmp,Path(sel['checkpoint']).read_bytes());log=np.asarray(model.apply(p,jnp.asarray(x)));T=cal['temperature'] if cal['used'] else 1.;prob=1/(1+np.exp(-np.clip(log/T,-30,30)));rows=[]
 for sid,pr in zip(fa['state_ids'],prob):
  lm=int(np.argmax(pr));rows.append(dict(state_id=str(sid),sparse_mode_id=lm,original_mode_id=modes[lm],confidence=float(pr[lm]),abstained=bool(pr[lm]<thr['tau']),fixed_original_mode=cfg['best_fixed_original_mode']))
 return rows
def prior_raw():
 out=[]
 for p in sorted((F/'raw').glob('shard*.jsonl')):out += [json.loads(x) for x in open(p) if x.strip()]
 return out
def eta_modes():
 cb=read(S/'codebook_eta.csv');return {int(r['mode_id']):np.array([float(r[f'eta{i}']) for i in (1,2,3)]) for r in cb}
def prepare():
 pred=frozen_predictions();write('fresh_predictions.csv',pred); modes=json.load(open(H/'sparse_codebook.json'))['selected_original_mode_ids'];sel=json.load(open(H/'selected_model.json'));cal=json.load(open(H/'calibration.json'));thr=json.load(open(H/'selected_threshold.json'));eta=eta_modes();old=prior_raw();cache=defaultdict(list)
 for r in old:
  e=np.asarray(r['eta'],float)
  for m in modes:
   if np.max(np.abs(e-eta[m]))<=1e-10:cache[(r['state_id'],m)].append(r)
 needed=set()
 for r in pred:
  sid=r['state_id'];needed.add((sid,int(r['fixed_original_mode'])))
  if not r['abstained']:needed.add((sid,int(r['original_mode_id'])))
 cached={k:v for k,v in cache.items() if k in needed and len(v)==64};todo=sorted(needed-set(cached));tasks=[]
 for sid,m in todo:
  for fi in range(64):tasks.append(dict(state_id=sid,mode_id=m,controller=f'sparse_mode_{m:02d}',eta=eta[m].tolist(),future_index=fi,phase='sparse_fresh_primary'))
 pth=H/'fresh_plans';pth.mkdir(exist_ok=True);groups=defaultdict(list)
 for t in tasks:groups[(t['state_id'],t['mode_id'])].append(t)
 load=[0]*6;owner={}
 for g in sorted(groups):j=int(np.argmin(load));owner[g]=j;load[j]+=len(groups[g])
 for j in range(6):
  with (pth/f'shard{j}.jsonl').open('w') as f:
   for t in tasks:
    if owner[(t['state_id'],t['mode_id'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
 with (H/'fresh_cached.jsonl').open('w') as f:
  for (sid,m),v in sorted(cached.items()):
   for r in v:r=dict(r);r['mode_id']=m;f.write(json.dumps(r,sort_keys=True)+'\n')
 manifest={'cohort_source':str(F/'fresh_state_manifest.json'),'states':len(pred),'source_isolated_from_codebook':True,'frozen_model':sel,'frozen_temperature':cal,'frozen_threshold':thr,'required_state_mode_pairs':len(needed),'cached_pairs':len(cached),'new_pairs':len(todo),'new_continuations':len(tasks),'shards':load};dump('fresh_manifest.json',manifest);print(json.dumps(manifest,indent=2))
def collect_all():
 rr=[json.loads(x) for x in open(H/'fresh_cached.jsonl') if x.strip()]
 for p in sorted((H/'fresh_raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 return rr
def metric(v):
 o=Counter(x['outcome'] for x in v);k=sum(x['success'] for x in v);js=[x['J_def'] for x in v if x['success']]
 return dict(successes=k,trials=len(v),Q64=k/len(v),B63=k>=63,deadlock=o['safe_deadlock'],timeout=o['timeout'],collision=o['collision'],J_def=float(np.mean(js)) if js else None,episode_length=float(np.mean([x['continuation_steps'] for x in v])))
def aggregate():
 pred=frozen_predictions();rr=collect_all();by=defaultdict(list)
 for r in rr:by[(r['state_id'],int(r['mode_id']))].append(r)
 safety=defaultdict(list)
 for r in prior_raw():
  if np.max(np.abs(np.asarray(r['eta'],float)))<1e-12:safety[r['state_id']].append(r)
 rows=[];audit=set()
 for p in pred:
  sid=p['state_id'];fm=int(p['fixed_original_mode']);sm=int(p['original_mode_id']);f=metric(by[(sid,fm)]);s=metric(safety[sid]) if p['abstained'] else metric(by[(sid,sm)]);row=dict(state_id=sid,selector_mode=sm,abstained=p['abstained'],fixed_mode=fm)
  for k,v in f.items():row['fixed_'+k]=v
  for k,v in s.items():row['selector_'+k]=v
  rows.append(row)
  if not s['B63']:
   for m in json.load(open(H/'sparse_codebook.json'))['selected_original_mode_ids']:
    if len(by[(sid,m)])!=64:audit.add((sid,m))
 write('fresh_replication.csv',rows);eta=eta_modes();tasks=[]
 for sid,m in sorted(audit):
  for fi in range(64):tasks.append(dict(state_id=sid,mode_id=m,controller=f'sparse_audit_{m:02d}',eta=eta[m].tolist(),future_index=fi,phase='sparse_fresh_failure_audit'))
 pth=H/'audit_plans';pth.mkdir(exist_ok=True);load=[0]*6;owner={};groups=defaultdict(list)
 for t in tasks:groups[(t['state_id'],t['mode_id'])].append(t)
 for g in sorted(groups):j=int(np.argmin(load));owner[g]=j;load[j]+=64
 for j in range(6):
  with (pth/f'shard{j}.jsonl').open('w') as f:
   for t in tasks:
    if owner[(t['state_id'],t['mode_id'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
 dump('fresh_audit_manifest.json',{'selector_nonB63_states':sum(not r['selector_B63'] for r in rows),'missing_oracle_pairs':len(audit),'new_continuations':len(tasks),'shards':load});print(json.dumps(json.load(open(H/'fresh_audit_manifest.json')),indent=2))
def finalize():
 rows=read(H/'fresh_replication.csv');rr=collect_all()
 for p in sorted((H/'audit_raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 by=defaultdict(list)
 for r in rr:by[(r['state_id'],int(r['mode_id']))].append(r)
 modes=json.load(open(H/'sparse_codebook.json'))['selected_original_mode_ids'];aud=[]
 for r in rows:
  if r['selector_B63']=='True':continue
  vals=[]
  for m in modes:
   vals.append(metric(by[(r['state_id'],m)]) if len(by[(r['state_id'],m)])==64 else None)
  valid=[(m,q) for m,q in zip(modes,vals) if q is not None];best=max(valid,key=lambda x:x[1]['Q64']) if valid else (None,None);aud.append(dict(state_id=r['state_id'],selector_mode=r['selector_mode'],selector_Q64=r['selector_Q64'],oracle_mode=best[0] if best[1] else '',oracle_Q64=best[1]['Q64'] if best[1] else '',oracle_B63=best[1]['B63'] if best[1] else '',diagnosis=('SELECTOR_PREDICTION_FAILURE' if best[1] and best[1]['B63'] else 'SPARSE_CODEBOOK_COVERAGE_FAILURE') if best[1] else 'UNDERRESOLVED'))
 write('fresh_failure_oracle_audit.csv',aud)
 fs=sum(int(r['fixed_successes']) for r in rows);ss=sum(int(r['selector_successes']) for r in rows);rescue=brk=0
 # Matched differences can be reconstructed from exact state/mode/safety raw.
 safety=defaultdict(list)
 for z in prior_raw():
  if np.max(np.abs(np.asarray(z['eta'],float)))<1e-12:safety[z['state_id']].append(z)
 for r in rows:
  sid=r['state_id'];fm=int(r['fixed_mode']);sm=int(r['selector_mode']);fv={int(z['future_index']):z for z in by[(sid,fm)]};sv={int(z['future_index']):z for z in (safety[sid] if r['abstained']=='True' else by[(sid,sm)])}
  rescue+=sum((not fv[i]['success']) and sv[i]['success'] for i in range(64));brk+=sum(fv[i]['success'] and (not sv[i]['success']) for i in range(64))
 action_rows=[]
 for action in ('3','9','safety'):
  q=[r for r in rows if ('safety' if r['abstained']=='True' else r['selector_mode'])==action]
  if q:action_rows.append(dict(action=action,states=len(q),selector_B63=sum(r['selector_B63']=='True' for r in q),selector_mean_Q64=float(np.mean([float(r['selector_Q64']) for r in q])),fixed_B63=sum(r['fixed_B63']=='True' for r in q),fixed_mean_Q64=float(np.mean([float(r['fixed_Q64']) for r in q]))))
 write('fresh_mode_selection_statistics.csv',action_rows)
 summary={'states':len(rows),'fixed_B63':sum(r['fixed_B63']=='True' for r in rows),'fixed_mean_Q64':fs/(64*len(rows)),'selector_B63':sum(r['selector_B63']=='True' for r in rows),'selector_mean_Q64':ss/(64*len(rows)),'selector_rescue_vs_fixed':rescue,'selector_break_vs_fixed':brk,'selector_nonB63':sum(r['selector_B63']!='True' for r in rows),'action_breakdown':action_rows,'failure_diagnosis':dict(Counter(r['diagnosis'] for r in aud))};dump('fresh_summary.json',summary);print(json.dumps(summary,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','aggregate','finalize']);a=ap.parse_args();{'prepare':prepare,'aggregate':aggregate,'finalize':finalize}[a.stage]()
if __name__=='__main__':main()
