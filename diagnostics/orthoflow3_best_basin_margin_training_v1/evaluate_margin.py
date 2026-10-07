#!/usr/bin/env python3
"""Frozen VAL selection and held-out TEST evaluation for G_MARGIN."""
from __future__ import annotations
import argparse,csv,importlib.util,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1';SHARED=D/'orthoflow3_shared_conservative_family_v1'
sys.path.insert(0,str(SHARED));from shared_families import contains,violation
sys.path.insert(0,str(H));from pipeline import fit_local_body,force_anchor
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);SEEDS=(17,23,41)
def read(p):return list(csv.DictReader(open(p)))
def write(n,rows,fields=None):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(n,x):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
class G(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)
def load(seed):
 m=G();p0=m.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));p=serialization.from_bytes(p0,(H/f'seed{seed}/checkpoint.msgpack').read_bytes());return m,p
def predictions(seed,mask):
 a=np.load(H/'margin_arrays.npz');m,p=load(seed);z=np.asarray(m.apply(p,jnp.asarray(a['x'][mask])));return z,AFF+SCALE*z
def plan(name,tasks):
 d=H/f'plans/{name}';d.mkdir(parents=True,exist_ok=True)
 for sh in range(2):
  with open(d/f'shard{sh}.jsonl','w') as f:
   for t in tasks:
    if hash((t['state_id'],t['controller']))%2==sh:f.write(json.dumps(t,sort_keys=True)+'\n')
def prepare_val():
 a=np.load(H/'margin_arrays.npz');mask=a['splits']=='val';rows=[];tasks=[]
 for seed in SEEDS:
  z,e=predictions(seed,mask)
  for sid,zz,eta in zip(a['state_ids'][mask],z,e):
   rows.append(dict(state_id=str(sid),seed=seed,eta1=eta[0],eta2=eta[1],eta3=eta[2],z1=zz[0],z2=zz[1],z3=zz[2]))
   for fi in range(16):tasks.append(dict(state_id=str(sid),controller=f'g_margin_seed{seed}',eta=eta.tolist(),future_index=fi,phase='val16'))
 write('val_predictions_all_seeds.csv',rows);plan('val16',tasks);print(json.dumps({'tasks':len(tasks)}))
def run(name,shard):
 tasks=[json.loads(x) for x in open(H/f'plans/{name}/shard{shard}.jsonl') if x.strip()];states={x['state_id']:dict(x,feature_index=x['dataset_index']) for x in json.load(open(POINT/'final_source_split.json'))['states'] if x['state_id'] in {t['state_id'] for t in tasks}};features=np.load(POINT/'point_learning_arrays.npz')['features'];spec=importlib.util.spec_from_file_location('point_driver',POINT/'run_eval_shard.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);out=H/f'runs/{name}/shard{shard}';out.mkdir(parents=True,exist_ok=True);m,O=mod.loadmod(out,states);o=O(states,features,np.zeros(3),np.ones(3));st=time.time();res=o.ensure(tasks,name);dest=H/f'raw/{name}';dest.mkdir(parents=True,exist_ok=True)
 with open(dest/f'shard{shard}.jsonl','w') as f:
  for r in res:f.write(json.dumps({k:v for k,v in r.items() if k!='_source'},sort_keys=True)+'\n')
 dump(f'raw/{name}/shard{shard}_runtime.json',dict(tasks=len(tasks),new_continuations=o.new,reused_continuations=len(tasks)-o.new,physical_steps=o.steps,wall_seconds=time.time()-st))
def raw(name):
 out=[]
 for p in (H/f'raw/{name}').glob('shard*.jsonl'):out += [json.loads(x) for x in open(p) if x.strip()]
 return out
def aggregate_val():
 rr=raw('val16');by=defaultdict(list)
 for r in rr:by[r['controller']].append(r)
 rows=[]
 for seed in SEEDS:
  x=by[f'g_margin_seed{seed}'];ss=sum(q['success'] for q in x);states=defaultdict(list)
  for q in x:states[q['state_id']].append(q)
  rows.append(dict(seed=seed,total_successes=ss,trials=len(x),states_16of16=sum(sum(q['success'] for q in v)==16 for v in states.values()),deadlock=sum(q['outcome']=='safe_deadlock' for q in x),timeout=sum(q['outcome']=='timeout' for q in x),collision=sum(q['outcome']=='collision' for q in x),val_margin_loss=json.load(open(H/f'seed{seed}/summary.json'))['best_val_loss']))
 rows.sort(key=lambda r:(-r['total_successes'],-r['states_16of16'],r['deadlock'],r['timeout'],r['val_margin_loss'],r['seed']));write('val_closedloop.csv',rows);dump('selected_checkpoint.json',dict(**rows[0],checkpoint=str(H/f"seed{rows[0]['seed']}/checkpoint.msgpack")));print(json.dumps(rows[0]))
def fit_test_models():
 evidence=read(H/'geometry_evidence_inventory.csv');split={x['state_id']:x for x in json.load(open(POINT/'final_source_split.json'))['states']};targets={x['state_id']:x for x in read(POINT/'selected_eta_targets.csv')};out={}
 for sid,st in split.items():
  if st['split']!='test':continue
  v=[r for r in evidence if r['state_id']==sid];P=np.array([[float(r['z1']),float(r['z2']),float(r['z3'])] for r in v if r['B63']=='True']);N=np.array([[float(r['z1']),float(r['z2']),float(r['z3'])] for r in v if r['B63']=='False']).reshape(-1,3);a=(np.array([float(targets[sid][f'target_eta{i}']) for i in (1,2,3)])-AFF)/SCALE;P=force_anchor(P,a);out[sid]=fit_local_body(P,N,np.ones(len(P)))
 dump('test_diagnostic_models.json',out);return out
def prepare_test():
 sel=json.load(open(H/'selected_checkpoint.json'));a=np.load(H/'margin_arrays.npz');mask=a['splits']=='test';z,e=predictions(int(sel['seed']),mask);models=fit_test_models();rows=[];tasks=[]
 for sid,zz,eta in zip(a['state_ids'][mask],z,e):
  m=models[str(sid)];inner=json.loads(json.dumps(m));inner['gamma']*=.8
  rows.append(dict(state_id=str(sid),controller='g_margin',eta1=eta[0],eta2=eta[1],eta3=eta[2],z1=zz[0],z2=zz[1],z3=zz[2],inside_inner=bool(contains(inner,zz[None],True)[0]),inside_retained=bool(contains(m,zz[None],True)[0]),inside_full=bool(contains(m,zz[None],False)[0]),normalized_violation=float(violation(inner,zz[None],True)[0])))
  for fi in range(64):tasks.append(dict(state_id=str(sid),controller='g_margin',eta=eta.tolist(),future_index=fi,phase='test64'))
 write('test_predictions.csv',rows);plan('test64',tasks);print(json.dumps({'tasks':len(tasks)}))
def aggregate_test():
 rr=raw('test64');by=defaultdict(list)
 for r in rr:by[r['state_id']].append(r)
 pred={r['state_id']:r for r in read(H/'test_predictions.csv')};rows=[]
 for sid,v in sorted(by.items()):
  s=sum(q['success'] for q in v);out=Counter(q['outcome'] for q in v);ok=[q for q in v if q['success']]
  d=dict(pred[sid]);d.update(successes=s,trials=len(v),Q64=s/64,B63=s>=63,deadlock=out['safe_deadlock'],timeout=out['timeout'],collision=out['collision'],J_def_mean=float(np.mean([q['J_def'] for q in ok])) if ok else '',episode_length_mean=float(np.mean([q['continuation_steps'] for q in v]))) ; rows.append(d)
 write('test_q64.csv',rows);print(json.dumps({'states':len(rows),'success':sum(r['successes'] for r in rows)}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare-val','run','aggregate-val','prepare-test','aggregate-test']);ap.add_argument('--plan');ap.add_argument('--shard',type=int);a=ap.parse_args();{'prepare-val':prepare_val,'run':lambda:run(a.plan,a.shard),'aggregate-val':aggregate_val,'prepare-test':prepare_test,'aggregate-test':aggregate_test}[a.stage]()
if __name__=='__main__':main()
