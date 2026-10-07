#!/usr/bin/env python3
"""Prepare predictions and aggregate true closed-loop VAL/TEST evaluation."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1';DIRECT=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);SEEDS=(17,23,41);SHARDS=6
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def read(p):return list(csv.DictReader(open(p)))
def write(p,rr,fields=None):
 fields=fields or (list(rr[0]) if rr else ['state_id'])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
class G(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)
class GLowJ(nn.Module):
 @nn.compact
 def __call__(self,x):
  low=jnp.array([0.,-.5,0.]);high=jnp.array([1.25,.5,.75]);x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return low+nn.sigmoid(nn.Dense(3)(x))*(high-low)
def load_g(path):
 m=G();t=m.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));return m,serialization.from_bytes(t,Path(path).read_bytes())
def arrays():return np.load(HERE/'point_learning_arrays.npz')
def predict_seed(seed,mask):
 a=arrays();s=json.load(open(HERE/f'seed{seed}/summary.json'));m,p=load_g(s['checkpoint']);yn=np.asarray(m.apply(p,jnp.asarray(a['x'][mask])));return yn,AFF+SCALE*yn,s
def write_plan(name,tasks):
 d=HERE/'plans'/name;d.mkdir(parents=True,exist_ok=True);groups=sorted({(x['state_id'],x['controller']) for x in tasks});owner={g:i%SHARDS for i,g in enumerate(groups)}
 for s in range(SHARDS):
  with open(d/f'shard{s}.jsonl','w') as f:
   for x in tasks:
    if owner[(x['state_id'],x['controller'])]==s:f.write(json.dumps(x,sort_keys=True)+'\n')
 dump(d/'summary.json',{'tasks':len(tasks),'shards':SHARDS,'groups':len(groups),'files':[{'shard':s,'sha256':sha(d/f'shard{s}.jsonl')} for s in range(SHARDS)]})
def rollout_rows(name):
 rr=[]
 for p in sorted((HERE/'raw'/name).glob('shard*.jsonl')):
  rr += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return rr

def prepare_val():
 a=arrays();mask=a['splits']=='val';ids=a['state_ids'][mask];out=[];tasks=[]
 for seed in SEEDS:
  yn,yp,s=predict_seed(seed,mask)
  for sid,norm,eta in zip(ids,yn,yp):
   out.append({'state_id':str(sid),'seed':seed,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'eta_norm1':norm[0],'eta_norm2':norm[1],'eta_norm3':norm[2],'checkpoint':s['checkpoint'],'checkpoint_sha256':s['checkpoint_sha256']})
   for fi in range(16):tasks.append({'state_id':str(sid),'controller':f'seed{seed}','seed':seed,'eta':eta.tolist(),'future_index':fi,'phase':'val16','checkpoint_sha256':s['checkpoint_sha256']})
 write(HERE/'val_predictions_all_seeds.csv',out);write_plan('val16',tasks);print(json.dumps({'predictions':len(out),'tasks':len(tasks)}))
def aggregate_val():
 pred=read(HERE/'val_predictions_all_seeds.csv');rr=rollout_rows('val16');rows=[]
 for seed in SEEDS:
  ctl=f'seed{seed}';x=[z for z in rr if z['controller']==ctl];by=defaultdict(list)
  for z in x:by[z['state_id']].append(z)
  sm=json.load(open(HERE/f'seed{seed}/summary.json'));success=sum(z['success'] for z in x);dead=sum(z['outcome']=='safe_deadlock' for z in x);tout=sum(z['outcome']=='timeout' for z in x);coll=sum(z['outcome']=='collision' for z in x);full=sum(sum(z['success'] for z in v)==16 for v in by.values());j=[z['J_def'] for z in x if z['success']]
  rows.append({'seed':seed,'total_successes':success,'trials':len(x),'states_16of16':full,'deadlock':dead,'timeout':tout,'collision':coll,'successful_J_def_mean':float(np.mean(j)) if j else None,'successful_J_def_median':float(np.median(j)) if j else None,'val_eta_mse':sm['best_val_mse'],'checkpoint':sm['checkpoint'],'checkpoint_sha256':sm['checkpoint_sha256']})
 rows.sort(key=lambda x:(-x['total_successes'],-x['states_16of16'],x['deadlock'],x['timeout'],x['val_eta_mse'],x['seed']));win=rows[0];write(HERE/'val_closedloop.csv',rows);dump(HERE/'selected_checkpoint.json',win);(HERE/'checkpoint_sha256.txt').write_text(win['checkpoint_sha256']+'\n');print(json.dumps(win,indent=2))
def predict_lowj(mask):
 a=arrays();norm=json.load(open(DIRECT/'normalization.json'));x=((a['features'][mask]-np.asarray(norm['h_mean']))/np.asarray(norm['h_std'])).astype(np.float32);sel=json.load(open(DIRECT/'selected_checkpoint.json'));m=GLowJ();t=m.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));p=serialization.from_bytes(t,Path(sel['checkpoint']).read_bytes());return np.asarray(m.apply(p,jnp.asarray(x))),sel
def prepare_test():
 a=arrays();mask=a['splits']=='test';ids=a['state_ids'][mask];sel=json.load(open(HERE/'selected_checkpoint.json'));yn,yp,_=predict_seed(int(sel['seed']),mask);out=[];tasks=[]
 for sid,target,norm,eta in zip(ids,a['physical_target'][mask],yn,yp):
  err=float(np.linalg.norm(norm-(target-AFF)/SCALE));out.append({'state_id':str(sid),'controller':'g_point_t0','eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'target_eta1':target[0],'target_eta2':target[1],'target_eta3':target[2],'normalized_eta_error':err,'checkpoint_sha256':sel['checkpoint_sha256']})
  for fi in range(64):tasks.append({'state_id':str(sid),'controller':'g_point_t0','eta':eta.tolist(),'future_index':fi,'phase':'test64','checkpoint_sha256':sel['checkpoint_sha256']})
 for sid in ids:
  for fi in range(64):tasks.append({'state_id':str(sid),'controller':'safety','eta':[0.,0.,0.],'future_index':fi,'phase':'test64'})
 low,lows=predict_lowj(mask)
 for sid,eta in zip(ids,low):
  for fi in range(64):tasks.append({'state_id':str(sid),'controller':'g_lowj','eta':eta.tolist(),'future_index':fi,'phase':'test64','checkpoint_sha256':lows['checkpoint_sha256']})
 write(HERE/'test_predictions.csv',out);write_plan('test64',tasks);dump(HERE/'frozen_lowj_reference.json',{'status':'COMPATIBLE','checkpoint':lows,'checkpoint_hash_verified':sha(lows['checkpoint'])==lows['checkpoint_sha256'],'retrained':False});print(json.dumps({'predictions':len(out),'tasks':len(tasks)}))
def aggregate_test():
 pred=read(HERE/'test_predictions.csv');rr=rollout_rows('test64');by=defaultdict(list)
 for x in rr:by[(x['state_id'],x['controller'])].append(x)
 rows=[]
 for (sid,ctl),v in sorted(by.items()):
  out=Counter(x['outcome'] for x in v);s=sum(x['success'] for x in v);j=[x['J_def'] for x in v if x['success']];ln=[x['continuation_steps'] for x in v]
  rows.append({'state_id':sid,'controller':ctl,'successes':s,'trials':len(v),'Q64':s/64,'B63':s>=63,'deadlock':out['safe_deadlock'],'timeout':out['timeout'],'collision':out['collision'],'successful_J_def_mean':float(np.mean(j)) if j else None,'successful_J_def_median':float(np.median(j)) if j else None,'episode_length_mean':float(np.mean(ln))})
 write(HERE/'test_q64.csv',[x for x in rows if x['controller']=='g_point_t0']);write(HERE/'safety_test_q64.csv',[x for x in rows if x['controller']=='safety']);write(HERE/'lowj_test_q64.csv',[x for x in rows if x['controller']=='g_lowj'])
 paired=[]
 for sid in sorted({x['state_id'] for x in rr}):
  p={x['future_index']:x for x in by[(sid,'g_point_t0')]};s={x['future_index']:x for x in by[(sid,'safety')]}
  rescue=sum((not s[i]['success']) and p[i]['success'] for i in p);brk=sum(s[i]['success'] and (not p[i]['success']) for i in p);paired.append({'state_id':sid,'rescue':rescue,'break':brk,'net':rescue-brk})
 write(HERE/'rescue_break.csv',paired);geo={x['state_id']:x for x in pred};err=[]
 for x in rows:
  if x['controller']=='g_point_t0':err.append({**x,'normalized_eta_error':geo[x['state_id']]['normalized_eta_error']})
 write(HERE/'eta_error_vs_control.csv',err);print(json.dumps({'rows':len(rows),'rescue':sum(x['rescue'] for x in paired),'break':sum(x['break'] for x in paired)},indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare-val','aggregate-val','prepare-test','aggregate-test']);a=ap.parse_args();{'prepare-val':prepare_val,'aggregate-val':aggregate_val,'prepare-test':prepare_test,'aggregate-test':aggregate_test}[a.stage]()
if __name__=='__main__':main()
