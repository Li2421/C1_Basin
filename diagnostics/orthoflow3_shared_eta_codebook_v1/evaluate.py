#!/usr/bin/env python3
"""Freeze threshold on VAL, build TEST selector/oracle, and plan comparators."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1';DIRECT=D/'orthoflow3_direct_eta_baseline_v1'
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);TAUS=(.50,.60,.70,.80,.90,.95)
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(rows[0]);
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
class GPoint(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)
class GLowJ(nn.Module):
 @nn.compact
 def __call__(self,x):
  lo=jnp.array([0.,-.5,0.]);hi=jnp.array([1.25,.5,.75]);x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return lo+nn.sigmoid(nn.Dense(3)(x))*(hi-lo)
def raw():
 out=[]
 for p in sorted((H/'raw').glob('shard*.jsonl')):out += [json.loads(x) for x in open(p) if x.strip()]
 # Cached point-search records predate the matrix schema; normalize their
 # metadata deterministically from the frozen state split and codebook.
 a=np.load(H/'state_features.npz');splits={str(s):str(q) for s,q in zip(a['state_ids'],a['splits'])}
 cb=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in read(H/'codebook_eta.csv')])
 for r in out:
  r['split']=r.get('split',splits[r['state_id']])
  if 'controller' not in r:
   e=np.asarray(r['eta'],float)
   if np.max(np.abs(e))<1e-12:r['controller']='safety';r['mode_id']=-1
   else:
    m=int(np.argmin(np.linalg.norm(cb-e,axis=1)))
    if np.max(np.abs(cb[m]-e))>1e-8:raise RuntimeError(('unmapped cached eta',r['state_id'],e.tolist()))
    r['controller']=f'mode_{m:02d}';r['mode_id']=m
 return out
def pred(split):
 sel=json.load(open(H/'selected_model.json'));cal=json.load(open(H/'calibration.json'));z=np.load(H/sel['model']/'predictions.npz')[split];T=cal['temperature'] if cal['used'] else 1.;p=1/(1+np.exp(-np.clip(z/T,-30,30)));return p,sel,cal
def select_threshold():
 a=np.load(H/'state_features.npz');ids=a['state_ids'][a['splits']=='val'];p,sel,cal=pred('val');mrows={(r['state_id'],int(r['mode_id'])):r for r in read(H/'val_mode_counts.csv')};safe={r['state_id']:r for r in read(H/'safety_val_q32.csv')};rr=raw();outcome={(r['state_id'],r['controller'],int(r['future_index'])):r for r in rr if r['split']=='val'};rows=[];summ=[]
 for tau in TAUS:
  total=dead=brk=0
  for i,sid0 in enumerate(ids):
   sid=str(sid0);mode=int(np.argmax(p[i]));conf=float(p[i,mode]);ab=conf<tau;r=safe[sid] if ab else mrows[(sid,mode)];ctl='safety' if ab else f'mode_{mode:02d}';total+=int(r['successes']);dead+=int(r['deadlock'])
   for fi in range(32):brk+=outcome[(sid,'safety',fi)]['success'] and not outcome[(sid,ctl,fi)]['success']
   rows.append(dict(tau=tau,state_id=sid,selected_mode=mode,confidence=conf,abstained=ab,successes=r['successes'],trials=32,empirical_Q=r['empirical_Q'],deadlock=r['deadlock'],timeout=r['timeout'],collision=r['collision']))
  summ.append(dict(tau=tau,total_successes=total,breaks=int(brk),deadlock=dead))
 win=sorted(summ,key=lambda r:(-r['total_successes'],r['breaks'],r['deadlock'],-r['tau']))[0];dump('selected_threshold.json',{**win,'grid':list(TAUS),'selection':'VAL only: successes, breaks, deadlocks, larger tau','temperature_used':cal['used'],'selected_model':sel['model']});write('val_selector.csv',rows);print(json.dumps(win))
def test_selector():
 a=np.load(H/'state_features.npz');ids=a['state_ids'][a['splits']=='test'];p,sel,cal=pred('test');tau=json.load(open(H/'selected_threshold.json'))['tau'];mrows={(r['state_id'],int(r['mode_id'])):r for r in read(H/'test_mode_q64.csv')};safe={r['state_id']:r for r in read(H/'safety_test_q64.csv')};rr=raw();out={(r['state_id'],r['controller'],int(r['future_index'])):r for r in rr if r['split']=='test'};selector=[];oracle=[];reg=[];rb=[]
 for i,sid0 in enumerate(ids):
  sid=str(sid0);mode=int(np.argmax(p[i]));conf=float(p[i,mode]);ab=conf<tau;sr=safe[sid] if ab else mrows[(sid,mode)];ctl='safety' if ab else f'mode_{mode:02d}';allm=[mrows[(sid,m)] for m in range(p.shape[1])];best=sorted(allm,key=lambda r:(-float(r['empirical_Q']),int(r['mode_id'])))[0]
  selector.append(dict(state_id=sid,selected_mode=mode,confidence=conf,abstained=ab,**{k:sr[k] for k in ('successes','trials','empirical_Q','B63','deadlock','timeout','collision','J_def_mean','episode_length_mean')}));oracle.append(dict(state_id=sid,oracle_mode=best['mode_id'],**{k:best[k] for k in ('successes','trials','empirical_Q','B63','deadlock','timeout','collision','J_def_mean','episode_length_mean')}));reg.append(dict(state_id=sid,oracle_mode=best['mode_id'],oracle_Q64=best['empirical_Q'],selected_mode=mode,selector_Q64=sr['empirical_Q'],abstained=ab,regret=float(best['empirical_Q'])-float(sr['empirical_Q'])))
  rescue=brk=0
  for fi in range(64):
   ss=out[(sid,'safety',fi)]['success'];pp=out[(sid,ctl,fi)]['success'];rescue+=(not ss) and pp;brk+=ss and (not pp)
  rb.append(dict(state_id=sid,rescue=rescue,break_count=brk,net=rescue-brk))
 write('test_selector.csv',selector);write('codebook_oracle_coverage.csv',oracle);write('selector_regret.csv',reg);write('rescue_break.csv',rb)
 # Prevalence and common-mode audit.
 prev=[]
 for split,fn in [('train','train_mode_counts.csv'),('val','val_mode_counts.csv'),('test','test_mode_q64.csv')]:
  z=read(H/fn)
  for m in range(p.shape[1]):
   v=[q for q in z if int(q['mode_id'])==m];prev.append(dict(split=split,mode_id=m,mean_success_probability=float(np.mean([float(q['empirical_Q']) for q in v])),B63_states=sum(q['B63']=='True' for q in v) if split=='test' else '',states=len(v),high_success_states=sum(float(q['empirical_Q'])>=.9 for q in v)))
 write('mode_prevalence.csv',prev)
 # Frozen TRAIN-prior common-mode baseline, plus per-mode calibration of the
 # selected learner. Neither is allowed to alter selector deployment.
 trrows=read(H/'train_mode_counts.csv');trmean=[np.mean([float(r['empirical_Q']) for r in trrows if int(r['mode_id'])==m]) for m in range(p.shape[1])];cm=int(np.argmax(trmean));common=[mrows[(str(s),cm)] for s in ids];write('common_mode_audit.csv',common)
 pm=[]
 for split,fn in [('train','train_mode_counts.csv'),('val','val_mode_counts.csv'),('test','test_mode_q64.csv')]:
  pp,_,_=pred(split);yy=np.array([[float(q['empirical_Q']) for q in sorted([r for r in read(H/fn) if r['state_id']==str(s)],key=lambda r:int(r['mode_id']))] for s in a['state_ids'][a['splits']==split]])
  for m in range(p.shape[1]):pm.append(dict(split=split,mode_id=m,mean_predicted_probability=float(pp[:,m].mean()),mean_empirical_probability=float(yy[:,m].mean()),brier=float(np.mean((pp[:,m]-yy[:,m])**2))))
 write('per_mode_calibration.csv',pm)
 dump('selector_summary.json',{'selector_B63':sum(r['B63']=='True' for r in selector),'selector_mean_Q64':float(np.mean([float(r['empirical_Q']) for r in selector])),'oracle_B63_coverage':sum(r['B63']=='True' for r in oracle)/len(oracle),'oracle_mean_Q64':float(np.mean([float(r['empirical_Q']) for r in oracle])),'mean_regret':float(np.mean([r['regret'] for r in reg])),'rescue':sum(r['rescue'] for r in rb),'break':sum(r['break_count'] for r in rb),'train_prior_common_mode':cm,'common_mode_B63_states':sum(r['B63']=='True' for r in common),'common_mode_mean_Q64':float(np.mean([float(r['empirical_Q']) for r in common])),'max_single_mode_B63_fraction':max(sum(q['B63']=='True' and int(q['mode_id'])==m for q in read(H/'test_mode_q64.csv'))/len(oracle) for m in range(p.shape[1]))})
 print(json.dumps(json.load(open(H/'selector_summary.json')),indent=2))
def prepare_comparators():
 a=np.load(H/'state_features.npz');mask=a['splits']=='test';ids=a['state_ids'][mask];F=a['features'][mask];pn=json.load(open(POINT/'normalization.json'));px=((F-np.array(pn['h_mean']))/np.array(pn['h_std'])).astype(np.float32);ps=json.load(open(POINT/'selected_checkpoint.json'));mp=GPoint();t=mp.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));pp=serialization.from_bytes(t,Path(ps['checkpoint']).read_bytes());zp=np.asarray(mp.apply(pp,jnp.asarray(px)));ep=AFF+SCALE*zp
 ln=json.load(open(DIRECT/'normalization.json'));lx=((F-np.array(ln['h_mean']))/np.array(ln['h_std'])).astype(np.float32);ls=json.load(open(DIRECT/'selected_checkpoint.json'));ml=GLowJ();t=ml.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));lp=serialization.from_bytes(t,Path(ls['checkpoint']).read_bytes());el=np.asarray(ml.apply(lp,jnp.asarray(lx)))
 tasks=[];predrows=[]
 for sid,e1,e2 in zip(ids,ep,el):
  predrows += [dict(state_id=str(sid),controller='g_point_t0',eta1=e1[0],eta2=e1[1],eta3=e1[2]),dict(state_id=str(sid),controller='g_lowj',eta1=e2[0],eta2=e2[1],eta3=e2[2])]
  for ctl,e in [('g_point_t0',e1),('g_lowj',e2)]:
   for fi in range(64):tasks.append(dict(state_id=str(sid),split='test',controller=ctl,eta=e.tolist(),future_index=fi,phase='test_comparator'))
 write('comparator_predictions.csv',predrows);groups=defaultdict(list)
 for t in tasks:groups[(t['state_id'],t['controller'])].append(t)
 load=[0]*6;owner={}
 for g in sorted(groups):j=int(np.argmin(load));owner[g]=j;load[j]+=len(groups[g])
 pth=H/'comparator_plans6';pth.mkdir(exist_ok=True)
 for j in range(6):
  with (pth/f'shard{j}.jsonl').open('w') as f:
   for t in tasks:
    if owner[(t['state_id'],t['controller'])]==j:f.write(json.dumps(t,sort_keys=True)+'\n')
 dump('comparator_plan.json',{'tasks':len(tasks),'shards':load,'point_checkpoint':ps,'lowj_checkpoint':ls});print(json.dumps({'tasks':len(tasks),'shards':load}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['threshold','test','comparators']);a=ap.parse_args();{'threshold':select_threshold,'test':test_selector,'comparators':prepare_comparators}[a.stage]()
if __name__=='__main__':main()
