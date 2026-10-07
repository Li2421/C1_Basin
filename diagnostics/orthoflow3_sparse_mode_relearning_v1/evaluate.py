#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import sys;sys.path.insert(0,str(Path(__file__).parent))
from train import metrics

H=Path(__file__).parent;S=H.parent/'orthoflow3_shared_eta_codebook_v1';TAUS=(.50,.60,.70,.80,.90,.95)
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;fields=fields or (list(rows[0]) if rows else ['state_id']);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def cfg():return json.load(open(H/'sparse_codebook.json'))
def features():return np.load(H/'state_features.npz')
def countmap(split):return {(r['state_id'],int(r['sparse_mode_id'])):r for r in read(H/f'sparse_{split}_counts.csv')}
def logits(model,split):
 a=features();n=sum(a['splits']==split);c=cfg();M=c['M_sparse']
 if model=='global_prior':
  p=np.array(json.load(open(H/'models/global_prior.json'))['train_prior_probability']);return np.tile(np.log(p/(1-p)),(n,1))
 if model=='nearest_neighbor':return np.load(H/'models/nearest_neighbor_predictions.npz')[split]
 return np.load(H/'models'/model/'predictions.npz')[split]
def source_raw():
 rr=[]
 for p in sorted((S/'raw').glob('shard*.jsonl')):rr += [json.loads(x) for x in open(p) if x.strip()]
 a=np.load(S/'state_features.npz');splits={str(x):str(y) for x,y in zip(a['state_ids'],a['splits'])};cb=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in read(S/'codebook_eta.csv')])
 out={}
 for r in rr:
  sid=r['state_id'];r['split']=r.get('split',splits[sid])
  if 'mode_id' in r:m=int(r['mode_id'])
  else:
   e=np.asarray(r['eta'],float)
   if np.max(np.abs(e))<1e-12:m=-1
   else:m=int(np.argmin(np.linalg.norm(cb-e,axis=1)))
  out[(sid,m,int(r['future_index']))]=r
 return out
def select_threshold():
 a=features();ids=[str(x) for x in a['state_ids'][a['splits']=='val']];sel=json.load(open(H/'selected_model.json'));cal=json.load(open(H/'calibration.json'));T=cal['temperature'] if cal['used'] else 1.;p=1/(1+np.exp(-np.clip(logits(sel['model'],'val')/T,-30,30)));cm=countmap('val');safe={r['state_id']:r for r in read(S/'safety_val_q32.csv')};raw=source_raw();modes=cfg()['selected_original_mode_ids'];rows=[];summary=[]
 for tau in TAUS:
  suc=dead=brk=abn=0
  for i,sid in enumerate(ids):
   lm=int(np.argmax(p[i]));om=modes[lm];conf=float(p[i,lm]);ab=conf<tau;r=safe[sid] if ab else cm[(sid,lm)];suc+=int(r['successes']);dead+=int(r['deadlock']);abn+=ab
   for fi in range(32):
    ss=raw[(sid,-1,fi)]['success'];pp=raw[(sid,-1 if ab else om,fi)]['success'];brk+=ss and not pp
   rows.append(dict(tau=tau,state_id=sid,sparse_mode_id=lm,original_mode_id=om,confidence=conf,abstained=ab,successes=r['successes'],trials=r['trials'],empirical_Q=r['empirical_Q'],deadlock=r['deadlock'],timeout=r['timeout'],collision=r['collision']))
  summary.append(dict(tau=tau,total_successes=suc,breaks=int(brk),deadlock=dead,abstained_states=int(abn)))
 win=sorted(summary,key=lambda x:(-x['total_successes'],x['breaks'],x['deadlock'],-x['tau']))[0];dump('selected_threshold.json',{**win,'grid':list(TAUS),'selection':'VAL only: successes, breaks, deadlocks, larger tau','selected_model':sel['model'],'temperature_used':cal['used']});write('val_threshold_sweep.csv',rows);print(json.dumps(win,indent=2))
def row_summary(name,rows):
 suc=sum(int(r['successes']) for r in rows);tr=sum(int(r['trials']) for r in rows)
 return dict(model=name,states=len(rows),B63_states=sum(str(r['B63']).lower()=='true' for r in rows),mean_Q64=suc/tr,successes=suc,trials=tr,deadlock=sum(int(r['deadlock']) for r in rows),timeout=sum(int(r['timeout']) for r in rows),collision=sum(int(r['collision']) for r in rows),J_def=float(np.average([float(r['J_def_mean']) for r in rows if r['J_def_mean']!=''],weights=[int(r['successes']) for r in rows if r['J_def_mean']!=''])) if any(r['J_def_mean']!='' and int(r['successes']) for r in rows) else None,episode_length=float(np.average([float(r['episode_length_mean']) for r in rows],weights=[int(r['trials']) for r in rows])))
def test():
 a=features();ids=[str(x) for x in a['state_ids'][a['splits']=='test']];sel=json.load(open(H/'selected_model.json'));cal=json.load(open(H/'calibration.json'));thr=json.load(open(H/'selected_threshold.json'));T=cal['temperature'] if cal['used'] else 1.;modes=cfg()['selected_original_mode_ids'];cm=countmap('test');safe={r['state_id']:r for r in read(S/'safety_test_q64.csv')};p=1/(1+np.exp(-np.clip(logits(sel['model'],'test')/T,-30,30)))
 selector=[];oracle=[];cover=[];selection=[]
 for i,sid in enumerate(ids):
  vals=[cm[(sid,j)] for j in range(len(modes))];qs=np.array([float(r['empirical_Q']) for r in vals]);best=int(np.argmax(qs));lm=int(np.argmax(p[i]));conf=float(p[i,lm]);ab=conf<thr['tau'];chosen=safe[sid] if ab else vals[lm];mode_row=vals[lm];cov=bool(np.any(qs>=63/64))
  selector.append(dict(state_id=sid,sparse_mode_id=lm,original_mode_id=modes[lm],confidence=conf,abstained=ab,selected_mode_Q64=mode_row['empirical_Q'],selected_mode_B63=mode_row['B63'],**{k:chosen[k] for k in ('successes','trials','empirical_Q','B63','deadlock','timeout','collision','J_def_mean','episode_length_mean')}))
  oracle.append(dict(state_id=sid,oracle_sparse_mode=best,oracle_original_mode=modes[best],oracle_Q64=qs[best],oracle_B63=qs[best]>=63/64,coverable=cov,selector_mode_Q64=qs[lm],deployed_Q64=float(chosen['empirical_Q']),mode_regret=qs[best]-qs[lm],deployed_regret=qs[best]-float(chosen['empirical_Q'])))
  cover.append(dict(state_id=sid,coverable=cov,oracle_mode=modes[best],oracle_Q64=qs[best],selector_mode=modes[lm],selector_mode_Q64=qs[lm],selector_mode_B63=qs[lm]>=63/64,abstained=ab,deployed_Q64=chosen['empirical_Q'],fixed_mode=cfg()['best_fixed_original_mode'],fixed_Q64=qs[modes.index(cfg()['best_fixed_original_mode'])],fixed_B63=qs[modes.index(cfg()['best_fixed_original_mode'])]>=63/64))
  selection.append(dict(state_id=sid,sparse_mode_id=lm,original_mode_id=modes[lm],confidence=conf,abstained=ab,selected_mode_B63=qs[lm]>=63/64))
 write('test_selector.csv',selector);write('test_oracle.csv',oracle);write('coverable_state_analysis.csv',cover);write('test_mode_selections.csv',selection)
 # Baseline and network ranking metrics are revealed only after all freezes.
 _,Y,_,sp,_=__import__('train').arrays();yt=Y[sp=='test'];models=['global_prior','linear','nearest_neighbor']+[f'mlp_seed{s}' for s in (17,23,41)];metricrows=[];deployment=[]
 for model in models:
  z=logits(model,'test');mm=metrics(z,yt);metricrows.append(dict(model=model,split='test',**mm));top=np.argmax(z,axis=1);rs=[cm[(sid,int(top[i]))] for i,sid in enumerate(ids)];deployment.append(row_summary(model,rs))
 metricrows.append(dict(model=sel['model']+'_calibrated',split='test',**metrics(logits(sel['model'],'test')/T,yt)));write('test_model_metrics.csv',metricrows);write('test_model_deployment.csv',deployment)
 # Add frozen TEST metrics to the pre-existing TRAIN/VAL summary.
 trainrows=read(H/'training_summary.csv');write('training_summary.csv',trainrows+metricrows)
 fixed_local=modes.index(cfg()['best_fixed_original_mode']);fixed=[cm[(sid,fixed_local)] for sid in ids];ors=[cm[(sid,int(np.argmax([float(cm[(sid,j)]['empirical_Q']) for j in range(len(modes))])))] for sid in ids]
 controller=[row_summary('BEST_FIXED_SPARSE_MODE',fixed),row_summary('SPARSE_SELECTOR',selector),row_summary('SPARSE_ORACLE',ors),row_summary('SAFETY',[safe[sid] for sid in ids])];write('test_controller_comparison.csv',controller)
 cv=[r for r in cover if r['coverable']=='True' or r['coverable'] is True];correct=sum(str(r['selector_mode_B63']).lower()=='true' for r in cv);fixedok=sum(str(r['fixed_B63']).lower()=='true' for r in cv);cnt=Counter(r['original_mode_id'] for r in selection if str(r['abstained']).lower()!='true');n=sum(cnt.values());ent=-sum((v/n)*math.log(v/n) for v in cnt.values()) if n else 0.;stats=[]
 for m in modes:stats.append(dict(metric='mode_count',original_mode_id=m,value=cnt[str(m)] if str(m) in cnt else cnt[m],note='non-abstained TEST selections'))
 stats += [dict(metric='abstentions',original_mode_id='',value=sum(str(r['abstained']).lower()=='true' for r in selection),note='states'),dict(metric='selection_entropy_nats',original_mode_id='',value=ent,note='maximum ln(M)'),dict(metric='normalized_selection_entropy',original_mode_id='',value=ent/math.log(len(modes)) if len(modes)>1 else 0,note='1 means balanced')];write('mode_selection_statistics.csv',stats)
 summary={'test_states':len(ids),'coverable_states':len(cv),'selector_mode_B63_on_coverable':correct/len(cv) if cv else 0.,'fixed_B63_on_coverable':fixedok/len(cv) if cv else 0.,'selector_mean_mode_regret':float(np.mean([float(r['mode_regret']) for r in oracle])),'selector_mean_deployed_regret':float(np.mean([float(r['deployed_regret']) for r in oracle])),'selection_counts':{str(k):v for k,v in cnt.items()},'selection_entropy_nats':ent,'normalized_entropy':ent/math.log(len(modes)) if len(modes)>1 else 0.,'controllers':controller,'fresh_trigger':bool(len(cv)>0 and correct/len(cv)>=.80 and np.mean([float(r['mode_regret']) for r in oracle])<=.10 and controller[1]['mean_Q64']>=controller[2]['mean_Q64']-.10)};dump('test_summary.json',summary);print(json.dumps(summary,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['threshold','test']);a=ap.parse_args();select_threshold() if a.stage=='threshold' else test()
if __name__=='__main__':main()
