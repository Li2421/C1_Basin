#!/usr/bin/env python3
"""Freeze the completed 24/8/8 dataset and train G_POINT_T0."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
import optax

ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1';PACT=ROOT/'diagnostics/orthoflow3_t0_pact_training_readiness_v1';T0=ROOT/'diagnostics/orthoflow3_t0_basin_structure_v1';COMP=ROOT/'diagnostics/orthoflow3_t0_basin_completion_v1'
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75]);SEEDS=(17,23,41)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def read(p):return list(csv.DictReader(open(p)))
def write(p,rr,fields=None):
 fields=fields or (list(rr[0]) if rr else ['state_id'])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
class G(nn.Module):
 @nn.compact
 def __call__(self,x):
  x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)

def prepare():
 oldrows=read(PACT/'point_dataset_candidates.csv'); oldstates={x['state_id']:x for x in json.load(open(COMP/'frozen_8state_manifest.json'))['attempted_states']}; oldfeat=np.load(T0/'synthetic_qdir/conditioning_features.npz')['features'];balls={x['state_id']:x for x in read(COMP/'completed_t0_balls.csv')}
 newstates={x['state_id']:x for x in json.load(open(HERE/'eligible_state_manifest.json'))['selected_states']};newfeat=np.load(HERE/'conditioning_features.npz')['features'];new=[]
 for p in sorted((HERE/'state_runs').glob('*/selected_target.json')):
  x=json.load(open(p))
  if x['status']=='POINT_USABLE':new.append(x)
 counts={s:sum(x['split']==s for x in oldrows) for s in ('train','val','test')};need={s:{'train':24,'val':8,'test':8}[s]-counts[s] for s in counts}
 selected_new=[]
 for s in ('train','val','test'):
  cand=sorted((x for x in new if x['split']==s),key=lambda x:int(x['split_rank']))
  if len(cand)<need[s]:raise RuntimeError(f'dataset gate {s}: need {need[s]} have {len(cand)}')
  selected_new+=cand[:need[s]]
 rows=[];features=[];states=[]
 for x in oldrows:
  sid=x['state_id'];st=dict(oldstates[sid]);st['split']=x['split'];h=np.asarray(oldfeat[int(st['feature_index'])],np.float64);r=float(balls[sid]['r_ball'])
  tag='ROBUST_INTERIOR_STRONG' if r>=.025-1e-12 else 'ROBUST_POINT_ONLY'
  rows.append({'state_id':sid,'source_group':x['source_group'],'split':x['split'],'target_eta1':x['target_eta1'],'target_eta2':x['target_eta2'],'target_eta3':x['target_eta3'],'target_Q64':x['center_Q64'],'quality_tag':tag,'target_provenance':'existing_independently_verified_robust_center','local_robust_count':'6' if tag=='ROBUST_INTERIOR_STRONG' else '','valid_local_perturbations':'6' if tag=='ROBUST_INTERIOR_STRONG' else '','feature_sha256':hashlib.sha256(h.tobytes()).hexdigest()});features.append(h);st['dataset_index']=len(rows)-1;states.append(st)
 for x in selected_new:
  sid=x['state_id'];st=dict(newstates[sid]);h=np.asarray(newfeat[int(st['feature_index'])],np.float64)
  if hashlib.sha256(h.tobytes()).hexdigest()!=x['feature_sha256']:raise RuntimeError(('feature hash',sid))
  rows.append({'state_id':sid,'source_group':x['source_group'],'split':x['split'],'target_eta1':x['eta1'],'target_eta2':x['eta2'],'target_eta3':x['eta3'],'target_Q64':x['Q64'],'quality_tag':x['quality_tag'],'target_provenance':'frozen_capped_sobol_search','local_robust_count':x['local_robust_count'],'valid_local_perturbations':x['valid_local_perturbations'],'feature_sha256':x['feature_sha256']});features.append(h);st['dataset_index']=len(rows)-1;states.append(st)
 got={s:sum(x['split']==s for x in rows) for s in ('train','val','test')}
 if got!={'train':24,'val':8,'test':8}:raise RuntimeError(got)
 fam={s:{x['source_group'] for x in rows if x['split']==s} for s in got};leak=(fam['train']&fam['val'])|(fam['train']&fam['test'])|(fam['val']&fam['test'])
 if leak:raise RuntimeError(('source leakage',leak))
 feat=np.stack(features);target=np.array([[float(x[f'target_eta{i}']) for i in (1,2,3)] for x in rows]);split=np.array([x['split'] for x in rows]);tr=split=='train';hm=feat[tr].mean(0);hs=feat[tr].std(0);hs=np.where(hs<1e-8,1.,hs);tn=(target-AFF)/SCALE
 np.savez_compressed(HERE/'point_learning_arrays.npz',x=((feat-hm)/hs).astype(np.float32),target=tn.astype(np.float32),physical_target=target.astype(np.float64),splits=split,state_ids=np.array([x['state_id'] for x in rows]),features=feat)
 write(HERE/'selected_eta_targets.csv',rows);dump(HERE/'final_source_split.json',{'counts':got,'source_groups':{s:sorted(fam[s]) for s in fam},'source_leakage':False,'states':states});dump(HERE/'normalization.json',{'h_mean':hm.tolist(),'h_std':hs.tolist(),'fit_train_only':True,'eta_affine_center':AFF.tolist(),'eta_scale':SCALE.tolist()})
 manifest={'rows':40,'split_counts':got,'independent_source_groups':{s:len(fam[s]) for s in fam},'source_leakage':False,'true_t0':40,'feature_dim':214,'selected_targets_sha256':sha(HERE/'selected_eta_targets.csv'),'arrays_sha256':sha(HERE/'point_learning_arrays.npz'),'normalization_sha256':sha(HERE/'normalization.json'),'frozen_before_training':True};dump(HERE/'dataset_manifest.json',manifest)
 model=G();p=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));n=sum(np.asarray(z).size for z in jax.tree_util.tree_leaves(p));dump(HERE/'common_model_config.json',{'architecture':[214,128,128,3],'activation':'SiLU','parameter_count':int(n),'optimizer':'AdamW','learning_rate':1e-3,'weight_decay':1e-5,'batch_size':24,'max_epochs':3000,'early_stop_patience':180,'seeds':list(SEEDS),'loss':'normalized eta MSE','output':'unconstrained normalized eta'})
 xn=((feat-hm)/hs).astype(np.float64);nn={}
 for s in ('val','test'):
  qi=np.where(split==s)[0];ti=np.where(tr)[0];pred=[];nearest=[]
  for q in qi:
   d=np.linalg.norm(xn[ti]-xn[q],axis=1);j=int(np.argmin(d));pred.append(tn[ti[j]]);nearest.append({'state_id':rows[q]['state_id'],'nearest_train_state_id':rows[ti[j]]['state_id'],'h_distance':float(d[j])})
  nn[s]={'metrics':metrics(np.asarray(pred),tn[qi]),'pairs':nearest}
 dump(HERE/'nearest_neighbor_baseline.json',nn)
 print(json.dumps(manifest,indent=2))

def metrics(pred,y):
 d=pred-y
 return {'eta_l2_mean':float(np.linalg.norm(d,axis=1).mean()),'eta_l2_median':float(np.median(np.linalg.norm(d,axis=1))),'mse':float(np.mean(np.sum(d*d,axis=1))),'mae_dim1':float(np.abs(d[:,0]).mean()),'mae_dim2':float(np.abs(d[:,1]).mean()),'mae_dim3':float(np.abs(d[:,2]).mean()),'target_variance_dim1':float(np.var(y[:,0])),'target_variance_dim2':float(np.var(y[:,1])),'target_variance_dim3':float(np.var(y[:,2])),'prediction_variance_dim1':float(np.var(pred[:,0])),'prediction_variance_dim2':float(np.var(pred[:,1])),'prediction_variance_dim3':float(np.var(pred[:,2]))}

def train(seed):
 a=np.load(HERE/'point_learning_arrays.npz');x=a['x'];y=a['target'];sp=a['splits'];tr=sp=='train';va=sp=='val';model=G();p=model.init(jax.random.PRNGKey(seed),jnp.asarray(x[:1]));opt=optax.adamw(1e-3,weight_decay=1e-5);ost=opt.init(p);rng=np.random.default_rng(seed)
 @jax.jit
 def step(p,ost,xb,yb):
  def loss(pp):return jnp.mean(jnp.sum((model.apply(pp,xb)-yb)**2,axis=1))
  v,g=jax.value_and_grad(loss)(p);u,ost=opt.update(g,ost,p);return optax.apply_updates(p,u),ost,v
 best=None;bestv=math.inf;beste=-1;wait=0;hist=[];start=time.time()
 for epoch in range(3000):
  ii=np.where(tr)[0];ii=ii[rng.permutation(len(ii))]
  p,ost,_=step(p,ost,jnp.asarray(x[ii]),jnp.asarray(y[ii]));pt=np.asarray(model.apply(p,jnp.asarray(x[tr])));pv=np.asarray(model.apply(p,jnp.asarray(x[va])));mt=metrics(pt,y[tr]);mv=metrics(pv,y[va]);v=mv['mse'];hist.append({'seed':seed,'epoch':epoch,**{f'train_{k}':z for k,z in mt.items()},**{f'val_{k}':z for k,z in mv.items()}})
  if v<bestv-1e-8:bestv=v;beste=epoch;best=jax.tree_util.tree_map(lambda z:np.asarray(z),p);wait=0
  else:wait+=1
  if wait>=180:break
 d=HERE/f'seed{seed}';d.mkdir(exist_ok=True);ck=d/'checkpoint.msgpack';ck.write_bytes(serialization.to_bytes(best));write(d/'training_history.csv',hist);pred={s:metrics(np.asarray(model.apply(best,jnp.asarray(x[sp==s]))),y[sp==s]) for s in ('train','val','test')};summary={'seed':seed,'best_epoch':beste,'best_val_mse':bestv,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'epochs':len(hist),'training_seconds':time.time()-start,'metrics':pred};dump(d/'summary.json',summary);print(json.dumps(summary,indent=2))

def aggregate():
 ss=[json.load(open(HERE/f'seed{s}/summary.json')) for s in SEEDS];write(HERE/'training_summary.csv',[{'seed':x['seed'],'best_epoch':x['best_epoch'],'best_val_mse':x['best_val_mse'],'epochs':x['epochs'],'training_seconds':x['training_seconds'],'checkpoint':x['checkpoint'],'checkpoint_sha256':x['checkpoint_sha256']} for x in ss]);dump(HERE/'retained_checkpoints.json',{'selection':'minimum VAL normalized eta MSE per seed before true closed-loop VAL selection','seeds':ss});print(json.dumps({'seeds':len(ss)}))

def main():
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True);sp.add_parser('prepare');t=sp.add_parser('train');t.add_argument('--seed',type=int,choices=SEEDS,required=True);sp.add_parser('aggregate');a=ap.parse_args()
 if a.cmd=='prepare':prepare()
 elif a.cmd=='train':train(a.seed)
 else:aggregate()
if __name__=='__main__':main()
