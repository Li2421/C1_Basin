#!/usr/bin/env python3
"""Prepare and train the frozen CENTER-vs-MARGIN comparison."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1'
DIRECT=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
PREV=ROOT/'diagnostics/orthoflow3_large_margin_ball_transfer_v1'
PILOT=ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1'
AFF=np.array([.875,0.,.375],np.float64); SCALE=np.array([.75,1.,.75],np.float64)
GAMMA=.80; EPS=1e-8; SEEDS=(17,23,41)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def rows(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rr,fields):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)

class G(nn.Module):
 @nn.compact
 def __call__(self,x):
  x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)

def halfspaces():
 rr=rows(HERE/'eligible_new_anchor_source/ebridge_halfspaces.csv')
 return np.array([[float(x['n1']),float(x['n2']),float(x['n3']),float(x['b'])] for x in rr])

def metrics(pred,c,r,eq):
 dist=np.linalg.norm(pred-c,axis=1);rho=dist/r
 viol=np.max(pred@eq[:,:3].T+eq[:,3],axis=1)>1e-9
 if len(pred)>1:
  d=np.linalg.norm(pred[:,None,:]-pred[None,:,:],axis=-1);pair=float(d[np.triu_indices(len(pred),1)].mean())
 else:pair=0.
 return {'center_mse':float(np.mean(dist**2)),'mean_rho':float(rho.mean()),'median_rho':float(np.median(rho)),
  'inside_ball_fraction':float(np.mean(rho<=1.)),'retained_fraction':float(np.mean(rho<=GAMMA)),
  'margin_loss':float(np.mean(np.maximum(0,rho-GAMMA)**2)),'zero_loss_fraction':float(np.mean(rho<=GAMMA+1e-12)),
  'eta_norm_mean':float(np.mean(np.linalg.norm(pred,axis=1))),'ebridge_violations':int(viol.sum()),
  'output_variance_1':float(np.var(pred[:,0])),'output_variance_2':float(np.var(pred[:,1])),'output_variance_3':float(np.var(pred[:,2])),
  'mean_pairwise_eta_distance':pair}

def prepare():
 frozen=HERE/'verified_ball_dataset.csv';before=sha(frozen);rr=rows(frozen)
 assert len(rr)==44 and {s:sum(x['split']==s for x in rr) for s in ('train','val','test')}=={'train':25,'val':10,'test':9}
 base=np.load(DIRECT/'conditioning_features.npz')['features']
 prev=np.load(PREV/'neighbor_features.npz')['features']
 new=np.load(HERE/'new_anchor_transfers/neighbor_features.npz')['features']
 feat=[]
 for x in rr:
  if x['label_kind']=='anchor':h=base[int(x['feature_index'])]
  elif str(x['provenance'])==str(PREV):h=prev[int(x['feature_index'])]
  else:h=new[int(x['feature_index'])]
  got=hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest()
  if got!=x['feature_sha256']:raise RuntimeError(('feature hash',x['state_id'],got,x['feature_sha256']))
  feat.append(np.asarray(h,dtype=np.float64))
 feat=np.stack(feat);np.savez_compressed(HERE/'learning_features.npz',features=feat,state_ids=np.array([x['state_id'] for x in rr]))
 tr=np.array([x['split']=='train' for x in rr]);hm=feat[tr].mean(0);hs=feat[tr].std(0);hs=np.where(hs<1e-8,1.,hs)
 centers=(np.array([[float(x['c1']),float(x['c2']),float(x['c3'])] for x in rr])-AFF)/SCALE
 radii=np.array([float(x['r_ball']) for x in rr])
 np.savez_compressed(HERE/'learning_arrays.npz',x=((feat-hm)/hs).astype(np.float32),centers=centers.astype(np.float32),radii=radii.astype(np.float32),splits=np.array([x['split'] for x in rr]),state_ids=np.array([x['state_id'] for x in rr]))
 dump(HERE/'normalization.json',{'h_mean':hm.tolist(),'h_std':hs.tolist(),'fit_train_only':True,'eta_affine_center':AFF.tolist(),'eta_scale':SCALE.tolist(),'eta_normalization':'(eta-affine_center)/eta_scale','verified_ball_geometry':'normalized eta coordinates','verified_dataset_sha256':before})
 # Exact common intersection: all centers coincide, so their common center is feasible for every retained region.
 tc=centers[tr];trad=radii[tr]*GAMMA;common=tc[0];inside=np.linalg.norm(tc-common,axis=1)<=trad+1e-12
 assert inside.all()
 dump(HERE/'train_ball_overlap_analysis.json',{'train_balls':int(tr.sum()),'retained_fraction':GAMMA,'method':'exact shared-center certificate plus deterministic center-candidate enumeration','maximum_common_count':int(inside.sum()),'maximum_common_fraction':float(inside.mean()),'common_eta_normalized':common.tolist(),'common_eta_physical':(AFF+SCALE*common).tolist(),'all_train_centers_identical':bool(np.max(np.abs(tc-tc[0]))<1e-12),'minimum_retained_radius':float(trad.min()),'interpretation':'the objective admits an exact state-independent zero-loss prediction'})
 families={s:sorted({x['anchor_family'] for x in rr if x['split']==s}) for s in ('train','val','test')}
 dump(HERE/'frozen_dataset_integrity.json',{'dataset_sha256_before':before,'dataset_sha256_after':sha(frozen),'unchanged':before==sha(frozen),'rows':len(rr),'splits':{s:sum(x['split']==s for x in rr) for s in ('train','val','test')},'independent_source_families':{s:len(families[s]) for s in families},'source_family_leakage':bool(set(families['train'])&set(families['val']) or set(families['train'])&set(families['test']) or set(families['val'])&set(families['test'])),'centers_or_radii_modified':False,'new_ball_labels_generated':0,'orthoflow3_sha256':sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py')})
 dump(HERE/'dataset_semantics_summary.json',{'labels':44,'splits':{'train':25,'val':10,'test':9},'true_t0_labels':0,'intermediate_labels':44,'evaluation_regimes':{'heldout_test':'intermediate-state source-family-held-out generalization','fresh_wide':'out-of-distribution t0 extrapolation'},'label':'empirically verified conservative normalized-eta inner ball','not_full_success_basin':True})
 model=G();params=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214),jnp.float32));n=sum(np.asarray(z).size for z in jax.tree_util.tree_leaves(params))
 dump(HERE/'common_model_config.json',{'architecture':'214->128->128->3','activation':'SiLU','output':'unconstrained normalized eta; physical eta=affine_center+eta_scale*output','parameter_count':int(n),'optimizer':'AdamW','learning_rate':1e-3,'weight_decay':1e-5,'batch_size':32,'max_epochs':3000,'early_stop_patience':180,'improvement_tolerance':1e-8,'seeds':list(SEEDS),'gamma':GAMMA,'output_domain_audit':{'old_eta1_ge_0.5_constraint':False,'zero_snap':False,'old_active_box_clip':False,'posthoc_repair':False,'E_bridge_violations_recorded':True}})
 print(json.dumps({'prepared':len(rr),'dataset_sha256':before,'parameter_count':n,'common_intersection':int(inside.sum())},indent=2))

def train(arm,seed):
 a=np.load(HERE/'learning_arrays.npz');x=a['x'];c=a['centers'];r=a['radii'];sp=a['splits'];tr=sp=='train';va=sp=='val';eq=halfspaces()
 model=G();p=model.init(jax.random.PRNGKey(seed),jnp.asarray(x[:1]));opt=optax.adamw(1e-3,weight_decay=1e-5);state=opt.init(p);rng=np.random.default_rng(seed)
 def loss(par,xb,cb,rb):
  pred=model.apply(par,xb);dist=jnp.linalg.norm(pred-cb,axis=1)
  return jnp.mean(dist**2) if arm=='center' else jnp.mean(jnp.maximum(0.,dist/(rb+EPS)-GAMMA)**2)
 @jax.jit
 def step(par,st,xb,cb,rb):
  v,g=jax.value_and_grad(loss)(par,xb,cb,rb);u,st=opt.update(g,st,par);return optax.apply_updates(par,u),st,v
 best=None;bestv=math.inf;beste=-1;wait=0;hist=[];started=time.monotonic()
 for epoch in range(3000):
  order=np.where(tr)[0];order=order[rng.permutation(len(order))]
  for begin in range(0,len(order),32):
   ii=order[begin:begin+32];p,state,_=step(p,state,jnp.asarray(x[ii]),jnp.asarray(c[ii]),jnp.asarray(r[ii]))
  pt=np.asarray(model.apply(p,jnp.asarray(x[tr])));pv=np.asarray(model.apply(p,jnp.asarray(x[va])))
  mt=metrics(pt,c[tr],r[tr],eq);mv=metrics(pv,c[va],r[va],eq);v=mv['center_mse'] if arm=='center' else mv['margin_loss']
  rec={'arm':arm,'seed':seed,'epoch':epoch,**{f'train_{k}':v0 for k,v0 in mt.items()},**{f'val_{k}':v0 for k,v0 in mv.items()}};hist.append(rec)
  # Earliest epoch within the frozen 1e-8 tolerance is retained.
  if v < bestv-1e-8:
   bestv=v;beste=epoch;best=jax.tree_util.tree_map(lambda z:np.asarray(z),p);wait=0
  else:wait+=1
  if wait>=180:break
 out=HERE/f'g_{arm}'/'runs';out.mkdir(parents=True,exist_ok=True);ck=out/f'{arm}_seed{seed}.msgpack';ck.write_bytes(serialization.to_bytes(best))
 write_csv(out/f'{arm}_seed{seed}_history.csv',hist,list(hist[0]))
 predtr=np.asarray(model.apply(best,jnp.asarray(x[tr])));predva=np.asarray(model.apply(best,jnp.asarray(x[va])))
 summary={'arm':arm,'seed':seed,'best_epoch':beste,'best_val_loss':bestv,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'training_seconds':time.monotonic()-started,'epochs':len(hist),'train_metrics':metrics(predtr,c[tr],r[tr],eq),'val_metrics':metrics(predva,c[va],r[va],eq),'devices':[str(z) for z in jax.devices()]}
 dump(out/f'{arm}_seed{seed}_summary.json',summary);print(json.dumps(summary,indent=2))

def aggregate():
 for arm in ('center','margin'):
  d=HERE/f'g_{arm}';run=d/'runs';ss=[json.load(open(run/f'{arm}_seed{s}_summary.json')) for s in SEEDS]
  hh=[]
  for s in SEEDS:hh+=rows(run/f'{arm}_seed{s}_history.csv')
  write_csv(d/'training_history.csv',hh,list(hh[0]));dump(d/'seed_results.json',{'arm':arm,'seeds':ss})
  dump(d/'retained_checkpoints.json',{'selection':'minimum VAL center MSE' if arm=='center' else 'minimum VAL margin-set loss; earliest within 1e-8','checkpoints':[{k:x[k] for k in ('seed','best_epoch','best_val_loss','checkpoint','checkpoint_sha256')} for x in ss]})
 print(json.dumps({'aggregated':True}))

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
 sub.add_parser('prepare');t=sub.add_parser('train');t.add_argument('--arm',choices=['center','margin'],required=True);t.add_argument('--seed',type=int,choices=SEEDS,required=True);sub.add_parser('aggregate')
 a=ap.parse_args()
 if a.cmd=='prepare':prepare()
 elif a.cmd=='train':train(a.arm,a.seed)
 else:aggregate()
if __name__=='__main__':main()
