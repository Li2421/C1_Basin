#!/usr/bin/env python3
"""Integrity-check frozen assets, train 15 Q-guided actors, freeze VAL plans."""
from __future__ import annotations
import csv,hashlib,json,shutil,time
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np,optax
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';BASE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';QDIR=ROOT/'diagnostics/orthoflow3_q_learnability_v2'
LOW=np.array([0.,-.5,0.],np.float32);HIGH=np.array([1.25,.5,.75],np.float32);SCALE=np.array([.75,1.,.75],np.float32);LAMBDAS=(.01,.03,.10,.30,1.);SEEDS=(17,23,41)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def treehash(x):
 h=hashlib.sha256()
 for a in jax.tree_util.tree_leaves(x):h.update(np.asarray(a).tobytes())
 return h.hexdigest()
class G(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return jnp.asarray(LOW)+nn.sigmoid(nn.Dense(3)(x))*jnp.asarray(HIGH-LOW)
class Q(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(256)(x));x=nn.silu(nn.Dense(256)(x));return nn.Dense(1)(x).squeeze(-1)
def main():
 HERE.mkdir(parents=True,exist_ok=True);(HERE/'training_curves').mkdir(exist_ok=True)
 for name in ('eligible_state_manifest.json','conditioning_features.npz'):shutil.copy2(BASE/name,HERE/name)
 bsel=json.loads((BASE/'selected_checkpoint.json').read_text());qsel=json.loads((QDIR/'selected_checkpoint.json').read_text());qf=json.loads((QDIR/'frozen_q_manifest.json').read_text());tau=float(qf['zero_feasibility_threshold'])
 if sha(bsel['checkpoint'])!='bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7':raise RuntimeError('baseline checkpoint mismatch')
 if sha(qsel['checkpoint'])!=qf['checkpoint_sha256']:raise RuntimeError('Q checkpoint mismatch')
 bn=json.loads((BASE/'normalization.json').read_text());qn=json.loads((QDIR/'normalization.json').read_text());hm=np.asarray(bn['h_mean']);hs=np.asarray(bn['h_std']);qh=np.asarray(qn['h_mean']);qhs=np.asarray(qn['h_std']);ec=np.asarray(qn['eta_center']);es=np.asarray(qn['eta_scale'])
 states=json.loads((BASE/'eligible_state_manifest.json').read_text())['selected_states'];sm={r['state_id']:r for r in states};h=np.asarray(np.load(BASE/'conditioning_features.npz')['features'],np.float32);targets=[r for r in csv.DictReader(open(BASE/'robust_lowj_targets.csv')) if r['target_status']!='TARGET_UNRESOLVED']
 def data(sp):
  rr=[r for r in targets if r['split']==sp];x=np.stack([((h[sm[r['state_id']]['feature_index']]-hm)/hs) for r in rr]).astype(np.float32);xq=np.stack([((h[sm[r['state_id']]['feature_index']]-qh)/qhs) for r in rr]).astype(np.float32);y=np.array([[r['eta1'],r['eta2'],r['eta3']] for r in rr],np.float32);return rr,x,xq,y
 ds={s:data(s) for s in ('train','val','test')}
 # Exact baseline prediction replay on archived TEST rows.
 gm=G();gt=gm.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));bp=serialization.from_bytes(gt,Path(bsel['checkpoint']).read_bytes());arch=json.loads((BASE/'frozen_direct_eta_manifest.json').read_text())['test_predictions'];ar={r['state_id']:np.asarray(json.loads(r['pred_eta'])) for r in arch};pred=np.asarray(gm.apply(bp,jnp.asarray(ds['test'][1])));maxerr=max(float(np.max(np.abs(p-ar[r['state_id']]))) for r,p in zip(ds['test'][0],pred))
 integrity={'status':'PASS' if maxerr<=1e-7 else 'FAIL','baseline_checkpoint':bsel,'prediction_replay_max_abs_error':maxerr,'target_sha256':sha(BASE/'robust_lowj_targets.csv'),'normalization_sha256':sha(BASE/'normalization.json'),'features_sha256':sha(BASE/'conditioning_features.npz'),'architecture':'214->128->128->3 SiLU; sigmoid-affine output','output_low':LOW.tolist(),'output_high':HIGH.tolist(),'eta_loss_scale':SCALE.tolist()};dump(HERE/'baseline_integrity_check.json',integrity)
 if integrity['status']!='PASS':raise RuntimeError(integrity)
 qm=Q();qt=qm.init(jax.random.PRNGKey(0),jnp.zeros((1,217)));qp=serialization.from_bytes(qt,Path(qsel['checkpoint']).read_bytes());qbefore=treehash(qp)
 dump(HERE/'frozen_q_reference.json',{'checkpoint':qsel,'frozen_manifest':qf,'normalization_sha256':sha(QDIR/'normalization.json'),'conditioning_semantics_sha256':sha(QDIR/'q_conditioning_semantics.md'),'tau':tau,'tau_provenance':qf['threshold_selected_on']})
 dump(HERE/'lambda_grid.json',{'lambda_Q':list(LAMBDAS),'seeds':list(SEEDS),'archived_control_lambda':0,'tau':tau})
 dump(HERE/'training_config.json',{'actor_architecture':'214->128->128->3','activation':'SiLU','output_mapping':{'low':LOW.tolist(),'high':HIGH.tolist()},'loss':'normalized eta MSE + lambda*relu(tau-frozen_Q)^2','eta_scales':SCALE.tolist(),'optimizer':'AdamW lr=1e-3 weight_decay=1e-5','batch_size':32,'max_epochs':3000,'early_stop_patience':180,'checkpoint_metric':'VAL normalized eta MSE only','sampling':'unweighted frozen rows; no oversampling','tau':tau})
 results=[];started=time.monotonic()
 for lam in LAMBDAS:
  for seed in SEEDS:
   p=gm.init(jax.random.PRNGKey(seed),jnp.asarray(ds['train'][1][:1]));opt=optax.adamw(1e-3,weight_decay=1e-5);os=opt.init(p);best=None;bv=1e99;be=0;wait=0;hist=[];rng=np.random.default_rng(seed)
   def metrics(par,x,xq,y):
    ep=gm.apply(par,x);le=jnp.mean(jnp.sum(((ep-y)/jnp.asarray(SCALE))**2,axis=1));qx=jnp.concatenate([xq,(ep-jnp.asarray(ec))/jnp.asarray(es)],axis=1);qq=jax.nn.sigmoid(qm.apply(qp,qx));lq=jnp.mean(jnp.maximum(tau-qq,0.)**2);return le,lq,jnp.mean(qq),jnp.mean(qq<tau),ep
   @jax.jit
   def step(par,ost,x,xq,y):
    def objective(pp):
     le,lq,_,_,_=metrics(pp,x,xq,y);return le+lam*lq,(le,lq)
    (v,parts),gr=jax.value_and_grad(objective,has_aux=True)(par);up,ost=opt.update(gr,ost,par);return optax.apply_updates(par,up),ost,v,parts
   for epoch in range(3000):
    order=rng.permutation(len(ds['train'][1]));vals=[]
    for b in range(0,len(order),32):
     ii=order[b:b+32];p,os,v,parts=step(p,os,jnp.asarray(ds['train'][1][ii]),jnp.asarray(ds['train'][2][ii]),jnp.asarray(ds['train'][3][ii]));vals.append(float(v))
    vle,vlq,vq,vbelow,ve=metrics(p,jnp.asarray(ds['val'][1]),jnp.asarray(ds['val'][2]),jnp.asarray(ds['val'][3]));tle,tlq,tq,tbelow,te=metrics(p,jnp.asarray(ds['train'][1]),jnp.asarray(ds['train'][2]),jnp.asarray(ds['train'][3]));vle=float(vle);hist.append({'epoch':epoch,'train_total':float(np.mean(vals)),'train_L_eta':float(tle),'train_L_Q':float(tlq),'val_L_eta':vle,'val_L_Q':float(vlq),'train_mean_q':float(tq),'val_mean_q':float(vq),'train_fraction_q_below_tau':float(tbelow),'val_fraction_q_below_tau':float(vbelow)})
    if vle<bv-1e-8:bv=vle;be=epoch;best=jax.tree_util.tree_map(lambda z:np.asarray(z),p);wait=0
    else:wait+=1
    if wait>=180:break
   actor=f'L{lam:.2f}_S{seed}';path=HERE/f'actor_{actor}.msgpack';path.write_bytes(serialization.to_bytes(best));tr=metrics(best,jnp.asarray(ds['train'][1]),jnp.asarray(ds['train'][2]),jnp.asarray(ds['train'][3]));va=metrics(best,jnp.asarray(ds['val'][1]),jnp.asarray(ds['val'][2]),jnp.asarray(ds['val'][3]));etas=np.asarray(va[4]);row={'actor_id':actor,'lambda_Q':lam,'seed':seed,'best_epoch':be,'val_eta_mse':float(va[0]),'train_eta_mse':float(tr[0]),'train_L_Q':float(tr[1]),'val_L_Q':float(va[1]),'train_mean_q':float(tr[2]),'val_mean_q':float(va[2]),'train_fraction_q_below_tau':float(tr[3]),'val_fraction_q_below_tau':float(va[3]),'val_eta_norm_mean':float(np.linalg.norm(etas,axis=1).mean()),'val_eta_norm_max':float(np.linalg.norm(etas,axis=1).max()),'val_boundary_saturation_count':int(np.sum((etas-LOW<1e-4)|(HIGH-etas<1e-4))),'checkpoint':str(path),'checkpoint_sha256':sha(path)};results.append(row)
   with (HERE/'training_curves'/f'{actor}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
 with (HERE/'all_training_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 qafter=treehash(qp);dump(HERE/'q_freeze_integrity.json',{'status':'PASS' if qbefore==qafter else 'FAIL','checkpoint_sha256_before':sha(qsel['checkpoint']),'checkpoint_sha256_after':sha(qsel['checkpoint']),'in_memory_tree_hash_before':qbefore,'in_memory_tree_hash_after':qafter,'optimizer_state_for_Q':False,'mutable_Q_statistics':False})
 if qbefore!=qafter:raise RuntimeError('Q changed')
 # Freeze matched Stage-1 predictions.
 tasks=[]
 for run in results:
  pp=serialization.from_bytes(gt,Path(run['checkpoint']).read_bytes());etas=np.asarray(gm.apply(pp,jnp.asarray(ds['val'][1])))
  for r,eta in zip(ds['val'][0],etas):
   for fi in range(8):tasks.append({'task_id':f"val8__{run['actor_id']}__{r['state_id']}__f{fi:02d}",'candidate_id':f"{r['state_id']}__{run['actor_id']}",'state_id':r['state_id'],'split':'val','probe_id':run['actor_id'],'actor_id':run['actor_id'],'lambda_Q':run['lambda_Q'],'seed':run['seed'],'eta':eta.tolist(),'future_index':fi})
 dump(HERE/'validation_stage1_plan.json',{'schema':'q_guided_actor_val_stage1_v1','orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','future_root_seed':2026092715,'tasks':tasks,'matched_future_indices':list(range(8)),'training_seconds':time.monotonic()-started})
 print(json.dumps({'runs':len(results),'stage1_tasks':len(tasks),'training_seconds':time.monotonic()-started,'tau':tau},indent=2))
if __name__=='__main__':main()
