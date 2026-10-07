#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,math,time
from collections import Counter
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
H=Path(__file__).parent;SEEDS=(17,23,41);TAUS=(.50,.60,.70,.80,.90,.95)
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(rows[0]);
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
class Linear(nn.Module):
 @nn.compact
 def __call__(self,x):return nn.Dense(12,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x)
class MLP(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(64)(x));x=nn.silu(nn.Dense(64)(x));return nn.Dense(12)(x)
def arrays():
 a=np.load(H/'state_features.npz');ids=a['state_ids'];sp=a['splits'];idx={str(s):i for i,s in enumerate(ids)};Y=np.zeros((len(ids),12));N=np.zeros_like(Y)
 for fn in ('train_mode_counts.csv','val_mode_counts.csv','test_mode_q64.csv'):
  for r in read(H/fn):i=idx[r['state_id']];m=int(r['mode_id']);Y[i,m]=float(r['successes'])/float(r['trials']);N[i,m]=float(r['trials'])
 assert np.all(N>0);return a['x'].astype(np.float32),Y.astype(np.float32),N.astype(np.float32),sp,ids
def metrics(logits,y):
 p=1/(1+np.exp(-np.clip(logits,-30,30)));eps=1e-7;nll=float(np.mean(-(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))));brier=float(np.mean((p-y)**2));rank=[]
 for a,b in zip(p,y):
  if np.std(a)>1e-12 and np.std(b)>1e-12:rank.append(float(np.corrcoef(rankdata(a),rankdata(b))[0,1]))
 top=np.argmax(p,1);t3=np.argsort(p,axis=1)[:,-3:]
 return {'nll':nll,'brier':brier,'within_state_rank_correlation':float(np.mean(rank)) if rank else None,'top1_empirical_Q':float(np.mean(y[np.arange(len(y)),top])),'top3_empirical_Q':float(np.mean([y[i,t3[i]].max() for i in range(len(y))]))}
def fit(kind,seed):
 x,y,n,sp,ids=arrays();tr=np.flatnonzero(sp=='train');va=np.flatnonzero(sp=='val');model=Linear() if kind=='linear' else MLP();p=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,x.shape[1])));opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(1e-3,weight_decay=1e-4));ost=opt.init(p);rng=np.random.default_rng(seed);best=None;bv=math.inf;be=0;wait=0;hist=[];start=time.time()
 @jax.jit
 def step(p,ost,xb,yb):
  def loss(pp):return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(pp,xb),yb))
  v,g=jax.value_and_grad(loss)(p);u,ost=opt.update(g,ost,p);return optax.apply_updates(p,u),ost,v
 maxep=1600 if kind=='linear' else 2400
 for ep in range(maxep):
  for j in range(0,len(tr),16):
   q=rng.permutation(tr) if j==0 else q;ii=q[j:j+16];p,ost,_=step(p,ost,jnp.asarray(x[ii]),jnp.asarray(y[ii]))
  lt=np.asarray(model.apply(p,jnp.asarray(x[tr])));lv=np.asarray(model.apply(p,jnp.asarray(x[va])));mt=metrics(lt,y[tr]);mv=metrics(lv,y[va]);hist.append({'epoch':ep,'train_nll':mt['nll'],'val_nll':mv['nll'],'train_brier':mt['brier'],'val_brier':mv['brier'],'train_top1_Q':mt['top1_empirical_Q'],'val_top1_Q':mv['top1_empirical_Q']})
  if mv['nll']<bv-1e-7:best=jax.tree_util.tree_map(np.asarray,p);bv=mv['nll'];be=ep;wait=0
  else:wait+=1
  if wait>=180:break
 name='linear' if kind=='linear' else f'mlp_seed{seed}';d=H/name;d.mkdir(exist_ok=True);ck=d/'checkpoint.msgpack';ck.write_bytes(serialization.to_bytes(best));write(f'{name}/training_history.csv',hist);pred={s:np.asarray(model.apply(best,jnp.asarray(x[sp==s]))) for s in ('train','val')};np.savez_compressed(d/'predictions_train_val.npz',**pred);sm={'model':name,'kind':kind,'seed':seed,'best_epoch':be,'best_val_nll':bv,'epochs':len(hist),'training_seconds':time.time()-start,'checkpoint':str(ck),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'metrics':{s:metrics(pred[s],y[sp==s]) for s in pred}};dump(f'{name}/summary.json',sm);return sm
def raw_matrix():
 z=[]
 for p in (H/'raw').glob('matrix*.jsonl'):z += [json.loads(x) for x in open(p) if x.strip()]
 z=[r for r in z if r.get('scientific_outcome_valid') is True]
 z=[r for r in z if not (r['state_id']=='DB_MODE_val_008' and int(r['future_index'])==31)]
 out={}
 for r in z:
  fi=int(r['future_index'])
  if r['state_id']=='DB_MODE_val_008' and fi==32:fi=31
  out[(r['state_id'],r['controller'],int(r['mode_id']),fi)]=r
 return out
def main():
 x,y,n,sp,ids=arrays();tr=sp=='train';va=sp=='val';te=sp=='test';prior=np.clip(y[tr].mean(0),1e-5,1-1e-5);priorlog=np.log(prior/(1-prior));summ=[]
 # Outcome-independent baselines.
 gp={'model':'global_prior','metrics':{s:metrics(np.tile(priorlog,(sum(sp==s),1)),y[sp==s]) for s in ('train','val')}};dump('global_prior/summary.json',gp)
 ti=np.flatnonzero(tr);nnp={}
 for s in ('train','val','test'):
  q=[]
  for i in np.flatnonzero(sp==s):
   c=ti[ti!=i] if s=='train' else ti;j=c[np.argmin(np.linalg.norm(x[c]-x[i],axis=1))];v=np.clip(y[j],1e-4,1-1e-4);q.append(np.log(v/(1-v)))
  nnp[s]=np.asarray(q)
 dump('nearest_neighbor/summary.json',{'model':'nearest_neighbor','metrics':{s:metrics(nnp[s],y[sp==s]) for s in ('train','val')}});np.savez_compressed(H/'nearest_neighbor/predictions.npz',**nnp)
 lin=fit('linear',17);mlps=[fit('mlp',s) for s in SEEDS]
 win=min(mlps,key=lambda q:(q['metrics']['val']['nll'],q['metrics']['val']['brier'],q['seed']));dump('selected_selector.json',win)
 logits=np.load(H/win['model']/'predictions_train_val.npz')['val'];vy=y[va]
 z=minimize_scalar(lambda t:metrics(logits/np.exp(t),vy)['nll'],bounds=(-3,3),method='bounded');T=float(np.exp(z.x));rawm=metrics(logits,vy);calm=metrics(logits/T,vy);use=calm['nll']<rawm['nll']-1e-9;dump('calibration.json',{'temperature':T,'used':use,'raw_val':rawm,'calibrated_val':calm,'scope':'VAL only'})
 P=1/(1+np.exp(-np.clip(logits/(T if use else 1),-30,30)));valids=ids[va];mrows={(r['state_id'],int(r['mode_id'])):r for r in read(H/'val_mode_counts.csv')};safe={r['state_id']:r for r in read(H/'safety_mode_counts.csv') if r['split']=='val'};rr=raw_matrix();sweep=[]
 for tau in TAUS:
  total=dead=brk=0
  for i,sid0 in enumerate(valids):
   sid=str(sid0);m=int(np.argmax(P[i]));ab=P[i,m]<tau;r=safe[sid] if ab else mrows[(sid,m)];total+=int(r['successes']);dead+=int(r['deadlock']);ctl='safety' if ab else 'transformed';mm=-1 if ab else m
   for fi in range(32):brk+=bool(rr[(sid,'safety',-1,fi)]['success']) and not bool(rr[(sid,ctl,mm,fi)]['success'])
  sweep.append({'tau':tau,'total_successes':total,'breaks':brk,'deadlock':dead})
 tw=sorted(sweep,key=lambda r:(-r['total_successes'],r['breaks'],r['deadlock'],-r['tau']))[0];dump('selected_threshold.json',{**tw,'grid':list(TAUS),'scope':'VAL only','temperature_used':use});write('val_threshold_sweep.csv',sweep)
 # Everything is frozen above.  TEST metrics/outcomes are opened only now.
 model=MLP();template=model.init(jax.random.PRNGKey(0),jnp.zeros((1,x.shape[1])));params=serialization.from_bytes(template,Path(win['checkpoint']).read_bytes());testlog=np.asarray(model.apply(params,jnp.asarray(x[te])));np.savez_compressed(H/win['model']/'test_predictions.npz',test=testlog);Tuse=T if use else 1;testp=1/(1+np.exp(-np.clip(testlog/Tuse,-30,30)))
 # Complete model comparison metrics.
 models={'global_prior':np.tile(priorlog,(sum(te),1)),'linear':None,'nearest_neighbor':nnp['test'],win['model']:testlog}
 lp=serialization.from_bytes(Linear().init(jax.random.PRNGKey(0),jnp.zeros((1,x.shape[1]))),Path(lin['checkpoint']).read_bytes());models['linear']=np.asarray(Linear().apply(lp,jnp.asarray(x[te])))
 rows=[]
 for name,l in models.items():
  q=metrics(l,y[te]);rows.append({'model':name,'split':'test',**q})
 for q in [gp,lin,*mlps]:
  for s,m in q['metrics'].items():summ.append({'model':q['model'],'split':s,**m})
 summ += rows;write('training_summary.csv',summ)
 testids=ids[te];tm={(r['state_id'],int(r['mode_id'])):r for r in read(H/'test_mode_q64.csv')};ts={r['state_id']:r for r in read(H/'safety_mode_counts.csv') if r['split']=='test'};train_best=int(np.argmax(prior));selector=[];cover=[];selections=Counter()
 prevalence=[];structure=[]
 for split_name,fn,criterion in [('train','train_mode_counts.csv','strong_proxy'),('val','val_mode_counts.csv','strong_proxy'),('test','test_mode_q64.csv','B63')]:
  zz=read(H/fn);sids=sorted({r['state_id'] for r in zz})
  for m in range(12):
   q=[r for r in zz if int(r['mode_id'])==m];prevalence.append({'split':split_name,'mode_id':m,'feasible_states':sum(str(r[criterion]).lower()=='true' for r in q),'states':len(q),'prevalence':sum(str(r[criterion]).lower()=='true' for r in q)/len(q),'mean_Q':float(np.mean([float(r['empirical_Q']) for r in q]))})
  for sid in sids:
   q=[r for r in zz if r['state_id']==sid];structure.append({'split':split_name,'state_id':sid,'feasible_modes':sum(str(r[criterion]).lower()=='true' for r in q),'mean_mode_Q':float(np.mean([float(r['empirical_Q']) for r in q])),'best_mode_Q':max(float(r['empirical_Q']) for r in q)})
 write('mode_prevalence.csv',prevalence);write('state_mode_structure.csv',structure)
 # Pairwise overlap on exact TEST B63 supports.
 ov=[]
 for a in range(12):
  A={s for s in testids if str(tm[(str(s),a)]['B63']).lower()=='true'}
  for b in range(a+1,12):
   B={s for s in testids if str(tm[(str(s),b)]['B63']).lower()=='true'};u=A|B
   ov.append({'mode_a':a,'mode_b':b,'intersection':len(A&B),'union':len(u),'jaccard':len(A&B)/len(u) if u else ''})
 write('mode_overlap.csv',ov)
 for i,sid0 in enumerate(testids):
  sid=str(sid0);m=int(np.argmax(testp[i]));conf=float(testp[i,m]);ab=conf<tw['tau'];chosen=ts[sid] if ab else tm[(sid,m)];allm=[tm[(sid,j)] for j in range(12)];oracle=max(allm,key=lambda r:(float(r['empirical_Q']),-int(r['mode_id'])));fixed=tm[(sid,train_best)];selections['safety' if ab else str(m)]+=1
  ctl='safety' if ab else 'transformed';mm=-1 if ab else m;res=br=0
  for fi in range(64):
   a=bool(rr[(sid,'safety',-1,fi)]['success']);b=bool(rr[(sid,ctl,mm,fi)]['success']);res+=(not a) and b;br+=a and not b
  selector.append({'state_id':sid,'selected_mode':m,'confidence':conf,'abstained':ab,'selected_Q64':chosen['empirical_Q'],'selected_B63':chosen['B63'],'successes':chosen['successes'],'deadlock':chosen['deadlock'],'timeout':chosen['timeout'],'collision':chosen['collision'],'J_def_mean':chosen['J_def_mean'],'episode_length_mean':chosen['episode_length_mean'],'fixed_mode':train_best,'fixed_Q64':fixed['empirical_Q'],'fixed_B63':fixed['B63'],'oracle_mode':oracle['mode_id'],'oracle_Q64':oracle['empirical_Q'],'oracle_B63':oracle['B63'],'regret':float(oracle['empirical_Q'])-float(chosen['empirical_Q']),'rescue_vs_safety':res,'break_vs_safety':br})
  cov=oracle['B63']=='True' or oracle['B63'] is True;cover.append({'state_id':sid,'coverable':cov,'selector_B63':chosen['B63'],'selector_correct_B63':bool(cov and (chosen['B63']=='True' or chosen['B63'] is True)),'regret':float(oracle['empirical_Q'])-float(chosen['empirical_Q'])})
 write('test_results.csv',selector);write('coverable_state_analysis.csv',cover);write('mode_selection_statistics.csv',[{'selection':k,'count':v} for k,v in sorted(selections.items())])
 # Aggregate controller comparison; oracle is an upper bound using per-state outcomes.
 def agg(name,items):
  return {'controller':name,'states':len(items),'B63_states':sum(str(r['B63']).lower()=='true' for r in items),'mean_Q64':float(np.mean([float(r['empirical_Q']) for r in items])),'success':sum(int(r['successes']) for r in items),'trials':sum(int(r['trials']) for r in items),'deadlock':sum(int(r['deadlock']) for r in items),'timeout':sum(int(r['timeout']) for r in items),'collision':sum(int(r['collision']) for r in items),'J_def':float(np.average([float(r['J_def_mean']) for r in items],weights=[int(r['trials']) for r in items])),'episode_length':float(np.average([float(r['episode_length_mean']) for r in items],weights=[int(r['trials']) for r in items]))}
 safety_items=[ts[str(s)] for s in testids];fixed_items=[tm[(str(s),train_best)] for s in testids];sel_items=[];oracle_items=[]
 for r in selector:
  sid=r['state_id'];sel_items.append(ts[sid] if r['abstained'] else tm[(sid,int(r['selected_mode']))]);oracle_items.append(tm[(sid,int(r['oracle_mode']))])
 comp=[agg('Safety',safety_items),agg('best_fixed_transformed',fixed_items),agg('DB_selector',sel_items),agg('transformed_oracle',oracle_items)];write('controller_comparison.csv',comp)
 # Raw Toy diagnostic oracle.
 rawrows=read(H/'raw_toy_codebook_coverage.csv');raw_or=[]
 for sid0 in testids:
  q=[r for r in rawrows if r['state_id']==str(sid0)];raw_or.append(max(q,key=lambda r:(float(r['empirical_Q']),-int(r['mode_id']))))
 raw_summary=agg('raw_Toy_oracle',raw_or)
 ent=0.
 counts=np.array([selections.get(str(m),0) for m in range(12)],float);counts=counts[counts>0]
 if counts.size>1:
  p=counts/counts.sum();ent=float(-np.sum(p*np.log(p))/np.log(12))
 dec={'selected_model':win['model'],'selected_seed':win['seed'],'temperature':T,'temperature_used':use,'tau':tw['tau'],'train_prior_best_fixed_mode':train_best,'transformed_oracle':comp[-1],'selector':comp[2],'best_fixed':comp[1],'safety':comp[0],'raw_toy_oracle':raw_summary,'coverable_states':sum(r['coverable'] for r in cover),'selector_B63_accuracy_on_coverable':float(np.mean([r['selector_correct_B63'] for r in cover if r['coverable']])) if any(r['coverable'] for r in cover) else None,'selector_mean_regret':float(np.mean([float(r['regret']) for r in selector])),'normalized_selection_entropy':ent,'selection_counts':dict(selections),'rescue_vs_safety':sum(int(r['rescue_vs_safety']) for r in selector),'break_vs_safety':sum(int(r['break_vs_safety']) for r in selector)};dump('test_summary.json',dec);print(json.dumps(dec,indent=2))
if __name__=='__main__':main()
