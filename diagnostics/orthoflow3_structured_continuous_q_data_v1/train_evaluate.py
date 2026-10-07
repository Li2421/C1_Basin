#!/usr/bin/env python3
"""Train structured/sparse Toy critics and frozen-panel DB transfer diagnostics."""
from __future__ import annotations
import csv,hashlib,json,os
from collections import Counter,defaultdict
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import flax.linen as nn
from flax import serialization,core
import jax,jax.numpy as jnp
import numpy as np,optax
import pyarrow.parquet as pq
from scipy.stats import rankdata,spearmanr

ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;OLD=D/'orthoflow3_continuous_basin_critic_v1';SEEDS=(17,23,41)
def dump(n,x):p=H/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(n,rows):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0]) if rows else ['empty']
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def h(x):return hashlib.sha256(x.encode()).hexdigest()
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def safe(x):return None if x is None or not np.isfinite(x) else float(x)

class Critic(nn.Module):
 @nn.compact
 def __call__(self,h,e):
  z=nn.silu(nn.Dense(128,name='state1')(h));z=nn.silu(nn.Dense(64,name='state2')(z));q=nn.silu(nn.Dense(32,name='eta1')(e));x=jnp.concatenate([z,q],-1)
  x=nn.silu(nn.Dense(128,name='c1')(x));x=nn.silu(nn.Dense(64,name='c2')(x));return nn.Dense(1,name='out')(x)[...,0]
class EtaOnly(nn.Module):
 @nn.compact
 def __call__(self,e):
  x=nn.silu(nn.Dense(32)(e));x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(64)(x));return nn.Dense(1)(x)[...,0]
class Joint(nn.Module):
 def setup(self):
  self.toy1=nn.Dense(128);self.toy2=nn.Dense(64);self.db1=nn.Dense(128);self.db2=nn.Dense(64);self.eta1=nn.Dense(32);self.c1=nn.Dense(128);self.c2=nn.Dense(64);self.out=nn.Dense(1)
 def body(self,z,e):
  q=nn.silu(self.eta1(e));x=jnp.concatenate([z,q],-1);return self.out(nn.silu(self.c2(nn.silu(self.c1(x)))))[...,0]
 def toy(self,h,e):return self.body(nn.silu(self.toy2(nn.silu(self.toy1(h)))),e)
 def db(self,h,e):return self.body(nn.silu(self.db2(nn.silu(self.db1(h)))),e)
 def __call__(self,ht,et,hd,ed):return self.toy(ht,et),self.db(hd,ed)

def loss_logits(logit,y,w):return jnp.sum(w*optax.sigmoid_binary_cross_entropy(logit,y))/jnp.maximum(jnp.sum(w),1)
def nll(p,y,w):
 p=np.clip(p,1e-7,1-1e-7);return float(np.sum(w*(-y*np.log(p)-(1-y)*np.log(1-p)))/np.sum(w))
def normalize(rows,scenario):
 man=json.load(open(OLD/'dataset_manifest.json'));sn=man['state_normalization'][scenario];ec=np.asarray(man['eta_normalization']['center']);es=np.asarray(man['eta_normalization']['scale'])
 hh=np.asarray([r['h_raw'] for r in rows],np.float32);ee=np.asarray([r['eta'] if 'eta' in r else [r['eta1'],r['eta2'],r['eta3']] for r in rows],np.float32)
 return {'rows':rows,'h':(hh-np.asarray(sn['mean'],np.float32))/np.asarray(sn['std'],np.float32),'e':(ee-ec)/es,
         'y':np.asarray([r['empirical_q'] for r in rows],np.float32),'w':np.asarray([min(r['n_trials'],16) for r in rows],np.float32)}

def train_single(train,val,seed,init=None,freeze_shared=False,eta_only=False):
 model=EtaOnly() if eta_only else Critic();key=jax.random.PRNGKey(seed)
 params=init if init is not None else (model.init(key,jnp.zeros((1,3))) if eta_only else model.init(key,jnp.zeros((1,train['h'].shape[1])),jnp.zeros((1,3))))
 if freeze_shared:
  labels=core.unfreeze(jax.tree_util.tree_map(lambda _: 'freeze',params));labels['params']['state1']=jax.tree_util.tree_map(lambda _:'train',labels['params']['state1']);labels['params']['state2']=jax.tree_util.tree_map(lambda _:'train',labels['params']['state2']);labels=core.freeze(labels)
  opt=optax.multi_transform({'train':optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4)),'freeze':optax.set_to_zero()},labels)
 else:opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4))
 state=opt.init(params)
 if eta_only:
  @jax.jit
  def step(p,s,e,y,w):
   v,g=jax.value_and_grad(lambda q:loss_logits(model.apply(q,e),y,w))(p);u,s=opt.update(g,s,p);return optax.apply_updates(p,u),s,v
 else:
  @jax.jit
  def step(p,s,hh,e,y,w):
   v,g=jax.value_and_grad(lambda q:loss_logits(model.apply(q,hh,e),y,w))(p);u,s=opt.update(g,s,p);return optax.apply_updates(p,u),s,v
 rng=np.random.default_rng(seed);best=None;bestscore=1e99;beststep=0;stale=0
 for it in range(1,4001):
  ix=rng.integers(0,len(train['y']),128)
  if eta_only:params,state,_=step(params,state,jnp.asarray(train['e'][ix]),jnp.asarray(train['y'][ix]),jnp.asarray(train['w'][ix]))
  else:params,state,_=step(params,state,jnp.asarray(train['h'][ix]),jnp.asarray(train['e'][ix]),jnp.asarray(train['y'][ix]),jnp.asarray(train['w'][ix]))
  if it%50==0:
   log=np.asarray(model.apply(params,jnp.asarray(val['e']))) if eta_only else np.asarray(model.apply(params,jnp.asarray(val['h']),jnp.asarray(val['e'])));score=nll(sigmoid(log),val['y'],val['w'])
   if score<bestscore-1e-6:best=jax.tree_util.tree_map(np.asarray,params);bestscore=score;beststep=it;stale=0
   else:stale+=1
   if stale>=20:break
 return model,best,{'seed':seed,'best_step':beststep,'steps':it,'val_nll':bestscore}

def eval_pred(model,params,data,eta_only=False):return sigmoid(np.asarray(model.apply(params,jnp.asarray(data['e'])) if eta_only else model.apply(params,jnp.asarray(data['h']),jnp.asarray(data['e']))))
def auc_ap(y,s):
 y=np.asarray(y,bool);s=np.asarray(s);P=y.sum();N=(~y).sum()
 if not P or not N:return None,None
 ranks=rankdata(s);auc=(ranks[y].sum()-P*(P+1)/2)/(P*N);o=np.argsort(-s);ys=y[o];prec=np.cumsum(ys)/(np.arange(len(ys))+1);return float(auc),float(np.sum(prec*ys)/P)
def metrics(p,d):
 y=d['y'];w=d['w'];rho=spearmanr(p,y).statistic if np.std(p)>0 and np.std(y)>0 else np.nan
 out={'pairs':len(y),'nll':nll(p,y,w),'brier':float(np.mean((p-y)**2)),'mae':float(np.mean(abs(p-y))),'spearman':safe(rho),'mean_q':float(y.mean()),'mean_pred':float(p.mean())}
 use=np.asarray([r['n_trials']>=16 for r in d['rows']]);yb=y[use]>=15/16;pb=p[use];au,ap=auc_ap(yb,pb)
 out.update({'b15_pairs':int(use.sum()),'b15_auroc':au,'b15_auprc':ap,'b15_accuracy':float(np.mean((pb>=15/16)==yb)) if len(pb) else None,
             'b15_precision':float(np.sum((pb>=15/16)&yb)/max(np.sum(pb>=15/16),1)) if len(pb) else None,'b15_recall':float(np.sum((pb>=15/16)&yb)/max(np.sum(yb),1)) if len(pb) else None})
 return out
def save_model(dirname,seed,params,summary):
 p=H/'models'/dirname/f'seed{seed}';p.mkdir(parents=True,exist_ok=True);(p/'checkpoint.msgpack').write_bytes(serialization.to_bytes(params));dump(Path('models')/dirname/f'seed{seed}'/'summary.json',summary)

def edge_subset(rows,fraction):
 n=round(len(rows)*fraction);by_s=defaultdict(list);by_e=defaultdict(list)
 for i,r in enumerate(rows):by_s[r['state_id']].append(i);by_e[r['eta_uid']].append(i)
 chosen=set()
 for s,ix in sorted(by_s.items()):chosen.add(min(ix,key=lambda i:h(f'edge-state-{fraction}|'+rows[i]['eta_uid'])))
 for e,ix in sorted(by_e.items()):chosen.add(min(ix,key=lambda i:h(f'edge-eta-{fraction}|'+rows[i]['state_id'])))
 rest=[i for i in range(len(rows)) if i not in chosen];rest.sort(key=lambda i:h(f'edge-fill-{fraction}|'+rows[i]['state_id']+'|'+rows[i]['eta_uid']));chosen.update(rest[:n-len(chosen)])
 assert len(chosen)==n and len({rows[i]['state_id'] for i in chosen})==48 and len({rows[i]['eta_uid'] for i in chosen})==40
 return [rows[i] for i in sorted(chosen)]

def init_db_from_toy(toyparams,seed,dim=80):
 model=Critic();p=core.unfreeze(model.init(jax.random.PRNGKey(seed),jnp.zeros((1,dim)),jnp.zeros((1,3))));t=core.unfreeze(toyparams)
 for k in ('eta1','c1','c2','out'):p['params'][k]=t['params'][k]
 return model,core.freeze(p)

def train_joint(toytr,toyval,dbtr,dbval,toyparams,seed):
 model=Joint();p=core.unfreeze(model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)),jnp.zeros((1,3)),jnp.zeros((1,80)),jnp.zeros((1,3))));t=core.unfreeze(toyparams)
 p['params']['toy1']=t['params']['state1'];p['params']['toy2']=t['params']['state2']
 for k in ('eta1','c1','c2','out'):p['params'][k]=t['params'][k]
 p=core.freeze(p);opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4));st=opt.init(p)
 @jax.jit
 def step(q,s,ht,et,yt,wt,hd,ed,yd,wd):
  def L(v):
   a,b=model.apply(v,ht,et,hd,ed);return .5*loss_logits(a,yt,wt)+.5*loss_logits(b,yd,wd)
  z,g=jax.value_and_grad(L)(q);u,s=opt.update(g,s,q);return optax.apply_updates(q,u),s,z
 rt=np.random.default_rng(seed);rd=np.random.default_rng(seed+1000);best=None;score0=1e99;bstep=0;stale=0
 for it in range(1,4001):
  a=rt.integers(0,len(toytr['y']),128);b=rd.integers(0,len(dbtr['y']),128)
  p,st,_=step(p,st,*[jnp.asarray(toytr[k][a]) for k in ('h','e','y','w')],*[jnp.asarray(dbtr[k][b]) for k in ('h','e','y','w')])
  if it%50==0:
   lt,ld=model.apply(p,jnp.asarray(toyval['h']),jnp.asarray(toyval['e']),jnp.asarray(dbval['h']),jnp.asarray(dbval['e']))
   score=.5*nll(sigmoid(np.asarray(lt)),toyval['y'],toyval['w'])+.5*nll(sigmoid(np.asarray(ld)),dbval['y'],dbval['w'])
   if score<score0-1e-6:best=jax.tree_util.tree_map(np.asarray,p);score0=score;bstep=it;stale=0
   else:stale+=1
   if stale>=20:break
 return model,best,{'seed':seed,'best_step':bstep,'steps':it,'val_nll_mean':score0}

def ranking(p,d,Ks=(8,16,32)):
 rows=d['rows'];by=defaultdict(list)
 for i,r in enumerate(rows):by[r['state_id']].append(i)
 out=[]
 for K in Ks:
  vals=[]
  for sid,ix in by.items():
   # Complete D panel has exactly 32 eta. Nested deterministic spatially diverse subsets.
   if len(ix)<K:continue
   e=d['e'][ix];first=min(range(len(ix)),key=lambda j:h(f'rank-{K}|'+rows[ix[j]]['eta_uid']));sel=[first];dist=np.linalg.norm(e-e[first],axis=1);dist[first]=-1
   while len(sel)<K:
    j=int(np.argmax(dist));sel.append(j);dist=np.minimum(dist,np.linalg.norm(e-e[j],axis=1));dist[sel]=-1
   qix=np.asarray(ix)[sel];pick=qix[np.argmax(p[qix])];oracle=qix[np.argmax(d['y'][qix])];ordr=qix[np.argsort(-p[qix])];has=np.any(d['y'][qix]>=15/16)
   vals.append({'selected_q':d['y'][pick],'oracle_q':d['y'][oracle],'regret':d['y'][oracle]-d['y'][pick],'has':has,'b15':d['y'][pick]>=15/16,'top3':np.any(d['y'][ordr[:3]]>=15/16)})
  cov=[x for x in vals if x['has']];out.append({'K':K,'states':len(vals),'mean_selected_q':float(np.mean([x['selected_q'] for x in vals])),'mean_oracle_q':float(np.mean([x['oracle_q'] for x in vals])),
    'mean_regret':float(np.mean([x['regret'] for x in vals])),'b15_coverable_states':len(cov),'b15_selection_rate':float(np.mean([x['b15'] for x in cov])) if cov else None,'top3_b15_hit_rate':float(np.mean([x['top3'] for x in cov])) if cov else None})
 return out

def main():
 sr=pq.read_table(H/'structured_pair_table.parquet').to_pylist();sm=pq.read_table(H/'structured_matched_control_basis.parquet').to_pylist();sem=pq.read_table(H/'structured_eta_matched_control_basis.parquet').to_pylist();sp=pq.read_table(H/'sparse_matched_control.parquet').to_pylist();se=pq.read_table(H/'sparse_eta_matched_control.parquet').to_pylist()
 parts={p:[r for r in sr if r['matrix_partition']==p] for p in ('TRAIN_TRAIN','VAL_VAL','TESTSTATE_TRAINETA','TRAINSTATE_TESTETA','TESTSTATE_TESTETA')}
 toyval=normalize(parts['VAL_VAL'],'Toy');evals={'A_seen_state_seen_eta':normalize(parts['TRAIN_TRAIN'],'Toy'),'B_unseen_state_seen_eta':normalize(parts['TESTSTATE_TRAINETA'],'Toy'),
  'C_seen_state_unseen_eta':normalize(parts['TRAINSTATE_TESTETA'],'Toy'),'D_unseen_state_unseen_eta':normalize(parts['TESTSTATE_TESTETA'],'Toy')}
 families={'toy_structured':parts['TRAIN_TRAIN'],'toy_structured_matched':sm,'toy_sparse_control':sp,'toy_structured_eta_matched':sem,'toy_sparse_eta_control':se,'toy_structured_25':edge_subset(parts['TRAIN_TRAIN'],.25),'toy_structured_50':edge_subset(parts['TRAIN_TRAIN'],.5)}
 summaries=[];saved={}
 for name,rr in families.items():
  tr=normalize(rr,'Toy')
  for seed in SEEDS:
   model,p,s=train_single(tr,toyval,seed);s.update(model=name,train_pairs=len(rr));save_model(name,seed,p,s);summaries.append(s);saved[(name,seed)]=(model,p)
 selected={}
 for name in families:
  ss=[x for x in summaries if x['model']==name];z=min(ss,key=lambda x:(x['val_nll'],x['seed']));selected[name]=z
 toy_metrics=[]
 for name in families:
  z=selected[name];model,p=saved[(name,z['seed'])]
  for reg,d in evals.items():toy_metrics.append({'model':name,'regime':reg,**metrics(eval_pred(model,p,d),d)})
 write('toy_prediction_metrics.csv',toy_metrics)
 full=selected['toy_structured'];fm,fp=saved[('toy_structured',full['seed'])];dp=eval_pred(fm,fp,evals['D_unseen_state_unseen_eta'])
 ranks=ranking(dp,evals['D_unseen_state_unseen_eta']);write('toy_ranking_metrics.csv',ranks)
 write('toy_b15_metrics.csv',[{k:v for k,v in r.items() if k in ('model','regime','b15_pairs','b15_auroc','b15_auprc','b15_accuracy','b15_precision','b15_recall')} for r in toy_metrics])
 ab=[]
 for frac,name in [(25,'toy_structured_25'),(50,'toy_structured_50'),(100,'toy_structured')]:
  r=next(x for x in toy_metrics if x['model']==name and x['regime']=='D_unseen_state_unseen_eta');ab.append({'density_percent':frac,'train_pairs':len(families[name]),**{k:r[k] for k in ('nll','mae','spearman','b15_accuracy','b15_auroc')}})
 write('matrix_density_ablation.csv',ab)

 # Frozen DB dataset/test panel from v1, no new DB evidence or selection.
 old=pq.read_table(OLD/'pair_table.parquet').to_pylist();dbtr=normalize([r for r in old if r['scenario']=='DB' and r['sampled_train']],'DB');dbval=normalize([r for r in old if r['scenario']=='DB' and r['sampled_val']],'DB')
 dbtest_rows=[r for r in old if r['scenario']=='DB' and r['sampled_eval'] and r['regime']=='D_unseen_state_unseen_eta'];dbtest=normalize(dbtest_rows,'DB')
 dump('db_frozen_test_manifest.json',{'source':str(OLD/'pair_table.parquet'),'rows':len(dbtest_rows),'states':len({r['state_uid'] for r in dbtest_rows}),'eta':len({r['eta_uid'] for r in dbtest_rows}),
  'selection':'exact frozen D regime from ORTHOFLOW3_CONTINUOUS_BASIN_CRITIC_V1','new_db_rollout':0,'keys':[{'state_uid':r['state_uid'],'eta_uid':r['eta_uid'],'controller_uid':r['controller_uid']} for r in dbtest_rows]})
 dbsum=[];dbsaved={}
 for name,eta_only in [('db_only',False),('eta_only',True)]:
  for seed in SEEDS:
   m,p,s=train_single(dbtr,dbval,seed,eta_only=eta_only);s.update(model=name);save_model(name,seed,p,s);dbsum.append(s);dbsaved[(name,seed)]=(m,p,eta_only)
 # Toy-pretrained shared continuous layers, DB adapter only.
 for seed in SEEDS:
  tm,tp=saved[('toy_structured',seed)];m,init=init_db_from_toy(tp,seed);m,p,s=train_single(dbtr,dbval,seed,init=init,freeze_shared=True);s.update(model='toy_pretrained_db_adapter');save_model('toy_pretrained_db_adapter',seed,p,s);dbsum.append(s);dbsaved[('toy_pretrained_db_adapter',seed)]=(m,p,False)
 # Secondary joint fine-tune.
 toytr=normalize(parts['TRAIN_TRAIN'],'Toy')
 for seed in SEEDS:
  jm,jp,s=train_joint(toytr,toyval,dbtr,dbval,saved[('toy_structured',seed)][1],seed);s.update(model='joint_finetune');save_model('joint_finetune',seed,jp,s);dbsum.append(s);dbsaved[('joint_finetune',seed)]=(jm,jp,False)
 dbres=[]
 for name in ('eta_only','db_only','toy_pretrained_db_adapter','joint_finetune'):
  cand=[x for x in dbsum if x['model']==name];z=min(cand,key=lambda x:(x.get('val_nll',x.get('val_nll_mean')),x['seed']));m,p,eo=dbsaved[(name,z['seed'])]
  if name=='joint_finetune':pred=sigmoid(np.asarray(m.apply(p,jnp.asarray(dbtest['h']),jnp.asarray(dbtest['e']),method=m.db)))
  else:pred=eval_pred(m,p,dbtest,eo)
  dbres.append({'model':name,'seed':z['seed'],**metrics(pred,dbtest)})
 write('db_transfer_results.csv',dbres);write('training_summary.csv',summaries+dbsum);dump('selected_checkpoints.json',{'toy':selected,'db':{r['model']:{'seed':r['seed']} for r in dbres}})
 print(json.dumps({'toy_D':[r for r in toy_metrics if r['regime']=='D_unseen_state_unseen_eta'],'ranking':ranks,'ablation':ab,'db':dbres},indent=2))
if __name__=='__main__':main()
