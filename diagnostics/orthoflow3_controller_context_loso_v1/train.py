"""Source-label-only controller context experiment; frozen test gate."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json,time,hashlib,fcntl
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import jax,jax.numpy as jnp,optax
import flax.linen as nn
from flax import serialization
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'diagnostics/orthoflow3_loso_partial_count_v1'
FOLDS={'toy':'toy_giveway','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}
KINDS=('C1_full','C1_additive','C3_full')
jax.config.update('jax_default_matmul_precision','highest')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

class Critic(nn.Module):
 additive:bool=False
 @nn.compact
 def __call__(self,x,e,c):
  h=rep.Encoder(name='physical_encoder')(x);eta=nn.silu(nn.Dense(32,name='eta_encoder')(e))
  if self.additive:
   a=nn.silu(nn.Dense(128,name='a1')(jnp.concatenate([h,c],axis=-1)))
   a=nn.silu(nn.Dense(64,name='a2')(a));a=nn.Dense(1,name='a_out')(a)[...,0]
   b=nn.silu(nn.Dense(128,name='b1')(eta));b=nn.silu(nn.Dense(64,name='b2')(b))
   return a+nn.Dense(1,name='b_out')(b)[...,0]
  z=nn.silu(nn.Dense(128,name='shared1')(jnp.concatenate([h,eta,c],axis=-1)))
  z=nn.silu(nn.Dense(64,name='shared2')(z));return nn.Dense(1,name='out')(z)[...,0]

def materialize():
 rows=pq.read_table(OLD/'all_pairs.parquet').to_pylist();states=load(OLD/'states.json')
 c1=np.zeros((len(states),11),np.float32);c3=np.zeros((len(rows),11),np.float32);saw1=set();saw3=set();errors=[]
 for path in sorted((OUT/'features').glob('*.json')):
  for r in load(path):
   if r['candidate']=='C1':arr=c1;idx=r['state_index'];saw1.add(idx)
   else:arr=c3;idx=r['pair_index'];saw3.add(idx)
   if r['valid']:arr[idx]=np.r_[r['context'],1.]
   else:errors.append(r)
 assert len(saw1)==len(states),(len(saw1),len(states));assert len(saw3)==len(rows),(len(saw3),len(rows))
 np.savez_compressed(OUT/'source_context.npz',C1=c1,C3=c3)
 dump(OUT/'feature_audit.json',{'source_states':len(states),'source_pairs':len(rows),
   'C1_valid':int(c1[:,-1].sum()),'C3_valid':int(c3[:,-1].sum()),'invalid_policy':'keep label, zero context + explicit validity flag',
   'errors':errors,'full_rollouts':0,'maximum_simulated_seconds_per_probe':.15,
   'training_pairs_sha256':sha(OLD/'all_pairs.parquet'),'context_sha256':sha(OUT/'source_context.npz')})

def data(fold,kind):
 allrows=pq.read_table(OLD/'all_pairs.parquet').to_pylist();ii=np.array([i for i,r in enumerate(allrows) if r['scenario']!=FOLDS[fold]])
 rows=[allrows[i] for i in ii];si=np.array([r['state_index'] for r in rows]);x=dict(np.load(OLD/'entities.npz'))
 source=np.load(OUT/'source_context.npz');c=source['C1'][si] if kind.startswith('C1') else source['C3'][ii]
 mask=np.array([r['split']=='train' for r in rows]);norm=load(OLD/fold/'normalization.json')
 # State-only contexts normalize over unique source TRAIN states, not pair frequency.
 use=np.unique(si[mask],return_index=True)[1] if kind.startswith('C1') else np.arange(mask.sum())
 cc=c[mask][use];good=cc[:,-1]>0
 center=cc[good,:-1].mean(0);scale=np.maximum(cc[good,:-1].std(0),.05)
 c=np.c_[np.where(c[:,-1,None]>0,(c[:,:-1]-center)/scale,0),c[:,-1]].astype(np.float32)
 e=(np.asarray([r['eta'] for r in rows],np.float32)-np.asarray(norm['eta_center'],np.float32))/np.asarray(norm['eta_radius'],np.float32)
 s=np.asarray([r['s'] for r in rows],np.float32);f=np.asarray([r['f'] for r in rows],np.float32)
 groups={sc:{sp:np.array([i for i,r in enumerate(rows) if r['scenario']==sc and r['split']==sp]) for sp in ('train','validation')} for sc in FOLDS.values() if sc!=FOLDS[fold]}
 mean={sc:float((s+f)[g['train']].mean()) for sc,g in groups.items()};w=np.array([1/mean[r['scenario']] for r in rows],np.float32)
 dump(OUT/fold/kind/'normalization.json',{'context_center':center.tolist(),'context_scale':scale.tolist(),**norm,'source_scenes':list(groups),'target_data_used':False})
 return rows,x,e,c,si,s,f,w,groups

def train(fold,kind,seed):
 dest=OUT/fold/kind/f'seed{seed}'
 if (dest/'training.json').exists():return
 assert not (OUT/'target_predictions.json').exists()
 rows,x,e,c,si,s,f,w,groups=data(fold,kind);model=Critic(kind.endswith('additive'))
 p=model.init(jax.random.PRNGKey(seed),gather(x,[0]),jnp.zeros((1,3)),jnp.zeros((1,11)))
 opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(1e-3,weight_decay=1e-4));st=opt.init(p)
 @jax.jit
 def step(p,st,bx,be,bc,bs,bf,bw):
  def loss(pp):
   z=model.apply(pp,bx,be,bc)
   return jnp.mean((bs*jax.nn.softplus(-z)+bf*jax.nn.softplus(z))*bw)
  loss,g=jax.value_and_grad(loss)(p);u,st=opt.update(g,st,p);return optax.apply_updates(p,u),st,loss
 predict=jax.jit(lambda p,bx,be,bc:model.apply(p,bx,be,bc))
 rng=np.random.default_rng(seed);best=(np.inf,None,0);stale=0;hist=[];start=time.time()
 for it in range(1,4001):
  ix=np.concatenate([rng.choice(g['train'],96) for g in groups.values()])
  p,st,loss=step(p,st,gather(x,si[ix]),jnp.asarray(e[ix]),jnp.asarray(c[ix]),jnp.asarray(s[ix]),jnp.asarray(f[ix]),jnp.asarray(w[ix]))
  if it%100:continue
  val={}
  for sc,g in groups.items():
   ix=g['validation'];z=np.concatenate([np.asarray(predict(p,gather(x,si[j]),jnp.asarray(e[j]),jnp.asarray(c[j]))) for j in (ix[k:k+256] for k in range(0,len(ix),256))])
   val[sc]=float((s[ix]*np.logaddexp(0,-z)+f[ix]*np.logaddexp(0,z)).sum()/(s[ix]+f[ix]).sum())
  score=float(np.mean(list(val.values())));hist.append({'step':it,'train_loss':float(loss),'val':score,'scenes':val})
  if score<best[0]-1e-5:best=(score,serialization.to_bytes(p),it);stale=0
  else:stale+=1
  if it%500==0:print(json.dumps({'fold':fold,'kind':kind,'seed':seed,'step':it,'val':score,'seconds':time.time()-start}),flush=True)
  if stale>=10 and it>=1500:break
 dest.mkdir(parents=True,exist_ok=True);(dest/'checkpoint.msgpack').write_bytes(best[1]);dump(dest/'history.json',hist)
 dump(dest/'training.json',{'fold':fold,'kind':kind,'seed':seed,'validation_nll':best[0],'best_step':best[2],'steps':it,'seconds':time.time()-start,
  'checkpoint':str(dest/'checkpoint.msgpack'),'sha256':sha(dest/'checkpoint.msgpack'),'target_labels_used':False,'sources':list(groups),
  'dataset_sha256':sha(OLD/'all_pairs.parquet'),'features_sha256':sha(OUT/'source_context.npz'),'backend':jax.default_backend()})

def freeze():
 result={}
 for fold in FOLDS:
  result[fold]={}
  for kind in KINDS:
   runs=[load(OUT/fold/kind/f'seed{s}/training.json') for s in (17,23,41)]
   result[fold][kind]={'runs':runs,'selected':min(runs,key=lambda r:(r['validation_nll'],r['seed']))}
  # Candidate context chosen only by average source VAL likelihood across seeds.
  kinds=['C1_full','C3_full'];chosen=min(kinds,key=lambda k:np.mean([r['validation_nll'] for r in result[fold][k]['runs']]))
  result[fold]['source_selected_kind']=chosen
 dump(OUT/'models_frozen.json',{'folds':result,'selection':'source validation only','target_labels_used':False})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['materialize','train','freeze']);ap.add_argument('--fold',choices=FOLDS);ap.add_argument('--kind',choices=KINDS);ap.add_argument('--seed',type=int);a=ap.parse_args()
 if a.action=='materialize':materialize()
 elif a.action=='freeze':freeze()
 else:
  lock=OUT/a.fold/a.kind/f'seed{a.seed}'/'run.lock';lock.parent.mkdir(parents=True,exist_ok=True)
  with lock.open('a') as handle:
   fcntl.flock(handle,fcntl.LOCK_EX)
   train(a.fold,a.kind,a.seed)
