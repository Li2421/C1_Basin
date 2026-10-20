from common import *
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp,optax
from diagnostics.ring_controller_critic_v1 import fit as old
jax.config.update('jax_enable_x64',False)
class MixedNet(nn.Module):
 @nn.compact
 def __call__(self,x):
  for width in [64,64,16]:x=nn.silu(nn.Dense(width)(x))
  return nn.Dense(1)(x)[...,0]
def build_features(states,r5,flag):
 tr={k:v for k,v in np.load(D/'ring_controller_critic_v1/fits/full_transform.npz').items()};physical=old.physical23(states).astype(np.float32);x=old.features(physical,r5,None,tr,'A3');ii=[old.ARMS.index(a) for a in MENU];base=x[:,:,ii,:25];onehot=np.broadcast_to(np.eye(6,dtype=np.float32),(len(r5),len(states),6,6));flags=np.full((*base.shape[:-1],1),flag,np.float32)
 return np.concatenate([base,onehot,flags],-1).astype(np.float32)
def prepare():
 prereg();d=old.source_data();ii=[old.ARMS.index(a) for a in MENU];xf=build_features(d['states'],d['r5'],0);yf=d['y'][:,:,ii];nf=d['n'][:,:,ii];z=np.load(D/'other_controller_classes_v1/LABELS.npz');zz=np.load(D/'other_controller_classes_v1/R5_FEATURES.npz');names=z['controller_names'].tolist();chosen=[i for i,n in enumerate(names) if n.startswith(('M','P')) and n[1:].isdigit()];assert len(chosen)==12
 states=read(D/'other_controller_classes_v1/main_protocol.json')['states'];assert [s['state_uid'] for s in states]==z['state_uids'].tolist();r5=zz['features'][[zz['controller_names'].tolist().index(names[i]) for i in chosen]];xt=build_features(states,r5,1);yt=z['success'][chosen].sum(-1);nt=(1-z['unknown'][chosen]).sum(-1)
 X=np.concatenate([xf.reshape(-1,32),xt.reshape(-1,32)]);y=np.concatenate([yf.ravel(),yt.ravel()]).astype(np.float32);n=np.concatenate([nf.ravel(),nt.ravel()]).astype(np.float32)
 groups=np.concatenate([np.repeat(np.asarray(['F:'+str(x) for x in d['recipes']]),32*6),np.repeat(np.asarray([names[i][0] for i in chosen]),32*6)]);classes=np.concatenate([np.repeat('Flow',len(xf)*32*6),np.repeat(np.asarray(['MPC' if names[i].startswith('M') else 'potential' for i in chosen]),32*6)])
 np.savez_compressed(HERE/'MIXED_TRAINING_DATA.npz',X=X,y=y,n=n,groups=groups,classes=classes)
 paths=[D/'ring_controller_population_v1/dataset.jsonl',D/'ring_controller_population_v1/R5_FEATURES.npz',D/'other_controller_classes_v1/LABELS.npz',D/'other_controller_classes_v1/R5_FEATURES.npz',D/'ring_controller_critic_v1/fits/full_transform.npz',D/'state_nuisance_rootcause_v1/real_state_expansion/physical_inputs.npz',HERE/'MIXED_TRAINING_DATA.npz'];seal('MIXED_INPUTS.receipt.json',paths,old_source_only=True,new_type_labels_read=False,Flow_controllers=56,nonflow_controllers=12,rows=len(X),features=32)
 print('Mixed source prepared',X.shape,flush=True)
def fit_one(X,y,n,train,held,seed,folder,targets):
 folder.mkdir(parents=True,exist_ok=True);net=MixedNet();pa=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,32),jnp.float32));opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001));state=opt.init(pa);xx=jnp.asarray(X[train]);yy=jnp.asarray(y[train]);nnn=jnp.asarray(n[train]);valid=np.flatnonzero(n[train]>0);assert len(valid)
 if len(held):xv=jnp.asarray(X[held]);yv=jnp.asarray(y[held]);nv=jnp.asarray(n[held])
 @jax.jit
 def block(pa,state,ids):
  def step(carry,idx):
   p,st=carry
   def loss(pp):
    t=net.apply(pp,xx[idx]);return jnp.mean((yy[idx]*jax.nn.softplus(-t)+(nnn[idx]-yy[idx])*jax.nn.softplus(t))/4)
   grad=jax.grad(loss)(p);updates,st=opt.update(grad,st,p);return (optax.apply_updates(p,updates),st),None
  return jax.lax.scan(step,(pa,state),ids)[0]
 @jax.jit
 def validation(pa):
  t=net.apply(pa,xv);return jnp.sum(yv*jax.nn.softplus(-t)+(nv-yv)*jax.nn.softplus(t))/jnp.sum(nv)
 rng=np.random.default_rng(seed);previous=0;history=[]
 for target in targets:
  ids=rng.choice(valid,size=(target-previous,256));pa,state=block(pa,state,jnp.asarray(ids));jax.block_until_ready(pa);value=float(validation(pa)) if len(held) else None;history.append(value);(folder/f'step{target}.msgpack').write_bytes(serialization.to_bytes(pa));previous=target
 write(folder/'fit.json',dict(seed=seed,train_rows=len(train),held_rows=len(held),targets=targets,validation_nll=history,jax_enable_x64=bool(jax.config.x64_enabled)))
 return history

def train():
 prereg();assert all(d.platform=='gpu' for d in jax.devices()),'GPU required; no silent CPU fallback';z=np.load(HERE/'MIXED_TRAINING_DATA.npz');X,y,n=z['X'],z['y'],z['n'];groups=z['groups'];classes=z['classes'];unique=sorted(set(groups.tolist()));assert len(unique)==16;cfg=read(HERE/'DESIGN_CONFIG.json')['mixed_critic'];history=[]
 for fold in range(8):
  held=np.isin(groups,[unique[fold],unique[fold+8]]);tr=np.flatnonzero(~held);hv=np.flatnonzero(held)
  for seed in cfg['seeds']:
   h=fit_one(X,y,n,tr,hv,seed,HERE/'mixed/cv'/f'fold{fold}'/f'seed{seed}',cfg['steps']);history.append(h);print('CV',fold,seed,'done',flush=True)
 mean=np.mean(history,0);selected=cfg['steps'][int(mean.argmin())];write(HERE/'MIXED_SELECTED_STEP.json',dict(utc=utc(),selected_step=selected,candidate_steps=cfg['steps'],mean_validation_nll=mean,fold_seed_nll=history,groups=unique))
 for seed in cfg['seeds']:fit_one(X,y,n,np.arange(len(X)),np.array([],int),seed,HERE/'mixed/full'/f'seed{seed}',[selected])
 for heldtype in ['Flow','MPC','potential']:
  held=classes==heldtype
  for seed in cfg['seeds']:fit_one(X,y,n,np.flatnonzero(~held),np.flatnonzero(held),seed,HERE/'mixed/loto'/heldtype/f'seed{seed}',[cfg['LOTO_step']])
 files=list((HERE/'mixed').rglob('*.msgpack'))+[HERE/'MIXED_SELECTED_STEP.json',HERE/'MIXED_INPUTS.receipt.json',HERE/'train_mixed.py'];seal('MIXED_FROZEN.receipt.json',files,before_any_new_type_pilot_labels=True,jax_enable_x64=bool(jax.config.x64_enabled),platform=[str(d) for d in jax.devices()]);print('MIXED CRITIC FROZEN',selected,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
