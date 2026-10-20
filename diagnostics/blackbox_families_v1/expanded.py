"""Four-arm source-only controller selector; evaluation outcomes are never read."""
from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('round16_fit',D/'ring_controller_critic_v1/fit.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
import jax,jax.numpy as jnp,optax
from flax import serialization
from diagnostics.controller_q_transfer_round3_v1.state_generator import physical23
jax.config.update('jax_enable_x64',False)
ARMS=['F10','F13','S500','RHR'];SEEDS=[17,23,41];STEPS=list(range(500,12001,500))
def transform():
 with np.load(D/'ring_controller_critic_v1/fits/full_transform.npz') as z:return {k:z[k] for k in z.files}
def features(states,r5):
 x=old.features(physical23(states).astype(np.float32),r5,None,transform(),'A3')[:,:,0,:-8];nc,ns,_=x.shape
 return np.concatenate([np.broadcast_to(x[:,:,None],(nc,ns,4,25)),np.broadcast_to(np.eye(4,dtype=np.float32),(nc,ns,4,4))],-1).astype(np.float32)
def dataset():
 p=read(HERE/'source_training_protocol.json');names=list(p['profiles']);states=p['states'];y=np.zeros((26,16,4),np.float32);n=y.copy();index=[]
 for c in read(HERE/'source_training_CASES.json'):
  r=read(HERE/'source_training/raw'/f"{c['rollout_uid']}.json");ci=names.index(c['base_controller_name']);si=c['state_index'];ai=ARMS.index(c['arm']);n[ci,si,ai]=1-r['numerical_failure'];y[ci,si,ai]=r['success'];index.append(dict(raw_path=str(HERE/'source_training/raw'/f"{c['rollout_uid']}.json"),sha256=sha(HERE/'source_training/raw'/f"{c['rollout_uid']}.json")))
 cases=read(D/'blackbox_addendum_v1/right_hand_tune_CASES.json');rr={r['rollout_uid']:r for r in read(D/'blackbox_addendum_v1/right_hand_tune_DB_ROWS.json')}
 for c in cases:
  if c['grid_index']!=19:continue
  r=rr[c['rollout_uid']];ci=names.index(c['base_controller_name']);si=c['state_index'];n[ci,si,3]=1-r['numerical_failure'];y[ci,si,3]=r['success']
 with np.load(D/'controller_behavior_repr_v1/FEATURES.npz') as z:flow_names=list(z['controller_names']);flow=z['R5']
 with np.load(D/'axis2_controller_v1/R5_FEATURES.npz') as z:other_names=list(z['controller_names']);other=z['features']
 r5=np.asarray([flow[flow_names.index(name)] if name in flow_names else other[other_names.index(name)] for name in names]);X=features(states,r5)
 write(HERE/'EXPANDED_TRAINING_DATA.json',dict(utc=utc(),controllers=names,source_states=[s['state_uid'] for s in states],source_raw=index,reused_right_hand_rows=416,source_right_hand_DB_sha256=sha(D/'blackbox_addendum_v1/right_hand_tune_DB_ROWS.json'),known_labels=int(n.sum()),unknown_labels=int(n.size-n.sum()),feature_sources={str(p):sha(p) for p in [D/'controller_behavior_repr_v1/FEATURES.npz',D/'axis2_controller_v1/R5_FEATURES.npz',D/'ring_controller_critic_v1/fits/full_transform.npz']},evaluation_labels_read=False))
 np.savez_compressed(HERE/'EXPANDED_TRAINING_DATA.npz',X=X,y=y,n=n,names=names)
 return X,y,n,names

def fit(X,y,n,si,hi,seed,folder,step=None):
 folder.mkdir(parents=True,exist_ok=True);net=old.net_for('A3');xs=jnp.asarray(X[si].reshape(-1,29));xv=jnp.asarray(X[hi].reshape(-1,29));ys=jnp.asarray(y[si].reshape(-1));ns=jnp.asarray(n[si].reshape(-1));yv=jnp.asarray(y[hi].reshape(-1));nv=jnp.asarray(n[hi].reshape(-1));valid=np.flatnonzero(np.asarray(ns)>0);assert len(valid) and float(nv.sum())>0
 pa=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,29),jnp.float32));opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001));state=opt.init(pa)
 @jax.jit
 def block(pa,st,ids):
  def update(carry,idx):
   p,os=carry
   def loss(pp):
    logits=net.apply(pp,xs[idx]);return jnp.mean((ys[idx]*jax.nn.softplus(-logits)+(ns[idx]-ys[idx])*jax.nn.softplus(logits))/4)
   change,os=opt.update(jax.grad(loss)(p),os,p);return (optax.apply_updates(p,change),os),None
  return jax.lax.scan(update,(pa,st),ids)[0]
 @jax.jit
 def validation(p):
  z=net.apply(p,xv);return jnp.sum(yv*jax.nn.softplus(-z)+(nv-yv)*jax.nn.softplus(z))/nv.sum()
 rng=np.random.default_rng(seed);previous=0;history=[]
 for target in STEPS if step is None else [step]:
  ids=rng.choice(valid,size=(target-previous,128),replace=True).astype(np.int32);pa,state=block(pa,state,jnp.asarray(ids));value=float(validation(pa));assert np.isfinite(value);history.append(value);previous=target
  if step is not None:(folder/'model.msgpack').write_bytes(serialization.to_bytes(pa))
 write(folder/'FIT.json',dict(seed=seed,steps=STEPS if step is None else [step],heldout_nll=history,train_controllers=list(map(int,si)),held_controllers=list(map(int,hi))))
 return history

def train():
 prereg();assert all(d.platform=='gpu' for d in jax.devices());X,y,n,names=dataset();folds=[[] for _ in range(6)];flow=[i for i,name in enumerate(names) if not name.startswith(('G','DM'))]
 for j,i in enumerate(flow):folds[j%6].append(i)
 for j in range(6):folds[j].extend([names.index('G'+str(j)),names.index('DM'+str(j))])
 histories=[]
 for j,held in enumerate(folds):
  hi=np.asarray(held);si=np.asarray([i for i in range(26) if i not in held])
  for seed in SEEDS:histories.append(fit(X,y,n,si,hi,seed,HERE/'expanded'/f'fold{j}'/str(seed)))
 means=np.mean(histories,axis=0);selected=STEPS[int(np.argmin(means))];write(HERE/'EXPANDED_STEP_SELECTION.json',dict(selected_step=selected,mean_heldout_nll=means.tolist(),candidate_steps=STEPS,folds=folds))
 for seed in SEEDS:fit(X,y,n,np.arange(26),np.arange(26),seed,HERE/'expanded/full'/str(seed),selected)
 for family,held in [('Ring Flow',flow),('Gaussian behavior cloning',[names.index('G'+str(j)) for j in range(6)]),('Decentralized model predictive control',[names.index('DM'+str(j)) for j in range(6)])]:
  hi=np.asarray(held);si=np.asarray([i for i in range(26) if i not in held])
  for seed in SEEDS:fit(X,y,n,si,hi,seed,HERE/'expanded/loto'/family/str(seed),selected)
 seal('EXPANDED_MODEL_FROZEN.receipt.json',[HERE/'expanded.py',HERE/'EXPANDED_TRAINING_DATA.json',HERE/'EXPANDED_TRAINING_DATA.npz',HERE/'EXPANDED_STEP_SELECTION.json',*list((HERE/'expanded').rglob('*.msgpack')),*list((HERE/'expanded').rglob('FIT.json'))],before_evaluation_labels=True,LOTO_evidence='DEVELOPMENT')
 print('expanded selector frozen',selected,flush=True)
if __name__=='__main__':train()
