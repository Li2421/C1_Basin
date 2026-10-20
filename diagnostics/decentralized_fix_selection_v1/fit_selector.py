from support import *
from critic_fit import net_for,SEEDS,STEPS
from flax import serialization
import jax,jax.numpy as jnp,optax
G=guard()
GRID_A=[0,.01,.025,.05,.1,.2,.35,.5,.75,1]
GRID_B=[-20,-10,-5,0,2,4,6,8,10,12,16,20]
def standardize(x,train):
 m=x[train].mean(0);s=x[train].std(0);s=np.where(s>1e-6,s,1);return m,s
def encode(x,m,s):
 z=((x-m)/s).astype(np.float32);return np.concatenate([np.broadcast_to(z[:,None,:],(len(x),6,7)),np.broadcast_to(np.eye(6,dtype=np.float32),(len(x),6,6))],-1)
def fit(seed,x,labels,source,held,folder,step=None,held_x=None):
 folder.mkdir(parents=True,exist_ok=True);m,s=standardize(x,source);X=encode(x,m,s);np.savez_compressed(folder/'transform.npz',mean=m,scale=s);y=(labels==1).sum((2,3)).astype(np.float32);n=(labels>=0).sum((2,3)).astype(np.float32)
 xs=jnp.asarray(X[source].reshape(-1,13));yy=jnp.asarray(y[source].ravel());nn=jnp.asarray(n[source].ravel());valid=np.flatnonzero(np.asarray(nn)>0);net=net_for('A3');pa=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,13),jnp.float32));opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001));state=opt.init(pa)
 xv=jnp.asarray(X[held].reshape(-1,13));yv=jnp.asarray(y[held].ravel());nv=jnp.asarray(n[held].ravel())
 @jax.jit
 def block(pa,state,ids):
  def update(carry,idx):
   pp,st=carry
   def loss(p):
    logit=net.apply(p,xs[idx]);return jnp.mean((yy[idx]*jax.nn.softplus(-logit)+(nn[idx]-yy[idx])*jax.nn.softplus(logit))/64)
   grad=jax.grad(loss)(pp);delta,st=opt.update(grad,st,pp);return (optax.apply_updates(pp,delta),st),None
  return jax.lax.scan(update,(pa,state),ids)[0]
 @jax.jit
 def nll(pa):
  logits=net.apply(pa,xv);return jnp.sum(yv*jax.nn.softplus(-logits)+(nv-yv)*jax.nn.softplus(logits))/jnp.sum(nv)
 rng=np.random.default_rng(seed);targets=STEPS if step is None else [step];prev=0;history=[]
 for target in targets:
  ids=rng.choice(valid,size=(target-prev,128),replace=True).astype(np.int32);pa,state=block(pa,state,jnp.asarray(ids));path=folder/f'step{target}.msgpack';path.write_bytes(serialization.to_bytes(pa));history.append(float(nll(pa)));prev=target
 write(folder/'fit.json',dict(seed=seed,candidate_steps=targets,validation_nll=history,train_indices=source.tolist(),validation_indices=held.tolist(),checkpoint_sha256={str(t):sha(folder/f'step{t}.msgpack') for t in targets},guard=guard()))
 probability=None if held_x is None else np.asarray(jax.nn.sigmoid(net.apply(pa,jnp.asarray(encode(held_x,m,s).reshape(-1,13))))).reshape(len(held_x),6)
 return history,probability
def rule(x,A,B,R):return np.where(x[:,0]>A,R,np.where(x[:,4]>B,0,2))
def main():
 family=FAMILIES[int(sys.argv[1])] if sys.argv[1].isdigit() else sys.argv[1];prereg();dest=HERE/'folds'/family;receipt=read(dest/'SOURCE_ONLY.receipt.json')
 for p,h in receipt['files'].items():assert sha(p)==h
 z=np.load(dest/'SOURCE_ONLY.npz');x=z['features'];labels=z['labels'];fam=z['families'];assert family not in fam;hx=z['held_features'];assert np.isfinite(x).all() and np.isfinite(hx).all(),'Diagnostic gap: do not impute.'
 rate=(labels==1).mean((2,3));best=None;grid=[]
 for A in GRID_A:
  for B in GRID_B:
   for R in [3,4,5]:
    choice=rule(x,A,B,R);scores=rate[np.arange(len(x)),choice];score=float(np.mean([scores[fam==f].mean() for f in sorted(set(fam))]));r=dict(A=A,B=B,right_hand_arm=ARMS[R],source_family_mean_success=score);grid.append(r)
    if best is None or score>best['source_family_mean_success']+1e-14:best=r
 ri=ARMS.index(best['right_hand_arm']);rc=rule(hx,best['A'],best['B'],ri);write(dest/'RULE_GRID.json',grid);write(dest/'RULE_SELECTION.json',dict(**best,held_names=list(z['held_names']),choices=[ARMS[i] for i in rc],source_names=list(z['names']),held_labels_read=False))
 histories=[]
 for inner in sorted(set(fam)):
  source=np.flatnonzero(fam!=inner);held=np.flatnonzero(fam==inner)
  for seed in SEEDS:
   history,_=fit(seed,x,labels,source,held,dest/'cv'/inner/f'seed{seed}');histories.append(history);print(family,inner,seed,'CV complete',flush=True)
 mean=np.mean(histories,axis=0);step=STEPS[int(np.argmin(mean))];write(dest/'STEP_SELECTION.json',dict(selected_step=step,candidate_steps=STEPS,mean_source_validation_nll=mean,fold_seed_nll=histories,held_family=family,held_labels_read=False));prob=[]
 for seed in SEEDS:
  _,pred=fit(seed,x,labels,np.arange(len(x)),np.arange(len(x)),dest/'full'/f'seed{seed}',step=step,held_x=hx);prob.append(pred);print(family,seed,'full',step,flush=True)
 prob=np.asarray(prob);ensemble=prob.mean(0);choice=np.argmax(ensemble,axis=-1);np.savez_compressed(dest/'CRITIC_PREDICTIONS.npz',probability=prob,ensemble=ensemble,held_names=z['held_names'],arms=ARMS);write(dest/'CRITIC_SELECTION.json',dict(selected_step=step,held_names=list(z['held_names']),choices=[ARMS[i] for i in choice],seed_choices=[[ARMS[i] for i in np.argmax(p,axis=-1)] for p in prob],source_names=list(z['names']),held_labels_read=False));source_inventory(f'imports_selector_{family}.json');seal(f'folds/{family}/CHOICES_FROZEN.receipt.json',[dest/n for n in ['RULE_SELECTION.json','RULE_GRID.json','STEP_SELECTION.json','CRITIC_PREDICTIONS.npz','CRITIC_SELECTION.json']],before_held_evaluation=True,held_labels_read=False,guard=guard())
if __name__=='__main__':main()
