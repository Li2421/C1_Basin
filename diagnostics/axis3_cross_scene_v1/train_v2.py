from common import *
from features import fit_transform,apply_transform
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp,optax
class Critic(nn.Module):
 @nn.compact
 def __call__(self,x):
  for width in [64,64,16]:x=nn.silu(nn.Dense(width)(x))
  return nn.Dense(1)(x)[...,0]

def fit(X,y,n,trainmask,heldmask,seed,folder,targets):
 cfg=read(HERE/'DESIGN_CONFIG.json')
 if (folder/'fit.json').exists():
  old=read(folder/'fit.json');assert old['seed']==seed and old['targets']==targets and old['train_rows']==int(trainmask.sum()) and old['held_rows']==int(heldmask.sum());assert all((folder/f'step{t}.msgpack').exists() for t in targets);tr0=fit_transform(X[trainmask]);saved=np.load(folder/'transform.npz');assert all(np.array_equal(saved[k],v) for k,v in tr0.items());return old['validation_nll']
 if folder.exists():
  import shutil
  archive=HERE/'incomplete_fits_before_gres_repair'/folder.relative_to(HERE/'fits');archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();shutil.move(str(folder),str(archive))
 folder.mkdir(parents=True,exist_ok=True);tr=fit_transform(X[trainmask]);np.savez_compressed(folder/'transform.npz',**tr);net=Critic();params=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,24),jnp.float32));opt=optax.chain(optax.clip_by_global_norm(cfg['clip']),optax.adamw(cfg['lr'],weight_decay=cfg['wd']));state=opt.init(params);xx=jnp.asarray(apply_transform(X[trainmask],tr));yy=jnp.asarray(y[trainmask]);nnn=jnp.asarray(n[trainmask]);valid=np.flatnonzero(n[trainmask]>0);hx=jnp.asarray(apply_transform(X[heldmask],tr));hy=jnp.asarray(y[heldmask]);hn=jnp.asarray(n[heldmask]);rng=np.random.default_rng(seed);previous=0;history=[]
 @jax.jit
 def block(params,state,ids):
  def step(carry,idx):
   p,st=carry
   def loss(pp):
    logits=net.apply(pp,xx[idx]);return jnp.mean((yy[idx]*jax.nn.softplus(-logits)+(nnn[idx]-yy[idx])*jax.nn.softplus(logits))/4)
   grad=jax.grad(loss)(p);updates,st=opt.update(grad,st,p);return (optax.apply_updates(p,updates),st),None
  return jax.lax.scan(step,(params,state),ids)[0]
 @jax.jit
 def validation(pa):
  t=net.apply(pa,hx);return jnp.sum(hy*jax.nn.softplus(-t)+(hn-hy)*jax.nn.softplus(t))/jnp.sum(hn)
 for target in targets:
  ids=rng.choice(valid,size=(target-previous,cfg['batch']));params,state=block(params,state,jnp.asarray(ids));jax.block_until_ready(params);history.append(float(validation(params)) if heldmask.any() else None);(folder/f'step{target}.msgpack').write_bytes(serialization.to_bytes(params));previous=target
 write(folder/'fit.json',dict(seed=seed,train_rows=int(trainmask.sum()),held_rows=int(heldmask.sum()),targets=targets,validation_nll=history,jax_enable_x64=bool(jax.config.x64_enabled)));return history

def worker(index):
 prereg();assert all(d.platform=='gpu' for d in jax.devices());jax.config.update('jax_enable_x64',False);cfg=read(HERE/'DESIGN_CONFIG.json');z=np.load(HERE/'SOURCE_TRAINING_DATA.npz');X,y,n,sc=z['X'],z['y'],z['n'],z['scenes']
 for target in SCENES[index::2]:
  source=[s for s in SCENES if s!=target];hist=[];source_mask=sc!=target
  for held in source:
   trainmask=source_mask&(sc!=held);heldmask=sc==held;assert not (trainmask&(sc==target)).any() and not (heldmask&(sc==target)).any()
   for seed in cfg['fit_seeds']:hist.append(fit(X,y,n,trainmask,heldmask,seed,HERE/'fits'/target/'cv'/held/f'seed{seed}',cfg['steps']));print('CV',target,held,seed,flush=True)
  mean=np.asarray(hist).mean(0);selected=cfg['steps'][int(mean.argmin())];write(HERE/'fits'/target/'SELECTED_STEP.json',dict(utc=utc(),selected_step=selected,mean_validation_nll=mean,target_scene=target,source_scenes=source,target_rows_in_train=0,target_rows_in_cv=0))
  for seed in cfg['fit_seeds']:fit(X,y,n,source_mask,np.zeros(len(X),bool),seed,HERE/'fits'/target/'full'/f'seed{seed}',[selected])
  seal(f'{target}_MODEL_FROZEN.receipt.json',list((HERE/'fits'/target).rglob('*.msgpack'))+list((HERE/'fits'/target).rglob('transform.npz'))+[HERE/'fits'/target/'SELECTED_STEP.json',HERE/'TRAINING_INPUTS_FROZEN.receipt.json',HERE/'train_v2.py'],source_scenes=source,target_scene=target,target_labels_used=False)
 print('GPU shard complete',index,flush=True)

def submit():
 import subprocess,shlex
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True);external_gpu=[line.split('|')[0].split('_')[0] for line in q.splitlines() if 'gres/' in line.split('|')[-1] and 'r34_gpu_fit' not in line];cmd=f'JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"train_v2.py"))} worker $SLURM_ARRAY_TASK_ID';args=['sbatch','--parsable','-p','gpu','--gres=shard:1','--cpus-per-task=1','--mem=8G','--time','06:00:00','--array','0-1%2','-J','r34_gpu_fit','-o',str(HERE/'fit_%A_%a.log'),'--wrap',cmd];current=os.environ.get('SLURM_JOB_ID')
 deps=sorted(set(external_gpu));dependency=('afterany:'+':'.join(deps)) if deps else '';dependency+=(',' if dependency else '')+'afterok:'+current if current else '';
 if dependency:args[1:1]=['--dependency',dependency]
 seal('TRAINING_IMPLEMENTATION_V2_FROZEN.receipt.json',[HERE/'train_v2.py',HERE/'features.py',HERE/'SOURCE_TRAINING_DATA.npz',HERE/'TRAINING_INPUTS_FROZEN.receipt.json'],before_submission=True,queue=q);job=subprocess.check_output(args,text=True).strip();write(HERE/'TRAINING_SUBMISSIONS_V2.json',dict(utc=utc(),job=job,args=args));from submit import stage;stage('fit_done',job);print('Submitted GPU fits',job,flush=True)
if __name__=='__main__':worker(int(sys.argv[2])) if sys.argv[1]=='worker' else submit()
