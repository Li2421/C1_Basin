"""Read-only reuse, nested learning curves, outcome-blind shift. Run through Slurm GPU."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
OUT=Path(__file__).resolve().parent
OLD=ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion'
CONF=OLD.parent/'heldout_adaptive_selection_v1'
def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def data():
 p=read(OLD/'protocol.json');c=read(CONF/'protocol.json');a=dict(np.load(OLD/'physical_inputs.npz'));d=dict(np.load(OLD/'source_dataset.npz'));b=dict(np.load(CONF/'frozen_predictions.npz'));y=dict(np.load(CONF/'fresh_rollouts.npz'))
 assert p['controller']==c['controller'] and p['eta']==c['eta'] and p['environment']==c['environment']
 assert len(set(s['state_uid'] for s in p['states'])&set(s['state_uid'] for s in c['states']))==0
 assert (d['split'][:512]=='train').all() and (d['split'][512:544]=='validation').all()
 np.testing.assert_array_equal(b['state_uids'],y['state_uids'])
 # archived normalized eta, not physical eta, is the exact historical neural input
 return p,c,a,d,b,y

def run():
 import jax,jax.numpy as jnp,optax
 from flax import serialization
 from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import model
 assert any(z.platform=='gpu' for z in jax.devices()),'Reserve a GPU shard through Slurm'
 t=time.time();p,c,a,d,b,y=data();frozen=read(OLD/'models_frozen.json')
 x=np.concatenate([a['physical23'],b['state23']]);X=jnp.asarray(x);E=jnp.asarray(a['eta']);S=jnp.asarray(d['success'].astype('float32'));F=jnp.asarray(d['failure'].astype('float32'))
 cfg=p['training'];schedule=[50,100]+list(range(200,12001,200));V=np.arange(512,544);T=np.arange(608,864)
 tasks=[]
 orders=[np.arange(512)]+[np.random.default_rng(k).permutation(512) for k in [2026101101,2026101102]]
 for order,ix in enumerate(orders):
  for n in [32,64,128,256,512]:
   if order and n==512:continue
   tasks.append(dict(key=f'N1_o{order}_n{n}',phase='N1',order=order,n=n,train=ix[:n].tolist(),val=V.tolist(),test=T.tolist(),canonical=order==0))
 # Initial-state-only, signed goal angle chirality; no labels enter assignment.
 raw=np.concatenate([a['raw_physical23'],b['raw_state23']]);region=raw[:,11:15].mean(1)>0
 for side in [False,True]:
  candidates=np.flatnonzero(region[:512]==side);vv=V[region[V]==side];target=T[region[T]!=side];n=min(128,len(candidates))
  assert len(vv)>=8
  for order,seed in enumerate([2026101101,2026101102,2026101103]):
   rng=np.random.default_rng(seed);tr=rng.permutation(candidates)[:n]
   for mode in ['shift','random']:
    tr0=tr if mode=='shift' else rng.permutation(512)[:n]
    v0=vv if mode=='shift' else rng.permutation(V)[:len(vv)]
    tasks.append(dict(key=f'N3_{int(side)}_{mode}_o{order}',phase='N3',order=order,n=n,train=tr0.tolist(),val=v0.tolist(),test=target.tolist(),canonical=False,source_region=int(side),mode=mode))
 write(OUT/'splits.json',dict(tasks=tasks,source_states=[s['state_uid'] for s in p['states']],development_states=[s['state_uid'] for s in c['states']],region=region.tolist(),definition='mean signed goal antipodal deviation > 0'))
 manifest=[];pred={};repro={};fits=0;reused=0
 for arm,oldarm in [('scalar','physical23_eta'),('heads','physical23_heads')]:
  heads=arm=='heads';net=model('local_state_16heads' if heads else 'local_state_eta');opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001))
  @jax.jit
  def predict(pp,xx):
   if heads:return net.apply(pp,xx,jnp.zeros((len(xx),3)),jnp.zeros((len(xx),76)))
   ii=jnp.repeat(jnp.arange(len(xx)),16);jj=jnp.tile(jnp.arange(16),len(xx))
   return net.apply(pp,xx[ii],E[jj],jnp.zeros((len(ii),76))).reshape(len(xx),16)
  @jax.jit
  def block(pp,oo,ii,jj,nmean):
   def one(carry,ij):
    par,optim=carry;i,j=ij
    def loss(par):
     z=net.apply(par,X[i],E[j],jnp.zeros((len(i),76)))
     if heads:z=jnp.take_along_axis(z,j[:,None],1)[:,0]
     return jnp.mean(S[i,j]*jax.nn.softplus(-z)+F[i,j]*jax.nn.softplus(z))/nmean
    l,g=jax.value_and_grad(loss)(par);up,optim=opt.update(g,optim,par)
    return (optax.apply_updates(par,up),optim),l
   return jax.lax.scan(one,(pp,oo),(ii,jj))[0]
  for task in tasks:
   if task['phase'] != os.environ.get('STUDY_PHASE','N1'):continue
   key=task['key']+'_'+arm;dest=OUT/'fits'/key;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'complete.json').exists():
    rec=read(dest/'complete.json');z=np.load(dest/'predictions.npz')['logits'];manifest.append(rec);pred[key]=z;continue
   ts=time.time();tr=np.array(task['train']);vv=np.array(task['val']);hist=[];params=[];sources=[]
   if task['canonical']:
    sel=frozen['selections'][f'{oldarm}_n{task["n"]}'];step=sel['selected_step']
    for m in sel['models']:
     assert sha(m['path'])==m['sha256'];template=net.init(jax.random.PRNGKey(m['seed']),X[:1],jnp.zeros((1,3)),jnp.zeros((1,76)))
     params.append(serialization.from_bytes(template,Path(m['path']).read_bytes()));sources.append(m);reused+=1
   else:
    pairs=np.array([(i,j) for i in tr for j in range(16) if d['success'][i,j]+d['failure'][i,j]>0]);nmean=float(np.mean(d['success'][pairs[:,0],pairs[:,1]]+d['failure'][pairs[:,0],pairs[:,1]]));allparams=[]
    for seed in [17,23,41]:
     rng=np.random.default_rng(seed);pp=net.init(jax.random.PRNGKey(seed),X[:1],jnp.zeros((1,3)),jnp.zeros((1,76)));oo=opt.init(pp);last=0;hh=[];saved=[]
     for step in schedule:
      draw=rng.integers(len(pairs),size=(step-last,128));ii,jj=pairs[draw].transpose(2,0,1)
      pp,oo=block(pp,oo,jnp.asarray(ii),jnp.asarray(jj),jnp.asarray(nmean));zv=np.asarray(predict(pp,X[vv]));ss=d['success'][vv];ff=d['failure'][vv]
      hh.append(float((ss*np.logaddexp(0,-zv)+ff*np.logaddexp(0,zv)).sum()/(ss+ff).sum()));saved.append(serialization.to_bytes(pp));last=step
     hist.append(hh);allparams.append(saved);fits+=1
    best=int(np.argmin(np.mean(hist,axis=0)));step=schedule[best]
    for seed,saved in zip([17,23,41],allparams):
     cp=dest/f'seed{seed}_step{step}.msgpack';cp.write_bytes(saved[best]);params.append(serialization.from_bytes(pp,saved[best]));sources.append(dict(path=str(cp),sha256=sha(cp),seed=seed))
   zz=np.stack([np.asarray(predict(pp,X[T])) for pp in params]);pred[key]=zz
   if task['canonical'] and arm=='scalar' and task['n'] in [32,512]:
    err=float(abs(zz-b[f'physical23_eta_n{task["n"]}_logits']).max());assert err<1e-5;repro[str(task['n'])]=dict(max_logit_abs_error=err,choices_identical=bool(np.array_equal(zz.argmax(-1),b[f'physical23_eta_n{task["n"]}_logits'].argmax(-1))))
   rec=dict(key=key,task=task,arm=arm,checkpoint_step=step,model_sources=sources,validation_history=hist,elapsed_seconds=time.time()-ts,reused=task['canonical'])
   np.savez_compressed(dest/'predictions.npz',logits=zz);write(dest/'complete.json',rec);manifest.append(rec)
   print(key,'step',step,'seconds',round(time.time()-ts,2),flush=True)
 np.savez_compressed(OUT/('neural_predictions_'+os.environ.get('STUDY_PHASE','N1')+'.npz'),**pred);write(OUT/('training_manifest_'+os.environ.get('STUDY_PHASE','N1')+'.json'),dict(records=manifest,reproduction=repro,new_fits=fits,reused_checkpoints=reused,seconds=time.time()-t,jax_devices=[str(v) for v in jax.devices()]))
 # Hash all direct data/code dependencies, not new output checkpoint manifest (already hashed individually).
 files=[OLD/'protocol.json',CONF/'protocol.json',OLD/'source_dataset.npz',OLD/'physical_inputs.npz',OLD/'models_frozen.json',CONF/'fresh_rollouts.npz',CONF/'frozen_predictions.npz',Path(__file__),OLD.parent/'state_only_experiment.py']
 write(OUT/'DATA_INDEX.json',dict(files=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in files],train_cells=512*16,train_rollouts=512*16*4,validation_rollouts=32*16*16,development_rollouts=256*16*16,unique_rollouts=106496,new_rollouts=0,controller=p['controller'],eta_physical=p['eta']))
if __name__=='__main__':run()
