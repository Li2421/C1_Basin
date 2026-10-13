import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import OUT,OLD,data,read,write,sha

def main():
 import jax,jax.numpy as jnp
 from flax import serialization
 from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import model
 from diagnostics.ring_state_selection_transfer_v1.analyze_learning import metrics
 p,c,a,d,b,y=data();frozen=read(OLD/'models_frozen.json');V=np.arange(512,544);S=d['success'][V];Y=y['outcomes'][:,:,8:,0].mean(-1);U=y['outcomes'][:,:,8:,1].mean(-1);schedule=[50,100]+list(range(200,12001,200));results=[];choices=[];cache={}
 for arm,oldarm in [('scalar','physical23_eta'),('heads','physical23_heads')]:
  heads=arm=='heads';net=model('local_state_16heads' if heads else 'local_state_eta');X=jnp.array(b['state23']);E=jnp.array(a['eta']);template=net.init(jax.random.PRNGKey(17),X[:1],jnp.zeros((1,3)),jnp.zeros((1,76)))
  @jax.jit
  def pred(pp):
   if heads:return net.apply(pp,X,jnp.zeros((256,3)),jnp.zeros((256,76)))
   ii=jnp.repeat(jnp.arange(256),16);jj=jnp.tile(jnp.arange(16),256);return net.apply(pp,X[ii],E[jj],jnp.zeros((4096,76))).reshape(256,16)
  for n in [32,64,128,256,512]:
   score=[]
   for step in schedule:
    v=[np.load(OLD/'fits'/f'{oldarm}_n{n}'/f'seed{s}'/f'step{step}_source.npz')['validation'] for s in [17,23,41]];score.append(float(np.mean([S[np.arange(32),z.argmax(-1)].mean()/16 for z in v])))
   step=schedule[int(np.argmax(score))];choices.append(dict(arm=arm,n=n,chosen_by_validation_success=step,validation_success=score,schedule=schedule))
   for rule,st in [('VAL_SELECTION',step),('fixed2000',2000),('VAL_NLL',frozen['selections'][f'{oldarm}_n{n}']['selected_step'])]:
    pp=[serialization.from_bytes(template,(OLD/'fits'/f'{oldarm}_n{n}'/f'seed{s}'/f'step{st}.msgpack').read_bytes()) for s in [17,23,41]];z=np.stack([np.asarray(pred(pa)) for pa in pp]);cache[f'{arm}_n{n}_{rule}']=z;results.append(dict(arm=arm,n=n,rule=rule,step=st,**metrics(z,Y,U)))
 write(OUT/'CHECKPOINT_DIAGNOSTIC_RESULTS.json',dict(status='EXPLORATORY CANONICAL SUBSET ONLY',choices=choices,results=results,new_training=0));np.savez_compressed(OUT/'CHECKPOINT_DIAGNOSTIC_PREDICTIONS.npz',**cache)
 print([{k:r[k] for k in ['arm','n','rule','step','success_lower']} for r in results])
if __name__=='__main__':main()
