"""Conditional Q-only held-controller diagnostic, no fresh rollouts."""
import os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.ring_state_selection_transfer_v1.study import OUT,OLD,read,write,sha
from diagnostics.ring_state_selection_transfer_v1.analyze_learning import ci,metrics

def main():
 import jax,jax.numpy as jnp,optax
 from flax import serialization
 from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import model
 gate=read(OUT/'c1/RESULTS.json')['gate'];assert not gate['selection_transfer_qualified'],'Formal C2 design must be evaluated instead of the fallback'
 yy=np.load(OUT/'c1/labels.npz')['Y'][:5];static=np.load(OUT/'C2_STATIC_FEATURES.npz');a=np.load(OLD/'physical_inputs.npz');raw=a['raw_physical23'][:32];S=(yy[:,:,:,:4]==1).sum(-1).reshape(160,16).astype('float32');F=(yy[:,:,:,:4]==0).sum(-1).reshape(160,16).astype('float32');E=jnp.array(a['eta']);schedule=[50,100]+list(range(200,12001,200));order=np.random.default_rng(2026101123).permutation(32);tr=order[:16];vv=order[16:24];tt=order[24:];sources=[0,2,3];held=[1,4];train=np.array([c*32+s for c in sources for s in tr]);val=np.array([c*32+s for c in sources for s in vv]);sj,fj=jnp.array(S),jnp.array(F);rngroot=2026101123
 write(OUT/'C2_SPLITS.json',dict(train_states=tr.tolist(),validation_states=vv.tolist(),joint_test_states=tt.tolist(),source_controllers=sources,held_controllers=held,source_seed_indices=[0,1,2,3],target_evaluation_seed_indices=[4,5,6,7],unknown_identity='all zero source onehot before train-only normalization',status='DEVELOPMENT Q-only fallback; C2-main gate failed'))
 net=model('local_state_eta');opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001));records=[];predictions={};start=time.time()
 for arm in ['R0','R1','R2','R3','R4','R4_action','RB']:
  folder=OUT/'c2_fits'/arm;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'result.json').exists():
   records.append(read(folder/'result.json'));z=np.load(folder/'predictions.npz');predictions[arm]=z['matched'];predictions[arm+'_mismatched']=z['mismatched'];continue
  z=static[arm];xx=np.concatenate([np.broadcast_to(raw,(5,32,23)),z],-1).reshape(160,-1);mean=xx[train].mean(0);std=xx[train].std(0);scale=np.where(std>1e-5,std,1.);x=(xx-mean)/scale;X=jnp.array(x);mm=z.copy();mm[1]=z[3];mm[4]=z[0];xm=np.concatenate([np.broadcast_to(raw,(5,32,23)),mm],-1).reshape(160,-1);XM=jnp.array((xm-mean)/scale)
  @jax.jit
  def predict(pp,xx):
   ii=jnp.repeat(jnp.arange(len(xx)),16);jj=jnp.tile(jnp.arange(16),len(xx));return net.apply(pp,xx[ii],E[jj],jnp.zeros((len(ii),76))).reshape(len(xx),16)
  @jax.jit
  def block(pp,oo,ii,jj,nmean):
   def one(carry,ids):
    par,optim=carry;i,j=ids
    def loss(par):
     z=net.apply(par,X[i],E[j],jnp.zeros((len(i),76)));return jnp.mean(sj[i,j]*jax.nn.softplus(-z)+fj[i,j]*jax.nn.softplus(z))/nmean
    l,g=jax.value_and_grad(loss)(par);up,optim=opt.update(g,optim,par);return (optax.apply_updates(par,up),optim),l
   return jax.lax.scan(one,(pp,oo),(ii,jj))[0]
  pairs=np.array([(i,j) for i in train for j in range(16) if S[i,j]+F[i,j]>0]);nmean=float(np.mean(S[pairs[:,0],pairs[:,1]]+F[pairs[:,0],pairs[:,1]]));hist=[];saved=[]
  for seed in [17,23,41]:
   rng=np.random.default_rng(seed);pp=net.init(jax.random.PRNGKey(seed),X[:1],jnp.zeros((1,3)),jnp.zeros((1,76)));oo=opt.init(pp);last=0;h=[];pbuf=[]
   for step in schedule:
    draw=rng.integers(len(pairs),size=(step-last,128));ii,jj=pairs[draw].transpose(2,0,1);pp,oo=block(pp,oo,jnp.array(ii),jnp.array(jj),jnp.array(nmean));zv=np.asarray(predict(pp,X[val]));h.append(float((S[val]*np.logaddexp(0,-zv)+F[val]*np.logaddexp(0,zv)).sum()/(S[val]+F[val]).sum()));pbuf.append(serialization.to_bytes(pp));last=step
   hist.append(h);saved.append(pbuf)
  best=int(np.argmin(np.mean(hist,0)));selected=schedule[best];params=[]
  for seed,buf in zip([17,23,41],saved):
   (folder/f'seed{seed}.msgpack').write_bytes(buf[best]);params.append(serialization.from_bytes(pp,buf[best]))
  z=np.stack([np.asarray(predict(pa,X)).reshape(5,32,16) for pa in params]);zm=np.stack([np.asarray(predict(pa,XM)).reshape(5,32,16) for pa in params]);predictions[arm]=z;predictions[arm+'_mismatched']=zm
  if arm=='R0':np.testing.assert_array_equal(z,zm)
  rows=[]
  for c in held:
   for split,ix in [('seen_states',tr),('joint_unseen_states',tt)]:
    target=(yy[c,ix,:,4:]==1).mean(-1);unknown=(yy[c,ix,:,4:]<0).mean(-1);disc=(yy[c,ix,:,:4]==1).mean(-1);fixed=int(disc.mean(0).argmax());srcfixed=int((S[train].sum(0)/np.maximum((S+F)[train].sum(0),1)).argmax());nearest=0 if c==1 else 3;nearidx=nearest*32+tr;nearfix=int((S[nearidx].sum(0)/np.maximum((S+F)[nearidx].sum(0),1)).argmax());m=metrics(z[:,c,ix],target,unknown,fixed);mm=metrics(zm[:,c,ix],target,unknown,fixed);v=target[np.arange(len(ix)),z[:,c,ix].argmax(-1)].mean(0);vm=target[np.arange(len(ix)),zm[:,c,ix].argmax(-1)].mean(0);diff=v-vm
    rows.append(dict(controller_index=c,heldout_kind='interpolation' if c==1 else 'extrapolation',split=split,states=len(ix),target_discovery_fixed_eta=fixed,target_fixed_success=float(target[:,fixed].mean()),source_pooled_fixed_eta=srcfixed,source_pooled_fixed_success=float(target[:,srcfixed].mean()),nearest_source_fixed_eta=nearfix,nearest_source_fixed_success=float(target[:,nearfix].mean()),hindsight_target_best_fixed_success=float(target.mean(0).max()),matched=m,mismatched=mm,matched_minus_mismatched_success=float(diff.mean()),matched_minus_mismatched_ci95=ci(diff)))
  # Q controller-effect pattern correlation between the two held controllers; same fully held states.
  from scipy.special import expit
  truth=(yy[held][:,tt,:,4:]==1).mean(-1).mean(1);predictq=expit(z[:,held][:,:,tt]).mean((0,2));dt=truth[1]-truth[0];dp=predictq[1]-predictq[0];rho=float(np.corrcoef(dt,dp)[0,1]) if np.std(dp)>1e-10 and np.std(dt)>0 else None
  rec=dict(arm=arm,selected_step=selected,validation_nll_history=hist,rows=rows,held_controller_eta_effect_rho=rho);records.append(rec);write(folder/'result.json',rec);np.savez_compressed(folder/'normalization.npz',mean=mean,scale=scale);np.savez_compressed(folder/'predictions.npz',matched=z,mismatched=zm);print(arm,'step',selected,'rho',rho,flush=True)
 np.savez_compressed(OUT/'C2_FALLBACK_PREDICTIONS.npz',**predictions);write(OUT/'C2_FALLBACK_RESULTS.json',dict(status='COMPLETED EXPLORATORY Q-ONLY FALLBACK',formal_selection_transfer=False,new_rollouts=0,new_fits=21,seconds=time.time()-start,records=records,limitations=['16 train states per source controller, below N1 reliable scale','8 jointly unseen physical test states, Q4 evaluation; all previously used state pool DEVELOPMENT','Only affine endpoint-matched operator family, not real held-out Flow or geometry','Onehot unknown identity allzeros; target labels not used for calibration or normalization','Selection metrics descriptive because C1 selection-transfer gate failed']))
if __name__=='__main__':main()
