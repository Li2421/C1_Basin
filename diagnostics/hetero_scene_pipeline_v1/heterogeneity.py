"""Paired half-split global gap before any model fitting; fixed/hierarchical bootstrap."""
from common import *
from design import *
import numpy as np
P=200;B=5000;SEED=2026102651

def half_scores(counts,rng):
 # counts[scene,state,action] is success count over three controllers.
 M,S,A=counts.shape;order=np.argsort(rng.random((M,P,S)),axis=-1,kind='stable');train=counts[np.arange(M)[:,None,None],order[:,:,:S//2]].sum(2,dtype=np.int16);score=counts.sum(1,dtype=np.int16)[:,None]-train
 best=train.argmax(-1);own=np.take_along_axis(score,best[...,None],-1)[...,0];global_choice=train.sum(0,dtype=np.int32).argmax(-1);shared=score[:,np.arange(P),global_choice]
 return own,shared,best,global_choice,score

def boot(counts,rng,hierarchical):
 M,S,A=counts.shape;draws=np.empty(B)
 for begin in range(0,B,50):
  batch=min(50,B-begin);sc=rng.integers(M,size=(batch,M))if hierarchical else np.broadcast_to(np.arange(M),(batch,M));st=rng.integers(S,size=(batch,M,S));sample=counts[sc[:,:,None],st]
  order=np.argsort(rng.random((batch,M,P,S)),axis=-1,kind='stable')[:,:,:,:S//2]
  train=sample[np.arange(batch)[:,None,None,None],np.arange(M)[None,:,None,None],order].sum(3,dtype=np.int16);score=sample.sum(2,dtype=np.int16)[:,:,None]-train
  best=train.argmax(-1);own=np.take_along_axis(score,best[...,None],-1)[...,0];gc=train.sum(1,dtype=np.int32).argmax(-1)
  shared=np.take_along_axis(score,gc[:,None,:,None],axis=-1)[...,0];draws[begin:begin+batch]=(own-shared).sum((1,2),dtype=np.int64)/(M*P*3*(S//2))
 return draws

def main():
 assert not(HERE/'train_inner_COMPLETE.json').exists();tables=read(HERE/'LABEL_TABLES.json');evals=read(HERE/'EVALUATION_STATES.json')['scenes'];rng=np.random.default_rng(SEED);outputs={};draws={}
 for group,names in [('all8',list(tables)),('strongest3',[n for n in ['lit_circle_five','lit_swap8_obstacles','xsc_swap_n6']if n in tables])]:
  counts=np.stack([np.asarray(tables[n]['y'],dtype=np.int16)[:,[tables[n]['state_indices'].index(i)for i in evals[n]]].sum(0,dtype=np.int16)for n in names]);own,shared,best,gc,score=half_scores(counts,rng);den=3*8
  delta=(own-shared).mean()/den;fixed=boot(counts,rng,False);hier=boot(counts,rng,True);draws[group+'__states']=fixed;draws[group+'__scene_states']=hier
  full=counts.sum((0,1),dtype=np.int64).argmax();cross={}
  for i,n in enumerate(names):
   cross[n]={t:float((own[j]-score[j,np.arange(P),best[i]]).mean()/den)for j,t in enumerate(names)if j!=i}
  outputs[group]=dict(scenes=names,global_gap=float(delta),ci_states=np.percentile(fixed,[2.5,97.5]).tolist(),ci_scene_states=np.percentile(hier,[2.5,97.5]).tolist(),own_best_crossvalidated_rate=float(own.mean()/den),global_crossvalidated_rate=float(shared.mean()/den),
   global_full_sample_candidate=menu()[int(full)],global_selection_counts=np.bincount(gc,minlength=49).tolist(),global_candidates_by_split=[menu()[int(a)]for a in gc],
   per_scene={n:dict(own_best_rate=float(own[i].mean()/den),global_rate=float(shared[i].mean()/den),loss=float((own[i]-shared[i]).mean()/den),sample_best=menu()[int(counts[i].sum(0).argmax())])for i,n in enumerate(names)},cross_use_losses=cross,
   established=bool(delta>=.1 and np.percentile(fixed,2.5)>0))
 np.savez_compressed(HERE/'HETEROGENEITY_BOOTSTRAP.npz',**draws)
 write(HERE/'HETEROGENEITY_CHECK.json',dict(utc=utc(),before_any_training=True,groups=outputs,established=outputs['all8']['established'],seed=SEED,bootstrap_replicates=B,inner_half_splits=P,only_new_state_indices=evals,literature_states_not_new=True,all_thresholds_unrounded=True))
 print('Heterogeneity established:',outputs['all8']['established'],'gap',outputs['all8']['global_gap'],'CI',outputs['all8']['ci_states'],flush=True)
if __name__=='__main__':main()
