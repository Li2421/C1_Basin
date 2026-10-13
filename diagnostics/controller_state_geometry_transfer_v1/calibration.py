"""0/1/2/4 target-state calibration with disjoint final states, saved-model reuse."""
from pathlib import Path
import json,hashlib
import numpy as np
from . import metrics,learn
from .main_analysis import setup,write
HERE=Path(__file__).resolve().parent

def experiment(Y,Ps,prior,cal,test,cost_per_state):
 assert not(set(cal)&set(test))
 unknown_test=~np.isfinite(Y[:,test]);lower=np.nan_to_num(Y[:,test],nan=0.);upper=np.nan_to_num(Y[:,test],nan=1.)
 rng=np.random.default_rng(20261012);perms=[rng.permutation(cal).tolist() for _ in range(20)];out={};n=len(test);draws=np.random.default_rng(20261011).integers(0,n,(2000,n))
 for model,P in Ps.items():
  out[model]={}
  for k in [0,1,2,4]:
   records={m:[] for m in ['zero_shot','prior_only','prior_conditional']};arrays={m:[] for m in records}
   for order in perms:
    if k:
     ss=order[:k];cal_y=Y[:,ss,...,:2];den=np.isfinite(cal_y).sum((1,4));target=np.divide(np.nansum(cal_y,(1,4)),den,out=np.broadcast_to(prior,(Y.shape[0],)+prior.shape).copy(),where=den>0);offset=target-P[:,ss].mean(1)
     prior_pred=np.broadcast_to(target[:,None],P[:,test].shape);cond=np.clip(P[:,test]+offset[:,None],0,1)
    else:prior_pred=np.broadcast_to(prior,P[:,test].shape);cond=P[:,test]
    for method,pr in [('zero_shot',P[:,test]),('prior_only',prior_pred),('prior_conditional',cond)]:
     r,a=metrics.stats(lower,pr,prior);ru,au=metrics.stats(upper,pr,prior);a['success_upper']=au['success'];r['success_upper']=ru['success'];records[method].append(r);arrays[method].append(a)
   results={}
   for method in records:
    result={}
    for key in arrays[method][0]:
     value=np.mean([a[key] for a in arrays[method]],0);drawmeans=[r[key] for r in records[method]]
     result[key]={'mean':float(value.mean()),'state_ci95':np.percentile(value[draws].mean(1),[2.5,97.5]).tolist(),'calibration_draw_p025_p975':np.percentile(drawmeans,[2.5,97.5]).tolist()}
    results[method]=result
   delta=np.mean([a['success']-b['success'] for a,b in zip(arrays['prior_conditional'],arrays['prior_only'])],0)
   out[model][str(k)]={'methods':results,'conditional_minus_prior':{'mean':float(delta.mean()),'state_ci95':np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist()},'nominal_target_rollout_cost_per_geometry':k*cost_per_state}
   for method,key in [('prior_only','prior_minus_zero'),('prior_conditional','conditional_minus_zero')]:
    dd=np.mean([a['success']-b['success'] for a,b in zip(arrays[method],arrays['zero_shot'])],0);out[model][str(k)][key]={'mean':float(dd.mean()),'state_ci95':np.percentile(dd[draws].mean(1),[2.5,97.5]).tolist()}
 return {'results':out,'calibration_subsets_nested':perms,'calibration_states':cal,'test_states':test,'additional_physical_rollouts':0,'CI_unit':'physical test state, averaged over fixed calibration draws; draw variability separately','offset_equals_prior_plus_centered_prediction':True,'unknown_test_outcomes':int(unknown_test.sum()),'unknown_calibration_outcomes':int((~np.isfinite(Y[:,cal,...,:2])).sum()),'unknown_policy':'calibration uses known labels only with source-prior fallback for empty cell; test success is rigorous lower bound, success_upper is upper bound for SAME decisions; Q/oracle/regret under lower completion are sensitivity estimates when unknown>0'}

def main():
 assert (HERE/'legacy_diagnostic_results.json').exists(),'Phase1 diagnostic audit required'
 D,specs,sets,test,cal=setup(False);tr,y=learn.rows(D,specs['MANY']);pr=learn.prior(tr,y,2,8);gs=[D['W'].index(w) for w in [.44,.50,.56,.62]]
 ps={str(s):np.load(HERE/'models/legacy_diagnostic/MANY'/f'ENT_{s}.npz')['pred'][gs] for s in learn.SEEDS};ps['ensemble']=np.mean(list(ps.values()),0)
 out={'toy_width':experiment(D['Y'][gs],ps,pr,cal,test,32)}
 src=HERE.parent/'controller_ff_factorial_repair_v1';z=np.load(src/'dataset.npz',allow_pickle=True);r=np.load(src/'raw_labels.npz',allow_pickle=True);ops=['isotropic','anisotropic','rotate_positive','rotate_negative','coupled'];es=[0,5,10,15]
 scenes=['toy_give_way','double_bottleneck','four_way_intersection','ring_exchange'];folds=['held_toy','held_db','held_four','held_ring'];index=[]
 for scene,fold in zip(scenes,folds):
  train=sorted(set(z['state_uid'][(z['scene']==scene)&(z['split']=='train')]));testids=sorted(set(z['state_uid'][(z['scene']==scene)&(z['split']=='validation')]));states=train+testids
  rows=np.array([[np.flatnonzero((z['scene']==scene)&(z['state_uid']==s)&(z['controller']==f'flow_{op}'))[0] for op in ops] for s in states])
  yy=r['success'][rows][:,:,es].astype(float)[None];observed=r['observed'][rows][:,:,es];unknown=r['unknown'][rows][:,:,es];assert observed[...,0:2].all()
  # Some source train cells have only Q2; calibration only reads0/1. Test must be Q4.
  yy=np.where((observed[None]>0)&(unknown[None]==0),yy,np.nan)
  assert observed[len(train):].all(), 'incomplete final test rollout'
  ps={}
  for seed in learn.SEEDS:
   f=src/'fits'/fold/'encounter_same_all_labels'/f'seed{seed}'/'predictions.npz';q=np.load(f);assert list(q['eta_indices'])==es;ps[str(seed)]=q['probability'][rows][None];index.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  ps['ensemble']=np.mean(list(ps.values()),0)
  prior=[]
  for op in ops:
   ids=np.flatnonzero((z['scene']!=scene)&(z['split']=='train')&(z['controller']==f'flow_{op}'))
   suc=r['success'][ids][:,es,:2];obs=r['observed'][ids][:,es,:2]-r['unknown'][ids][:,es,:2];prior.append(suc.sum((0,2))/obs.sum((0,2)))
  out[scene]=experiment(yy,ps,np.asarray(prior),list(range(len(train))),list(range(len(train),len(states))),40)
  out[scene]['state_uids']=states;out[scene]['scope']='retrospective LOSO calibration; original Flow checkpoint remains confounded with scene'
 write(HERE/'calibration_results.json',out);write(HERE/'data/calibration_prediction_index.json',index)
 write(HERE/'phase2_complete.json',{'completed':True,'strong_panel_available':False,'legacy_target_test_disjoint':True,'new_rollouts':0,'used_test_labels_for_calibration':False,'sets':list(out)})
 for name,result in out.items():
  print(name,{k:{'prior':v['methods']['prior_only']['success']['mean'],'conditional':v['methods']['prior_conditional']['success']['mean'],'zero':v['methods']['zero_shot']['success']['mean']} for k,v in result['results']['ensemble'].items()},flush=True)
if __name__=='__main__':main()
