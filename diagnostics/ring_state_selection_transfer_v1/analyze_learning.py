import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
try:
 from .study import OUT,OLD,CONF,data,write,read
except ImportError:
 from study import OUT,OLD,CONF,data,write,read

def ci(v,alpha=.05):
 v=np.asarray(v);ix=np.random.default_rng(2026101103).integers(len(v),size=(5000,len(v)))
 return np.quantile(v[ix].mean(1),[alpha/2,1-alpha/2]).tolist()
def knn(x,tr,vv,test,s,f):
 candidates=[k for k in [1,3,5,9,17,33] if k<=len(tr)];scores=[];nn=np.argsort(((x[vv,None]-x[tr])**2).sum(-1),axis=1)
 for k in candidates:
  ix=tr[nn[:,:k]];q=(s[ix].sum(1)+.5)/(s[ix].sum(1)+f[ix].sum(1)+1);z=logit(q);scores.append(float((s[vv]*np.logaddexp(0,-z)+f[vv]*np.logaddexp(0,z)).sum()/(s[vv]+f[vv]).sum()))
 k=candidates[int(np.argmin(scores))];nn=np.argsort(((x[test,None]-x[tr])**2).sum(-1),axis=1);ix=tr[nn[:,:k]];q=(s[ix].sum(1)+.5)/(s[ix].sum(1)+f[ix].sum(1)+1)
 return logit(q)[None],dict(k=k,validation_nll=scores,candidates=candidates)
def metrics(z,Y,U,fixed=13):
 z=np.asarray(z);choice=z.argmax(-1);i=np.arange(len(Y));values=Y[i[None,:],choice];upper=values+U[i[None,:],choice];delta=values.mean(0)-Y[:,fixed];robust=values.mean(0)-(Y+U)[:,fixed]
 aa,bb=np.triu_indices(16,1);sgn=np.sign(Y[:,aa]-Y[:,bb]);ok=sgn!=0;rank=np.mean(((np.sign(z[:,:,aa]-z[:,:,bb])==sgn)[...,ok])) if False else float(np.mean([((np.sign(t[:,aa]-t[:,bb])==sgn)[ok]).mean() for t in z]))
 finite=Y.max(1);q=expit(z);qtrue=np.divide(Y,1-U,out=np.full_like(Y,np.nan),where=U<1)
 return dict(success_lower=float(values.mean()),success_upper=float(upper.mean()),success_by_fit=values.mean(1).tolist(),gain=float(delta.mean()),gain_ci95=ci(delta),gain_ci99=ci(delta,.01),gain_worst_unknown=float(robust.mean()),gain_worst_unknown_ci95=ci(robust),finite_eval_oracle_regret=float((finite-values.mean(0)).mean()),tie_aware_selection_accuracy=float((values==finite).mean()),ranking_accuracy=rank,q_mse=float(np.nanmean((q-qtrue)**2)),nll=float(np.mean(Y[None]*np.logaddexp(0,-z)+(1-Y-U)[None]*np.logaddexp(0,z))/np.mean(1-U)),selected_unknown=float((upper-values).mean()))
def main(phase='N1'):
 p,c,a,d,b,y=data();x=np.concatenate([a['physical23'],b['state23']]);Y=y['outcomes'][:,:,8:16,0].mean(-1);U=y['outcomes'][:,:,8:16,1].mean(-1);tasks=read(OUT/'splits.json')['tasks'];neural=dict(np.load(OUT/f'neural_predictions_{phase}.npz'));rows=[];allpred={};controls={}
 for task in tasks:
  if task['phase']!=phase:continue
  key=task['key'];tr=np.array(task['train']);vv=np.array(task['val']);ti=np.array(task['test'])-608
  kz,kinfo=knn(x,tr,vv,np.arange(608,864),d['success'],d['failure']);prior=(d['success'][tr].sum(0)+.5)/(d['success'][tr].sum(0)+d['failure'][tr].sum(0)+1)
  zs={arm:neural[key+'_'+arm] for arm in ['scalar','heads']};zs['knn']=kz;zs['prior']=np.broadcast_to(logit(prior),(1,256,16))
  for arm,z in zs.items():
   allpred[key+'_'+arm]=z;rows.append(dict(key=key,arm=arm,n=task['n'],order=task['order'],states=len(ti),mode=task.get('mode'),source_region=task.get('source_region'),**metrics(z[:,ti],Y[ti],U[ti]),**(kinfo if arm=='knn' else {})))
 summary=[]
 if phase=='N1':
  for n in [32,64,128,256,512]:
   for arm in ['scalar','heads','knn','prior']:
    rr=[r for r in rows if r['n']==n and r['arm']==arm];z=np.concatenate([allpred[r['key']+'_'+arm] for r in rr]);m=metrics(z,Y,U);summary.append(dict(n=n,arm=arm,subset_gains=[r['gain'] for r in rr],**m))
 nstar={arm:next((r['n'] for r in summary if r['arm']==arm and r['gain_ci95'][0]>0 and min(r['subset_gains'])>0),None) for arm in ['scalar','heads','knn']}
 repro=metrics(b['physical23_eta_n512_logits'],Y,U)
 result=dict(status='COMPLETED DEVELOPMENT',rows=rows,summary=summary,n_star_development=nstar,fixed13_success=float(Y[:,13].mean()),discovery_best_fixed=int(y['outcomes'][:,:,:8,0].mean((0,2)).argmax()),eval_hindsight_best_fixed=int(Y.mean(0).argmax()),eval_empirical_oracle=float(Y.max(1).mean()),historical_q512_recalculation=repro,unknown_total=int(y['numerical_unknown'].sum()),evidence_unit='physical state; fits and subsets not independent evidence',test_seed_indices=list(range(8,16)))
 np.savez_compressed(OUT/f'{phase}_predictions.npz',**allpred);write(OUT/f'{phase}_RESULTS.json',result)
 print(json.dumps(dict(phase=phase,nstar=nstar,summary=[{k:r[k] for k in ['arm','n','success_lower','gain','gain_ci95']} for r in summary]),indent=2))
if __name__=='__main__':
 import sys;main(sys.argv[1] if len(sys.argv)>1 else 'N1')
