"""Auxiliary state mechanism: outcome-free physical features predicting eta contrasts."""
import numpy as np
from study import OUT,data,read,write
from analyze_learning import ci

def features(states):
 rows=[]
 for s in states:
  p,v,g=[np.array(s['physical'][k]) for k in ['positions','velocities','goals']];r=np.linalg.norm(p,axis=1);rg=np.linalg.norm(g,axis=1);th=np.arctan2(p[:,1],p[:,0]);tg=np.arctan2(g[:,1],g[:,0]);wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi;gd=wrap(tg-th-np.pi);gap=np.diff(np.r_[np.sort(th),np.sort(th)[0]+2*np.pi]);op=wrap(th[[2,3]]-th[[0,1]]-np.pi);er=p/r[:,None];et=np.stack([-er[:,1],er[:,0]],-1);vr=(v*er).sum(-1);vt=(v*et).sum(-1);distance=np.linalg.norm(p[:,None]-p[None,:],axis=-1);distance[np.eye(4,dtype=bool)]=np.inf
  rows.append([r.mean(),r.std(),rg.mean(),rg.std(),gap.mean(),gap.std(),gap.min(),np.mean(abs(op)),gd.mean(),gd.std(),vr.mean(),vr.std(),vt.mean(),vt.std(),distance.min()-.32,np.linalg.norm(g-p,axis=1).mean()])
 names=['initial_radius_mean','initial_radius_std','goal_radius_mean','goal_radius_std','adjacent_angle_gap_mean','adjacent_angle_gap_std','adjacent_angle_gap_min','opposite_pair_abs_phase_error','goal_antipodal_angle_mean','goal_antipodal_angle_std','radial_velocity_mean','radial_velocity_std','tangential_velocity_mean','tangential_velocity_std','minimum_pair_clearance','straight_path_length_mean']
 return np.array(rows),names

def run():
 p,c,a,d,b,y=data();raw,names=features(p['states']+c['states']);ss=d['success'];ff=d['failure'];q=(ss+.5)/(ss+ff+1);tr=np.arange(512);val=np.arange(512,544);test=np.arange(608,864);Q=y['outcomes'][:,:,8:,0].mean(-1);U=y['outcomes'][:,:,8:,1].mean(-1);qt=Q/np.maximum(1-U,1e-9)
 def transform(train,degree):
  mu=raw[train].mean(0);sc=np.maximum(raw[train].std(0),1e-5);x=(raw-mu)/sc;out=x
  if degree==2:
   ii,jj=np.triu_indices(x.shape[1]);out=np.c_[x,x[:,ii]*x[:,jj]]
  return np.c_[np.ones(len(x)),out]
 def fit(train,degree,lam):
  x=transform(train,degree);z=q[train]-q[train].mean(1,keepdims=True);reg=np.eye(x.shape[1])*lam;reg[0,0]=1e-8;coef=np.linalg.solve(x[train].T@x[train]+reg,x[train].T@z);return x@coef,coef
 candidates=[]
 for degree in [1,2]:
  for lam in [.1,1.,10.,100.]:
   pred,coef=fit(tr,degree,lam);err=float(np.mean((pred[val]-(q[val]-q[val].mean(1,keepdims=True)))**2));candidates.append(dict(degree=degree,lam=lam,val_mse=err))
 best=min(candidates,key=lambda r:r['val_mse']);pred,coef=fit(tr,best['degree'],best['lam']);ix=pred[test].argmax(-1);gain=Q[np.arange(256),ix]-Q[:,13]
 true=qt-qt.mean(1,keepdims=True);prior=(q[tr]-q[tr].mean(1,keepdims=True)).mean(0);error=float(np.mean((pred[test]-true)**2));baseline=float(np.mean((prior-true)**2));folds=[]
 for f in range(4):
  tt=np.arange(512)[np.arange(512)%4!=f];vv=np.arange(512)[np.arange(512)%4==f];z,_=fit(tt,best['degree'],best['lam']);truth=q[vv]-q[vv].mean(1,keepdims=True);pr=(q[tt]-q[tt].mean(1,keepdims=True)).mean(0);folds.append(float(1-np.mean((z[vv]-truth)**2)/np.mean((pr-truth)**2)))
 # Simple one-feature correlations of eta contrasts on source/heldout, transparent direction checks.
 associations=[]
 for aeta,beta in [(1,13),(4,13),(10,13),(1,4)]:
  yy=q[tr,aeta]-q[tr,beta];yt=qt[:,aeta]-qt[:,beta];rr=[]
  for j,name in enumerate(names):
   if np.std(raw[tr,j])<1e-6:continue
   rr.append(dict(feature=name,source_r=float(np.corrcoef(raw[tr,j],yy)[0,1]),development_r=float(np.corrcoef(raw[test,j],yt)[0,1])))
  associations.append(dict(eta_pair=[aeta,beta],features=rr))
 np.savez_compressed(OUT/'N2_PHYSICAL_FEATURES.npz',raw=raw,predicted_eta_contrasts=pred,coefficients=coef,names=names)
 write(OUT/'N2_RESULTS.json',dict(status='COMPLETED AUXILIARY DEVELOPMENT',controller_relation='Fixed-C state-dependent interaction scale; cannot identify controller causality or cross-C invariance.',feature_names=names,features_definition=__file__,candidates=candidates,selected=best,heldout_contrast_mse=error,heldout_eta_prior_mse=baseline,heldout_contrast_R2_vs_prior=float(1-error/baseline),source_fourfold_R2=folds,success_lower=float(Q[np.arange(256),ix].mean()),gain_vs_eta13=float(gain.mean()),gain_ci95=ci(gain),associations=associations,limits=['Prediction from initial state only; no future labels in input','CV uses validation-selected regularization, so descriptive source stability, not nested-CV model selection','Statistical correlation is not a causal mechanism; controller changes remain untested by these features alone']))
 print(best,'R2',1-error/baseline,'gain',gain.mean())
if __name__=='__main__':run()
