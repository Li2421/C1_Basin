"""Paired finite-panel metrics; independent seed halves, no imputation."""
import numpy as np

def corr(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();a=a-a.mean();b=b-b.mean();d=np.linalg.norm(a)*np.linalg.norm(b)
 return float(a@b/d) if d>1e-12 else None

def center(a,axes):
 for ax in axes:a=a-a.mean(ax,keepdims=True)
 return a

def select(q,truth):return np.take_along_axis(truth,q.argmax(-1)[...,None],-1)[...,0]

def stats(y,p,prior):
 # shape G,S,C,E,R; final dimension is independent future seeds, not training seeds
 assert np.isfinite(y).all() and np.isfinite(p).all()
 a=y[...,:y.shape[-1]//2].mean(-1);b=y[...,y.shape[-1]//2:].mean(-1);q=y.mean(-1)
 chosen=select(p,b);oracle=select(a,b);fixed=select(np.broadcast_to(prior,p.shape),b)
 globalfixed=select(np.broadcast_to(prior.mean(0,keepdims=True),p.shape),b)
 meanstate=select(np.broadcast_to(p.mean(1,keepdims=True),p.shape),b)
 s=p.shape[1];ix=p.argmax(-1)
 # For each state evaluate the eta choices from every OTHER state, exactly.
 mismatch=np.zeros_like(chosen)
 for j in range(s):mismatch[:,j]=np.mean([np.take_along_axis(b[:,j],ix[:,k,:,None],-1)[...,0] for k in range(s) if k!=j],axis=0) if s>1 else chosen[:,j]
 arrays={'success':chosen.mean((0,2)),'crossfit_oracle':oracle.mean((0,2)),'regret':(oracle-chosen).mean((0,2)),
 'per_controller_fixed':fixed.mean((0,2)),'global_fixed':globalfixed.mean((0,2)),'random':b.mean((0,2,3)),
 'adaptive_gain':(chosen-meanstate).mean((0,2)),'matched_minus_mismatched':(chosen-mismatch).mean((0,2)),
 'oracle_adaptive_gain':(oracle-select(np.broadcast_to(a.mean(1,keepdims=True),a.shape),b)).mean((0,2)),
 'gain_over_prior':(chosen-fixed).mean((0,2)),
 'q_mse':((p-q)**2).mean((0,2,3)),'q_mse_unbiased':((p-a)*(p-b)).mean((0,2,3)),
 'joint_controller_eta_success':select(p.reshape(p.shape[:2]+(-1,)),b.reshape(b.shape[:2]+(-1,))).mean(0)}
 pp=np.clip(p,1e-7,1-1e-7)
 out={'q_mse':float(np.mean((p-q)**2)),'q_mse_unbiased':float(np.mean((p-a)*(p-b))),
 'bce':float(np.mean(-q*np.log(pp)-(1-q)*np.log1p(-pp))),'q_corr':corr(p,q),
 'naive_oracle_optimistic':float(q.max(-1).mean())}
 for name,prepare,axes in [('state_eta',lambda x:x,(1,3)),('controller_eta',lambda x:x.mean(1,keepdims=True),(2,3)),('geometry_eta',lambda x:x.mean((1,2),keepdims=True),(0,3))]:
  ap,bp,qp,pr=[center(prepare(x),axes) for x in (a,b,q,p)];en=float(np.mean(ap*bp));risk=float(np.mean((pr-ap)*(pr-bp)))
  out[name]={'energy':en,'reliability':corr(ap,bp),'corr':corr(pr,qp),'unbiased_skill':1-risk/en if en>1e-12 else None,'prediction_energy':float(np.mean(pr**2))}
 i,j=np.triu_indices(p.shape[-1],1);d=q[...,i]-q[...,j];pred=p[...,i]-p[...,j];m=np.abs(d)>=.25
 out['eta_pair_order_accuracy']=float(np.mean((np.sign(d[m])==np.sign(pred[m]))+.5*(pred[m]==0))) if m.any() else None
 out.update({k:float(v.mean()) for k,v in arrays.items()});return out,arrays

def evaluate(y,p,prior,seed=20261011,boot=2000):
 out,arrays=stats(y,p,prior);rng=np.random.default_rng(seed);draws=rng.integers(0,y.shape[1],(boot,y.shape[1]))
 for k,a in arrays.items():out[k+'_ci95']=np.percentile(a[draws].mean(1),[2.5,97.5]).tolist()
 # Correlation/skill uncertainty resamples entire physical-state blocks.
 vals=[]
 for ss in draws:
  a=y[:,ss,...,:y.shape[-1]//2].mean(-1);b=y[:,ss,...,y.shape[-1]//2:].mean(-1);pr=p[:,ss]
  a,b,pr=[center(x,(1,3)) for x in (a,b,pr)];en=np.mean(a*b);co=corr(pr,(a+b)/2)
  vals.append([en,np.nan if co is None else co,1-np.mean((pr-a)*(pr-b))/en if en>1e-12 else np.nan])
 vals=np.asarray(vals)
 for j,k in enumerate(['energy','corr','unbiased_skill']):out['state_eta'][k+'_ci95']=np.nanpercentile(vals[:,j],[2.5,97.5]).tolist() if np.isfinite(vals[:,j]).any() else None
 return out,arrays
