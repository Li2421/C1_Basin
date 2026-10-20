from core import *
from functools import lru_cache
precision()
@lru_cache(None)
def indices(n):
 rng=np.random.default_rng(2026101707);return rng.integers(0,n,(5000,n)),rng.integers(0,32,(5000,n,32))
def draws(x,per_controller=False):
 x=np.asarray(x,float);assert x.ndim==3 and x.shape[1]==32
 c,s=indices(len(x));v=x.mean(-1)[c[:,:,None],s].mean(2);return v if per_controller else v.mean(1)
def quantiles(x):
 x=np.asarray(x,float);good=np.isfinite(x);return dict(CI95=np.quantile(x[good],[.025,.975]).tolist() if good.any() else None,defined_bootstraps=int(good.sum()),undefined_bootstraps=int((~good).sum()))
def ratio(num,den,unknown=None):
 num=np.asarray(num,float);den=np.asarray(den,float);d=draws(den);n=draws(num);v=np.divide(n,d,out=np.full_like(n,np.nan),where=d>0);out=dict(estimate=float(num.sum()/den.sum()) if den.sum() else None,numerator=float(num.sum()),denominator=float(den.sum()),**quantiles(v))
 if unknown is not None:
  u=np.asarray(unknown,float);uu=draws(u);lower=np.divide(n,d+uu,out=np.full_like(n,np.nan),where=d+uu>0);upper=np.divide(n+uu,d+uu,out=np.full_like(n,np.nan),where=d+uu>0);N=num.sum();D=den.sum();U=u.sum();out.update(unknown=int(U),identification_interval=[float(N/(D+U)),float((N+U)/(D+U))] if D+U else None,CI_unknown_envelope=[quantiles(lower)['CI95'][0],quantiles(upper)['CI95'][1]] if np.isfinite(lower).any() else None)
 return out
def mean(x,mask=None):
 x=np.asarray(x,float);valid=np.isfinite(x)
 if mask is not None:valid&=np.asarray(mask,bool)
 return ratio(np.where(valid,x,0),valid.astype(float))
def difference(a,b,mask=None):
 valid=np.isfinite(a)&np.isfinite(b)
 if mask is not None:valid&=mask
 return mean(np.asarray(a)-np.asarray(b),valid)
def outcome_rate(success,unknown):
 lo=mean(np.asarray(success,float));hi=mean(np.asarray(success,float)+unknown);return dict(lower=lo,upper=hi,unknown=int(np.asarray(unknown).sum()),CI_unknown_envelope=[lo['CI95'][0],hi['CI95'][1]])
def paired_success(a,au,b,bu):
 low=np.asarray(a,float)-np.asarray(b,float)-bu;high=np.asarray(a,float)+au-np.asarray(b,float);return dict(lower=mean(low),upper=mean(high),CI_unknown_envelope=[mean(low)['CI95'][0],mean(high)['CI95'][1]])
