"""Reporting means/CIs for the two observed arms; missing methods stay missing."""
from common import *
def methods(z,seed):
 lo=z['lower'];hi=z['upper'];rng=np.random.default_rng(seed);bs=[]
 for _ in range(5000):
  cc=rng.integers(14,size=14);ss=rng.integers(16,size=(14,16));bs.append(np.stack([lo[cc[:,None],ss].mean((0,1,3)),hi[cc[:,None],ss].mean((0,1,3))],1))
 bs=np.asarray(bs);out={}
 for i,a in enumerate(['F10','S500']):out[a]=dict(success_pct=[lo[:,:,i].mean()*100,hi[:,:,i].mean()*100],success_95CI_pct=[np.quantile(bs[:,i,0],.025)*100,np.quantile(bs[:,i,1],.975)*100],events_pct={k:z[k][:,:,i].mean()*100 for k in ['collision','timeout','unknown']},mean_regret_pp=None,max_regret_pp=None,scope='Two-arm16state seed0 descriptive control; no fullmenu oracle, nominal/safety, catastrophe/harm ormain claim')
 return clean(out)
