"""Predeclared descriptive median-of-controller-max and paired bootstrap."""
from common import *

def bounds(lo,hi):
 return np.stack([lo.mean((1,3)).max(1),hi.mean((1,3)).max(1)],1)
def ci(a):
 z=np.asarray(a);return [float(np.quantile(z[:,0],.025)*100),float(np.quantile(z[:,1],.975)*100)]
def summarize(lo,hi,seed,base=None):
 nc,ns=lo.shape[:2];assert lo.shape==hi.shape==(14,16,2,1);per=bounds(lo,hi);med=np.median(per,axis=0);bs=[];delta=[];within=[];rng=np.random.default_rng(seed)
 for _ in range(5000):
  cc=rng.integers(nc,size=nc);ss=rng.integers(ns,size=(nc,ns));v=bounds(lo[cc[:,None],ss],hi[cc[:,None],ss]);m=np.median(v,axis=0);bs.append(m)
  if base is not None:
   b=bounds(base[0][cc[:,None],ss],base[1][cc[:,None],ss]);bm=np.median(b,axis=0);delta.append([m[0]-bm[1],m[1]-bm[0]]);within.append(np.median(np.stack([v[:,0]-b[:,1],v[:,1]-b[:,0]],1),axis=0))
 out=dict(median_max_success_pct=(med*100).tolist(),median_max_95CI_pct=ci(bs),controller_max_success_bounds_pct=(per*100).tolist(),arms_success_bounds_pct=np.stack([lo.mean((1,3)),hi.mean((1,3))],-1)*100,bootstrap=5000)
 if base is not None:
  b=bounds(*base);bm=np.median(b,axis=0);out.update(paired_difference_of_medians_pp=[float((med[0]-bm[1])*100),float((med[1]-bm[0])*100)],paired_difference_95CI_pp=ci(delta),median_within_controller_difference_pp=(np.median(np.stack([per[:,0]-b[:,1],per[:,1]-b[:,0]],1),axis=0)*100).tolist(),median_within_controller_difference_95CI_pp=ci(within))
 return clean(out)

def test():
 a=np.zeros((14,16,2,1));b=a.copy();a[:,:,0]=.5;b[:,:,0]=.5;assert np.all(bounds(a,b)==.5)
 b[:,:,1]=1;v=bounds(a,b);assert np.all(v[:,0]==.5) and np.all(v[:,1]==1)
 print('Capability max/median/unknown tests PASS')
if __name__=='__main__':test()
