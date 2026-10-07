"""Continuous source-only eta controls, source validation selects all settings."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import expit
from .train import data,metrics
from .build_source import SOURCES,OUT,dump,sha

def fit():
    assert not (OUT/'target_predictions.json').exists()
    rows,x,e,y,si=data();tr=np.array([r['split']=='train' for r in rows]);va=~tr
    weight=np.zeros(len(rows))
    for sc in SOURCES:
        ix=np.array([r['scenario']==sc and r['split']=='train' for r in rows]);weight[ix]=1./(3*ix.sum())
    z,inv=np.unique(e[tr],axis=0,return_inverse=True)
    w=np.bincount(inv,weights=weight[tr]);q=np.bincount(inv,weights=weight[tr]*y[tr])/w
    dist=cdist(z,z,'sqeuclidean');dv=cdist(e[va],z,'sqeuclidean');prior=float(weight@y)
    vrows=[r for r in rows if r['split']!='train'];curves=[];best=None
    def score(pred):return float(np.mean([metrics(pred[[r['scenario']==sc for r in vrows]],y[va][[r['scenario']==sc for r in vrows]])['nll'] for sc in SOURCES]))
    for bw in (.05,.1,.2,.4,.8,1.6):
        K=np.exp(-dist/(2*bw*bw))
        for ridge in (.001,.01,.1):
            alpha=solve(K+np.diag(ridge/(w*len(w))),q-prior,assume_a='pos')
            pred=np.clip(prior+np.exp(-dv/(2*bw*bw))@alpha,1e-6,1-1e-6)
            nll=score(pred);curves.append({'bw':bw,'ridge':ridge,'source_validation_nll':nll})
            if best is None or nll<best[0]:best=(nll,bw,ridge,alpha)
    np.savez(OUT/'eta_kernel.npz',z=z,alpha=best[3],bandwidth=best[1],prior=prior)
    X=np.column_stack([e,np.ones(len(e))])
    def obj(beta):
        pred=expit(X@beta);loss=np.sum(weight*(np.logaddexp(0,X@beta)-y*(X@beta)))+.001*np.sum(beta[:3]**2)
        grad=X.T@(weight*(pred-y));grad[:3]+=.002*beta[:3]
        return loss,grad
    opt=minimize(obj,np.zeros(4),jac=True,method='L-BFGS-B')
    np.savez(OUT/'eta_linear.npz',beta=opt.x)
    result={'kernel':{'validation_nll':best[0],'bandwidth':best[1],'ridge':best[2],'sha256':sha(OUT/'eta_kernel.npz')},'linear':{'validation_nll':score(expit(X[va]@opt.x)),'sha256':sha(OUT/'eta_linear.npz')},'search':curves,'target_labels_used':False}
    dump('eta_baselines_frozen.json',result);print(result['kernel']);return result
if __name__=='__main__':fit()
