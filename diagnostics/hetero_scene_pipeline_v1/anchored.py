"""Training-scene priors, leave-one-candidate bandwidth and bounded logits."""
from common import *
import numpy as np

BANDWIDTHS=(.05,.1,.2,.3,.5)
LAMBDAS=(0.,.25,.5,.75,1.)
CENTER=np.asarray([1.,0.,.75]);RADIUS=np.asarray([1.,1.5,.75])

def prior(tables,sources,etas):
    p0=np.mean([np.asarray(tables[n]['y'],float).mean((0,1))for n in sources],axis=0)
    z=(np.asarray(etas)-CENTER)/RADIUS;dist=((z[:,None]-z[None])**2).sum(-1);losses={}
    for b in BANDWIDTHS:
        logw=-dist/(2*b*b);np.fill_diagonal(logw,-np.inf);logw-=logw.max(1,keepdims=True)
        w=np.exp(logw);prediction=(w@p0)/w.sum(1);losses[str(b)]=float(np.mean((prediction-p0)**2))
    b=min(BANDWIDTHS,key=lambda b:(losses[str(b)],b))
    return dict(p0=p0.tolist(),bandwidth=b,leave_one_candidate_squared_errors=losses,sources=sources,
        best_index=int(np.argmax(p0)),unknown_as_failure=True,all_source_states=True)

def proposal_priors(etas,reference_etas,priors):
    z=(np.asarray(etas)-CENTER)/RADIUS;ref=(np.asarray(reference_etas)-CENTER)/RADIUS
    logw=-((z[:,None]-ref[None])**2).sum(-1)/(2*priors['bandwidth']**2);logw-=logw.max(1,keepdims=True)
    w=np.exp(logw)
    return(w@np.asarray(priors['p0']))/w.sum(1)

def logit(p):
    p=np.clip(np.asarray(p),.02,.98)
    return np.log(p)-np.log1p(-p)

def predictions(raw,p0,shrinkage,mode='full'):
    if mode=='no_prior':return np.asarray(raw)
    return logit(p0)+shrinkage*3*np.tanh(np.asarray(raw)/3)

def sigmoid(x):
    x=np.asarray(x);return np.exp(-np.logaddexp(0,-x))
