"""Outcome-free feature transforms, nested leave-one-scene tuning and fixed menu selection."""
from common import *
import numpy as np

LAMBDAS=(.1,1.,10.,100.)
KS=(1,2,3,5)


def design(features,etas):
    f=np.asarray(features,float);e=np.asarray(etas,float)
    return np.column_stack([np.repeat(f[None],len(e),axis=0),e,e*e,(f[None,:,None]*e[:,None,:]).reshape(len(e),-1)])


def ridge_fit(source_features,source_tables,etas,penalty):
    matrices=[design(f,etas)for f in source_features]
    X=np.vstack(matrices);y=np.asarray(source_tables,float).reshape(-1)
    mu=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-12,sd,1.)
    Z=(X-mu)/sd;ym=y.mean()
    # Each scene supplies the same49 rows: ordinary squared loss is scene equal.
    coef=np.linalg.solve(Z.T@Z+penalty*np.eye(Z.shape[1]),Z.T@(y-ym))
    return dict(mean=mu.tolist(),scale=sd.tolist(),coefficient=coef.tolist(),intercept=float(ym),penalty=float(penalty))


def ridge_predict(source_features,source_tables,target_features,etas,penalty):
    state=ridge_fit(source_features,source_tables,etas,penalty)
    return ((design(target_features,etas)-state['mean'])/state['scale'])@np.asarray(state['coefficient'])+state['intercept']


def knn_predict(source_features,source_tables,target_features,k):
    X=np.asarray(source_features,float);mu=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-12,sd,1.)
    distance=np.sum(((X-target_features)/sd)**2,axis=1)
    order=np.argsort(distance,kind='stable')[:min(k,len(X))]
    return np.asarray(source_tables)[order].mean(0)


def select(features,tables,target,etas):
    f=np.asarray(features,float);q=np.asarray(tables,float)
    ridge_cv={};knn_cv={}
    for penalty in LAMBDAS:
        scores=[]
        for i in range(len(f)):
            train=[j for j in range(len(f))if j!=i]
            choice=int(np.argmax(ridge_predict(f[train],q[train],f[i],etas,penalty)))
            scores.append(float(q[i,choice]))
        ridge_cv[str(penalty)]=float(np.mean(scores))
    for k in KS:
        scores=[]
        for i in range(len(f)):
            train=[j for j in range(len(f))if j!=i]
            choice=int(np.argmax(knn_predict(f[train],q[train],f[i],k)))
            scores.append(float(q[i,choice]))
        knn_cv[str(k)]=float(np.mean(scores))
    # Candidate ties retain the canonical safety,eta0..15,sE00..31 order.
    best=max(ridge_cv.values());penalty=next(p for p in LAMBDAS if abs(ridge_cv[str(p)]-best)<=1e-12)
    best=max(knn_cv.values());k=next(k for k in KS if abs(knn_cv[str(k)]-best)<=1e-12)
    state=ridge_fit(f,q,etas,penalty)
    rp=((design(target,etas)-state['mean'])/state['scale'])@np.asarray(state['coefficient'])+state['intercept']
    kp=knn_predict(f,q,target,k)
    return dict(ridge=int(np.argmax(rp)),knn=int(np.argmax(kp)),global_fixed=int(np.argmax(q.mean(0))),
                eta13=14,safety=0,penalty=penalty,k=k,ridge_inner_cv=ridge_cv,knn_inner_cv=knn_cv,
                ridge_predictions=rp.tolist(),knn_predictions=kp.tolist(),ridge_model=state,
                knn_model=dict(features=f.tolist(),success_tables=q.tolist(),k=k,
                    mean=f.mean(0).tolist(),scale=np.where(f.std(0)>1e-12,f.std(0),1.).tolist()))


def crossfit(y,indices=None):
    """y[controller,state,action], paired split membership fixed before bootstrap."""
    n=y.shape[1];index=np.arange(n)if indices is None else np.asarray(indices)
    train0=np.flatnonzero(index%2==0);train1=np.flatnonzero(index%2==1)
    if not len(train0)or not len(train1):return None
    q0=y[:,train0,:].mean((0,1));q1=y[:,train1,:].mean((0,1))
    a0=int(np.argmax(q0));a1=int(np.argmax(q1))
    score0=y[:,train1,a0].mean();score1=y[:,train0,a1].mean()
    # Equal weight to the two swapped evaluations (state counts equal in the design).
    return dict(rate=float((score0+score1)/2),actions=[a0,a1])


def calibration(y,seed=2026102413,repeats=200,draws=None):
    n=y.shape[1];rng=np.random.default_rng(seed);rates=[];choices=[]
    if draws is None:
        draws=[dict(actions=sorted(rng.choice(np.arange(17,49),16,replace=False).tolist()),
                    states=rng.choice(np.arange(n),4,replace=False).tolist())for _ in range(repeats)]
    for draw in draws:
        sample=np.asarray(draw['states']);menu=np.asarray(draw['actions']);rest=np.asarray([i for i in range(n)if i not in set(sample)])
        table=y[:,sample,:].mean((0,1));a=int(menu[np.argmax(table[menu])]);rates.append(float(y[:,rest,a].mean()));choices.append(a)
    return dict(rate=float(np.mean(rates)),choices=choices,draws=draws,rates=rates)
