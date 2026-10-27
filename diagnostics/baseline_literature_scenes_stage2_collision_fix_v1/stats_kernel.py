"""Unmodified v2 bootstrap and contrast mathematical kernels."""
import numpy as np

def bootstrap(M,types,iterations=5000):
    groups=[np.flatnonzero(types==t) for t in sorted(set(types))]
    point=np.mean([np.mean([np.nanmean(M[i],axis=0) for i in group],axis=0) for group in groups],axis=0)
    rng=np.random.default_rng(2026102003);out=[]
    available=[np.flatnonzero(np.isfinite(v).all(-1)) for v in M]
    for _ in range(iterations):
        ts=[]
        for ti in rng.integers(len(groups),size=len(groups)):
            group=groups[ti];cs=[]
            for ci in rng.choice(group,len(group),replace=True):
                ix=available[ci];cs.append(M[ci,rng.choice(ix,len(ix),replace=True)].mean(0))
            ts.append(np.mean(cs,axis=0))
        out.append(np.mean(ts,axis=0))
    return point,np.asarray(out)


def contrast(point,draws,arms,coefficients):
    lower=[];upper=[];ordinary=[]
    for arm,value in coefficients.items():
        j=2*arms.index(arm)
        ordinary.append((j,value))
        lower.append((j if value>0 else j+1,value))
        upper.append((j+1 if value>0 else j,value))
    calc=lambda vector,pairs:sum(vector[...,j]*value for j,value in pairs)
    return dict(delta=float(calc(point,ordinary)),ci95=np.quantile(calc(draws,ordinary),[.025,.975]).tolist(),
        identified_bounds=[float(calc(point,lower)),float(calc(point,upper))],
        unknown_sensitive_ci95=[float(np.quantile(calc(draws,lower),.025)),float(np.quantile(calc(draws,upper),.975))])


