"""Outcome-blind prediction, with preprocessing fitted on source controllers only."""
from dataclasses import dataclass
import numpy as np

ARMS = ('safety_baseline','F10','F13','S100','S300','S500','Sprog','RHR')
BASELINE = ARMS.index('S500')


def weights(types):
    types = np.asarray(types)
    return np.asarray([1/np.sum(types==t) for t in types]) / len(set(types))


def best_arm(scores):
    scores = np.asarray(scores)
    choice = scores.argmax(-1)
    return np.where(scores[...,BASELINE] >= scores.max(-1)-1e-12, BASELINE, choice)


@dataclass
class Regression:
    penalty: float = 1.
    pca_dimensions: int | None = None

    def fit(self, x, rates, types):
        x, rates = np.asarray(x,float), np.asarray(rates,float)
        if rates.shape != (len(x),len(ARMS)) or not np.isfinite(rates).all():
            raise ValueError('Need complete per-arm lower success rates for source only')
        finite = np.isfinite(x)
        w = weights(types)
        self.fill = np.sum(np.where(finite,x,0)*w[:,None],0)/np.maximum(np.sum(finite*w[:,None],0),1e-30)
        x = np.where(finite,x,self.fill)
        self.mean = np.sum(x*w[:,None],0)
        self.scale = np.sqrt(np.sum((x-self.mean)**2*w[:,None],0))
        self.scale = np.where(self.scale>1e-6,self.scale,1.)
        z = (x-self.mean)/self.scale
        self.components = None
        if self.pca_dimensions is not None:
            _,_,v = np.linalg.svd(z*np.sqrt(w[:,None]),full_matrices=False)
            self.components = v[:self.pca_dimensions]
            z = z@self.components.T
        target = rates-rates[:,BASELINE,None]
        self.intercept = np.sum(target*w[:,None],0)
        self.beta = np.linalg.solve(z.T@(w[:,None]*z)+self.penalty*np.eye(z.shape[-1]),
                                    z.T@(w[:,None]*(target-self.intercept)))
        self.source_types = sorted(set(map(str,types)))
        return self

    def predict(self,x):
        x = np.asarray(x,float)
        z = (np.where(np.isfinite(x),x,self.fill)-self.mean)/self.scale
        if self.components is not None:
            z = z@self.components.T
        pred = z@self.beta+self.intercept
        pred[:,BASELINE] = 0.
        return pred

    def choose(self,x):
        x=np.asarray(x,float)
        choices=best_arm(self.predict(x))
        return np.where(np.isfinite(x).any(axis=1),choices,BASELINE)

    def to_dict(self):
        result = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(self).items()}
        return dict(model='relative_success_ridge',arms=list(ARMS),baseline='S500',
                    unavailable_probe_rule='all input features unavailable -> S500; no probe rerun',**result)

    @classmethod
    def from_dict(cls,record):
        model=cls(record['penalty'],record['pca_dimensions'])
        for name in ('fill','mean','scale','intercept','beta'):
            setattr(model,name,np.asarray(record[name],float))
        model.components=None if record['components'] is None else np.asarray(record['components'],float)
        model.source_types=record['source_types']
        return model


def tune(x,rates,types,pca_dimensions=None):
    """Only this function sees source labels, never outer held-type labels."""
    types=np.asarray(types)
    history=[]
    if len(set(types))<2:
        return 1.,dict(status='single_source_type_fixed_penalty',history=[])
    for penalty in (.01,.1,1.,10.,100.):
        folds=[]
        for held in sorted(set(types)):
            tr,va=types!=held,types==held
            model=Regression(penalty,pca_dimensions).fit(x[tr],rates[tr],types[tr])
            prediction=model.predict(x[va]);choice=model.choose(x[va])
            value=rates[va][np.arange(va.sum()),choice]
            regret=rates[va].max(1)-value
            target=rates[va]-rates[va,BASELINE,None]
            folds.append(dict(type=str(held),regret=float(regret.mean()),
                              squared_error=float(np.mean((prediction-target)**2)),
                              selected=choice.tolist()))
        history.append(dict(penalty=penalty,folds=folds,
                            mean_regret=float(np.mean([f['regret'] for f in folds])),
                            mean_squared_error=float(np.mean([f['squared_error'] for f in folds]))))
    chosen=min(history,key=lambda v:(round(v['mean_regret'],12),-v['penalty']))
    return chosen['penalty'],dict(history=history,selection='type_equal_inner_regret; tie larger penalty')


def leave_one_type(x,rates,types,pca_dimensions=None):
    x,rates,types=np.asarray(x,float),np.asarray(rates,float),np.asarray(types)
    choices=np.full(len(x),-1,int);folds=[]
    for held in sorted(set(types)):
        tr,va=types!=held,types==held
        penalty,inner=tune(x[tr],rates[tr],types[tr],pca_dimensions)
        model=Regression(penalty,pca_dimensions).fit(x[tr],rates[tr],types[tr])
        choices[va]=model.choose(x[va])
        folds.append(dict(held_type=str(held),source_types=sorted(set(map(str,types[tr]))),
                          source_indices=np.flatnonzero(tr).tolist(),held_indices=np.flatnonzero(va).tolist(),
                          penalty=penalty,inner=inner,model=model.to_dict()))
    return choices,folds


def few_labels(tuning_success,k):
    """controller × tuning state × arm; unknown lower bound supplied by caller."""
    return best_arm(np.asarray(tuning_success)[:,:k].sum(1))


def crossfit_oracle(success,state_ids=None):
    """controller × state × arm × seed. Choice learned on opposite state parity."""
    success=np.asarray(success,float)
    nc,ns,na,nq=success.shape
    if na!=len(ARMS):
        raise ValueError('Oracle menu mismatch')
    parity=np.arange(ns)%2 if state_ids is None else np.asarray(state_ids)%2
    if parity.ndim==1:
        parity=np.broadcast_to(parity,(nc,ns))
    choice=np.empty((nc,ns),int)
    for c in range(nc):
        for fold in (0,1):
            train=parity[c]!=fold;test=parity[c]==fold
            # Empty train halves can occur in bootstrap; use the fixed baseline.
            a=best_arm(success[c,train].mean((0,2))) if train.any() else BASELINE
            choice[c,test]=a
    value=success[np.arange(nc)[:,None],np.arange(ns)[None,:],choice]
    return value,choice


def paired_interval(a,b,types,seed=202610180501,type_equal=True,replicates=5000):
    """type -> controller -> state paired bootstrap. Seed replicates stay clustered."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=3:
        raise ValueError('Paired controller × state × seed arrays required')
    types=np.asarray(types);groups=sorted(set(types));rng=np.random.default_rng(seed)
    def mean(delta):
        per=delta.mean((1,2))
        return float(np.mean([per[types==t].mean() for t in groups])) if type_equal else float(per.mean())
    samples=[];delta=(a-b).mean(-1)
    for _ in range(replicates):
        if type_equal:
            draws=[]
            for t in rng.choice(groups,len(groups),replace=True):
                pool=np.flatnonzero(types==t);cc=rng.choice(pool,len(pool),replace=True)
                ss=rng.integers(a.shape[1],size=(len(cc),a.shape[1]))
                draws.append(delta[cc[:,None],ss].mean())
            samples.append(np.mean(draws))
        else:
            cc=rng.integers(len(a),size=len(a));ss=rng.integers(a.shape[1],size=(len(cc),a.shape[1]))
            samples.append(delta[cc[:,None],ss].mean())
    return dict(point_pp=100*mean(a-b),CI95_pp=(100*np.quantile(samples,[.025,.975])).tolist(),
                replicates=replicates,type_equal=type_equal)


def paired_bounds(a_lo,a_hi,b_lo,b_hi,types,same_cases=None,**kwargs):
    zero=np.zeros_like(a_lo)
    same=np.zeros_like(a_lo,dtype=bool) if same_cases is None else np.broadcast_to(same_cases,np.shape(a_lo))
    lower=paired_interval(np.where(same,0.,a_lo-b_hi),zero,types,**kwargs)
    upper=paired_interval(np.where(same,0.,a_hi-b_lo),zero,types,**kwargs)
    point=paired_interval(a_lo,b_lo,types,**kwargs)
    return dict(point_pp=point['point_pp'],unknown_estimate_bounds_pp=[lower['point_pp'],upper['point_pp']],
                CI95_pp=[lower['CI95_pp'][0],upper['CI95_pp'][1]],
                replicates=point['replicates'],type_equal=point['type_equal'])
