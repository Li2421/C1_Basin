"""One small-capacity physical interaction control, source crossfit only.

Penalized observed-count likelihood, no ranking labels or controller ID inputs.
Ridge strengths are selected on inner source families, never outer folds.
"""
import numpy as np
from scipy.optimize import minimize
from .experiment import OUT,read,write,csvwrite,metrics

KINDS=('eta_quadratic','context_linear','context_interaction','state_context_interaction')
ALPHAS=(.01,.1,1.,10.)

def state_features(x):
    am=x['agent_mask']>0;n=am.shape[1]
    masks={'agents':am,'pairs':am[:,:,None]&am[:,None,:]&~np.eye(n,dtype=bool)[None],
           'obstacles':am[:,:,None]&(x['obstacle_mask'][:,None,:]>0)}
    features=[]
    for k,mask in masks.items():
        a=x[k];axis=tuple(range(1,a.ndim-1))
        mean=np.sum(a*mask[...,None],axis=axis)/np.maximum(mask.sum(axis=axis),1)[:,None]
        maximum=np.max(np.where(mask[...,None],a,-np.inf),axis=axis)
        minimum=np.min(np.where(mask[...,None],a,np.inf),axis=axis)
        features.extend((mean,maximum,minimum))
    return np.concatenate(features+[x['globals']],axis=1)

def train():
    d=dict(np.load(OUT/'source_data.npz'));x=dict(np.load(OUT/'source_entities.npz'));h=state_features(x)
    protocol=read(OUT/'protocol.json');allmetrics=[];allrows=[];selection=[]
    for fold in range(3):
        sp=protocol['folds'][fold]
        chosen=np.isin(d['eta_index'],[10,15])
        fi=np.flatnonzero(np.isin(d['state_index'],sp['fit'])&chosen)
        iv=np.flatnonzero(np.isin(d['state_index'],sp['inner'])&chosen)
        te=np.flatnonzero(np.isin(d['state_index'],sp['test'])&chosen)
        eta=d['eta'].astype(float);ctx=d['context'].astype(float)
        def scaled(a,fitvalues):
            center=fitvalues.mean(0);scale=fitvalues.std(0);active=scale>1e-6
            return ((a-center)/np.maximum(scale,1e-4))[...,active]
        e=scaled(eta,eta[fi]);c=scaled(ctx,ctx[:,fi].reshape(-1,24));hh=scaled(h,h[sp['fit']])
        estates=np.broadcast_to(e,(2,*e.shape))
        hstates=np.broadcast_to(hh[d['state_index']],(2,len(eta),hh.shape[-1]))
        for kind in KINDS:
            feature=np.concatenate([estates,estates**2],axis=-1)
            if kind!='eta_quadratic':feature=np.concatenate([feature,c],axis=-1)
            if kind=='state_context_interaction':feature=np.concatenate([feature,hstates],axis=-1)
            if kind.endswith('interaction'):
                base=c if kind=='context_interaction' else np.concatenate([c,hstates],axis=-1)
                interaction=(base[...,None]*estates[...,None,:]).reshape(2,len(eta),-1)
                feature=np.concatenate([feature,interaction],axis=-1)
            # Unit scale columns, retain one explicit unpenalized intercept.
            a=feature[:,fi].reshape(-1,feature.shape[-1]);mean=a.mean(0);std=a.std(0);active=std>1e-6
            F=((feature-mean)/np.maximum(std,1e-4))[...,active]
            F=np.concatenate([np.ones((*F.shape[:-1],1)),F],axis=-1)
            xx=F[:,fi].reshape(-1,F.shape[-1]);s=d['success'][:,fi].ravel();f=d['failure'][:,fi].ravel();n=s+f
            best=None
            for alpha in ALPHAS:
                def fun(w):
                    z=xx@w;sigmoid=1/(1+np.exp(-np.clip(z,-700,700)))
                    loss=np.sum(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/sum(n)+.5*alpha*np.sum(w[1:]**2)
                    grad=xx.T@(n*sigmoid-s)/sum(n);grad[1:]+=alpha*w[1:]
                    return loss,grad
                opt=minimize(fun,np.zeros(F.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':600,'ftol':1e-11,'gtol':1e-7})
                assert np.isfinite(opt.fun) and np.linalg.norm(opt.jac)<.002
                val,_=metrics(F[:,iv]@opt.x,d,iv)
                if best is None or val['NLL']<best[0]:best=(val['NLL'],alpha,opt.x.copy())
            nll,alpha,w=best
            state_order=sorted(sp['test']);mp={s:state_order[(j+1)%len(state_order)] for j,s in enumerate(state_order)}
            lookup={(int(d['state_index'][i]),int(d['eta_index'][i])):i for i in te}
            swapped=np.array([lookup[(mp[int(d['state_index'][i])],int(d['eta_index'][i]))] for i in te])
            # All physical feature groups swapped together (same eta/controller).
            for condition,z in [('correct',F[:,te]@w),('joint_state_context_shuffle',F[:,swapped]@w),('wrong_controller',F[::-1,te]@w)]:
                met,rows=metrics(z,d,te)
                allmetrics.append(dict(kind=kind,fold=fold,condition=condition,**met))
                allrows.extend(dict(kind=kind,fold=fold,condition=condition,**r) for r in rows)
            selection.append(dict(kind=kind,fold=fold,alpha=alpha,inner_NLL=nll,parameters=len(w)))
    csvwrite(OUT/'compact_metrics.csv',allmetrics);csvwrite(OUT/'compact_decisions.csv',allrows)
    write(OUT/'compact_selected.json',selection)
    print(selection)
    for kind in KINDS:
      for condition in ('correct','joint_state_context_shuffle','wrong_controller'):
        r=[m for m in allmetrics if m['kind']==kind and m['condition']==condition]
        print(kind,condition,'NLL',np.average([m['NLL'] for m in r],weights=[m['n_pairs'] for m in r]),'B15',sum(m['B15'] for m in r),'strong_correct',sum(m['strong_correct'] for m in r))

if __name__=='__main__':train()
