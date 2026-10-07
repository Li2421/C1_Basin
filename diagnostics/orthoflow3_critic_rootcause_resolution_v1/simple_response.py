"""Simple source-only controller-transfer controls; scipy, no added packages."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from .response_horizon import OUT,SOURCE,NAMES,load,read,write,csvwrite
DEST=OUT/'simple_controls'

def pooled(x):
    blocks=[]
    for key in ('agents','pairs','obstacles'):
        a=x[key]
        axes=tuple(range(1,a.ndim-1))
        blocks.extend([a.mean(axes),a.std(axes)])
    blocks.append(x['globals'])
    return np.concatenate(blocks,-1)

def loss(p,q):
    p=np.clip(p,1e-6,1-1e-6);return -q*np.log(p)-(1-q)*np.log1p(-p)

def main():
    DEST.mkdir(exist_ok=True);rows=[];decisions=[];saved={};settings=[]
    for held in range(4):
      controllers=[c for c in range(4) if c!=held]
      for horizon in (20,80):
        d,x,fit,val,norm=load(horizon,controllers);h=pooled(x);sidx=d['state_index']
        hs=np.unique(sidx[fit]);h=(h-h[hs].mean(0))/np.maximum(h[hs].std(0),.05)
        ctx=np.concatenate([d['input_context'][...,:24],d['input_context'][...,24:].reshape(4,len(sidx),4,16).mean(2)],-1)
        # Equal group scale avoids high-dimensional state blocks dominating distance.
        z=np.concatenate([np.broadcast_to(h[sidx],(4,len(sidx),h.shape[-1]))/np.sqrt(h.shape[-1]),ctx/np.sqrt(ctx.shape[-1])],-1)
        et=d['normalized_eta'];eta=np.broadcast_to(et,(4,*et.shape))
        design=np.concatenate([np.ones((4,len(sidx),1)),eta,z,(z[...,None]*eta[:,:,None,:]).reshape(4,len(sidx),-1)],-1)
        q=d['success']/(d['success']+d['failure']);a=design[controllers][:,fit].reshape(-1,design.shape[-1]);b=q[controllers][:,fit].ravel()
        av=design[controllers][:,val].reshape(-1,design.shape[-1]);bv=q[controllers][:,val].ravel()
        choices=[]
        for ridge in (.001,.01,.1,1.):
            def fun(w):
                zz=a@w;p=expit(zz);pen=w.copy();pen[0]=0
                v=np.mean(np.logaddexp(0,zz)-b*zz)+.5*ridge*(pen@pen)
                g=a.T@(p-b)/len(b)+ridge*pen
                return v,g
            result=minimize(fun,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-7})
            choices.append((float(loss(expit(av@result.x),bv).mean()),ridge,result.x,result.success))
        best=min(choices,key=lambda r:r[0]);preds={'ridge':expit(design[held,val]@best[2])}
        settings.append(dict(held=NAMES[held],horizon=horizon,method='ridge',setting=best[1],source_VAL_NLL=best[0],optimizer_converged=bool(best[3])))
        np.savez_compressed(DEST/f'linear_H{horizon}_held{held}.npz',weights=best[2],normalization=np.array([norm],object))
        source_z=z[controllers][:,fit].reshape(-1,z.shape[-1]);source_q=q[controllers][:,fit].ravel()
        source_e=np.tile(d['eta_index'][fit],3);val_e=np.tile(d['eta_index'][val],3)
        val_z=z[controllers][:,val].reshape(-1,z.shape[-1])
        def local(target_z,target_e,k):
            output=np.empty(len(target_z))
            for ei in (10,15):
                tr=np.flatnonzero(source_e==ei);te=np.flatnonzero(target_e==ei)
                dist=cdist(target_z[te],source_z[tr],metric='sqeuclidean');ix=np.argsort(dist,axis=1)[:,:k]
                dd=np.take_along_axis(dist,ix,axis=1);scale=np.maximum(np.median(dd,axis=1,keepdims=True),1e-10)
                weights=np.exp(-dd/scale);output[te]=(weights*source_q[tr[ix]]).sum(1)/weights.sum(1)
            return output
        knn=[(float(loss(local(val_z,val_e,k),bv).mean()),k) for k in (5,20,100)]
        kv=min(knn);preds['knn']=local(z[held,val],d['eta_index'][val],kv[1])
        settings.append(dict(held=NAMES[held],horizon=horizon,method='knn',setting=kv[1],source_VAL_NLL=kv[0],optimizer_converged=True))
        # Strong exact-eta training-mean comparator; no target outcomes.
        preds['eta_mean']=np.array([b[source_e==e].mean() for e in d['eta_index'][val]])
        true=q[held,val].reshape(-1,2);good=d['standard_success'][held,val].reshape(-1,2)>=15
        bad=d['standard_failure'][held,val].reshape(-1,2)>=2;rr=np.arange(len(true))
        for method,p in preds.items():
            p=p.reshape(-1,2);ch=p.argmax(1);delta=p[:,0]-p[:,1]
            rows.append(dict(held=NAMES[held],horizon=horizon,method=method,NLL=float(loss(p,true).mean()),MAE=float(abs(p-true).mean()),
                B15=int(good[rr,ch].sum()),oracle_B15=int(good.any(1).sum()),unknown=int((~good[rr,ch]&~bad[rr,ch]).sum()),
                selected_Q=float(true[rr,ch].mean()),contrast_correlation=float(np.corrcoef(delta,true[:,0]-true[:,1])[0,1]) if delta.std()>1e-8 else None))
            saved[f'{held}__{horizon}__{method}']=p
            for j in rr:decisions.append(dict(held=NAMES[held],horizon=horizon,method=method,state_index=int(sidx[val][2*j]),
                eta_index=(10,15)[ch[j]],predicted=float(p[j,ch[j]]),Q=float(true[j,ch[j]]),B15=bool(good[j,ch[j]])))
    csvwrite(DEST/'metrics.csv',rows);csvwrite(DEST/'selection_settings.csv',settings);csvwrite(DEST/'decisions.csv',decisions)
    np.savez_compressed(DEST/'predictions.npz',**saved);write(DEST/'audit.json',dict(new_rollouts=0,target_confirmation_labels_used=False,
        scope='Source-only controller holdout diagnosis; same4source functions,32heldfamilies; no independent confirmation claim'))
    print(rows)

if __name__=='__main__':main()
