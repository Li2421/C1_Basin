"""Frozen non-neural diagnostic on identical12-controller source support."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.spatial.distance import cdist
from .function_models import OUT,load,read,write
from .simple_response import pooled,loss
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha

def features(x,si,context,agent,norm):
    h=(pooled(x)-np.array(norm['h_center']))/np.array(norm['h_scale'])
    c=(context-np.array(norm['context_center']))/np.array(norm['context_scale'])
    a=(agent-np.array(norm['agent_center']))/np.array(norm['agent_scale'])
    cc=np.concatenate([c,a.mean(-2)],-1)
    return np.concatenate([h[si]/np.sqrt(h.shape[-1]),cc/np.sqrt(cc.shape[-1])],-1)

def predict(train_z,train_eta,train_q,z,eta,k):
    output=np.empty(len(z))
    for e in (10,15):
        tr=np.flatnonzero(train_eta==e);te=np.flatnonzero(eta==e)
        dist=cdist(z[te],train_z[tr],metric='sqeuclidean');ii=np.argsort(dist,axis=1)[:,:k];dd=np.take_along_axis(dist,ii,axis=1)
        width=np.maximum(np.median(dd,axis=1,keepdims=True),1e-10);w=np.exp(-dd/width)
        output[te]=(w*train_q[tr[ii]]).sum(1)/w.sum(1)
    return output

def main():
    dest=OUT/'local_diagnostic';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():raise FileExistsError('Diagnostic frozen')
    d,x,fit,val,norm=load();h=pooled(x);fs=np.unique(d['state_index'][fit])
    norm.update(h_center=h[fs].mean(0).tolist(),h_scale=np.maximum(h[fs].std(0),.05).tolist())
    z=np.array([features(x,d['state_index'],d['context'][c],d['agent_response'][c],norm) for c in range(12)])
    q=d['success']/(d['success']+d['failure']);train=z[:,fit].reshape(-1,z.shape[-1]);tq=q[:,fit].ravel();te=np.tile(d['eta_index'][fit],12)
    vz=z[:,val].reshape(-1,z.shape[-1]);ve=np.tile(d['eta_index'][val],12);vq=q[:,val].ravel()
    choices=[(float(loss(predict(train,te,tq,vz,ve,k),vq).mean()),k) for k in (5,20,100)];best=min(choices)
    np.savez_compressed(dest/'knn.npz',train_z=train,train_q=tq,train_eta=te)
    write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(k=best[1],source_VAL_NLL=best[0],candidates=choices,
        parameter_selection='Source12controller16VALfamilies only; no targetlabels',
        scope='Same-exact-eta localregression diagnostic; doesNOTestablishunseeneta/continuousspacegeneralization',
        data_sha256=sha(OUT/'dataset.npz'),new_rollouts=0))
    print(best)

if __name__=='__main__':main()
