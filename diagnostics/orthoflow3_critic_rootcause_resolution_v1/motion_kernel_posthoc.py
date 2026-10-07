"""Fixed source-selected non-neural diagnostic on already-opened confirmations.

This is explicitly NOT independent confirmation or a model-selection gate.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit
from . import motion_kernel as kernel
from .motion_confirmation_eval import normalized_context,measures

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'motion_kernel_posthoc'
read,write,sha=kernel.read,kernel.write,kernel.sha


def target_inputs(folder,norm,hnorm,condition):
    x=dict(np.load(folder/'entities.npz'))
    state=np.arange(64);shift=np.roll(state,-1)
    h=[]
    for key in ('agents','pairs','obstacles','globals'):
        a=x[key].reshape(64,-1)
        h.append((a-hnorm[key]['center'])/hnorm[key]['scale']/np.sqrt(a.shape[-1]))
    h=np.concatenate(h,-1)/2
    name='alt' if condition=='wrong_controller' else 'held'
    initial=dict(np.load(folder/f'inputs_{name}.npz'))
    response=dict(np.load(folder/f'goal_both_{name}.npz'))
    c=normalized_context(initial,response,norm)
    blocks=(c[:,:24],c[:,24:88].reshape(128,4,16).mean(1),c[:,88:104],c[:,104:])
    c=np.concatenate([a/np.sqrt(a.shape[-1]) for a in blocks],-1)/2
    pairs=read(folder/'pairs.json');e=np.array([r['eta'] for r in pairs])
    e=(e-norm['eta_center'])/norm['eta_scale']
    if condition=='joint_state_context_shuffle':
        c=c.reshape(64,2,-1)[shift].reshape(128,-1)
    if condition in ('joint_state_context_shuffle','state_shuffle'):h=h[shift]
    return np.concatenate((np.repeat(h,2,axis=0),c,e/np.sqrt(3)),-1)/np.sqrt(3),e


def main():
    OUT.mkdir(exist_ok=True)
    selection=read(kernel.OUT/'selection_frozen.json')
    assert selection['selected']['model']=='kernel_w0.5_r0.001'
    d,x,tr,va,norm=kernel.base.load(list(range(12)))
    features,hnorm=kernel.features(d,x,tr)
    ci,ii=np.repeat(np.arange(16),len(tr)),np.tile(tr,16)
    a=features['correct'][ci,ii]
    q=d['success'][ci,ii]/(d['success'][ci,ii]+d['failure'][ci,ii])
    offset,weights=kernel.prior(d,ci,ii)
    ds=cdist(a,a,'sqeuclidean');variance=float(np.median(ds[np.triu_indices(len(a),1)]))*.5**2
    k=np.exp(-ds/(2*variance));ev,v=eigh(k,check_finite=False,driver='evr')
    keep=ev>max(float(ev[-1])*1e-9,1e-10)
    phi=v[:,keep]*np.sqrt(ev[keep]);projection=v[:,keep]/np.sqrt(ev[keep])
    result=minimize(kernel.objective,np.zeros(phi.shape[1]),args=(phi,offset[ii],q,.001),jac=True,
        method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12,gtol=1e-6))
    assert result.success,result.message
    alpha=projection@result.x
    np.savez_compressed(OUT/'source_model.npz',alpha=alpha,features=a,variance=variance,eta_prior_weights=weights)
    write(OUT/'fit_frozen.json',dict(dataset_sha256=sha(kernel.base.OUT/'dataset.npz'),
        selection_sha256=sha(kernel.OUT/'selection_frozen.json'),model_sha256=sha(OUT/'source_model.npz'),
        target_labels_used_for_fitting=False,scope='Posthoc diagnostic,not an independent claim. No tuning or additional parameter variants on targets.'))
    rows=[];neighbors=[];scores={};distances=[]
    for target in (88132,88133,88134,88135):
        folder=ROOT/f'motion_independent_confirmation_{target}'
        target_features={c:target_inputs(folder,norm,hnorm,c) for c in kernel.CONDITIONS}
        predictions={}
        for condition,(xx,ee) in target_features.items():
            delta=cdist(xx,a,'sqeuclidean')
            zz=np.exp(-delta/(2*variance))@alpha+np.column_stack((np.ones(128),ee))@weights
            predictions[condition]=zz.reshape(64,2)
            if condition=='correct':
                nn=np.argsort(delta,axis=1,kind='stable')[:,:20]
                for j in range(128):
                    for rank,t in enumerate(nn[j],1):
                        neighbors.append(dict(target=target,state=j//2,eta_index=j%2,rank=rank,
                            source_controller=int(ci[t]),source_state=int(d['state_index'][ii[t]]),
                            source_eta_index=int(d['eta_index'][ii[t]]),source_Q=float(q[t]),
                            distance=float(np.sqrt(delta[j,t]))))
                distances.append(dict(target=target,nearest_median=float(np.median(np.sqrt(delta.min(1)))),
                    nearest_q90=float(np.quantile(np.sqrt(delta.min(1)),.9))))
        # Predictions fixed under the source-selected model, then read opened labels.
        data=dict(np.load(folder/'dataset.npz'));s,f=[data[k].reshape(64,2) for k in ('success','failure')]
        for condition,zz in predictions.items():
            scores[f'{target}__{condition}']=zz
            rows.append(dict(target=target,condition=condition,posthoc=True,**measures(zz,s,f)))
    np.savez_compressed(OUT/'posthoc_predictions.npz',**scores)
    kernel.base.csvwrite(OUT/'metrics.csv',rows);kernel.base.csvwrite(OUT/'nearest_TRAIN.csv',neighbors)
    write(OUT/'audit.json',dict(scope='Opened-target diagnostic only;do not select or declare a new model confirmed from these results.',
        target_labels_used_for_fitting=False,model_changed_after_target_scoring=False,new_rollouts=0,
        input_support=distances,source_selected_model=selection['selected']))
    print([r for r in rows if r['condition']=='correct'],flush=True)


if __name__=='__main__':main()
