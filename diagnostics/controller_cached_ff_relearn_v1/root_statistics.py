"""State-cluster uncertainty for the root-cause evidence; no fitting or rollout."""
import numpy as np
from .study import HERE, read, write


def run():
    folder=HERE/'extended128';d=np.load(folder/'labels.npz')
    q=np.load(folder/'predictions_branch_nominal.npz')['matched']
    boot=np.load(folder/'bootstrap.npz')['indices'];n=len(q)
    truth=d['success']/np.maximum(d['success']+d['failure'],1)
    a=q[:,1]-q[:,0];b=truth[:,1]-truth[:,0]
    samples={key:[] for key in ('delta','eta_centered_delta','state_eta_centered_delta')}
    def corr(x,y):
        x=x.reshape(len(x),-1);y=y.reshape(len(y),-1)
        x=x-x.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True)
        return np.sum(x*y,-1)/np.sqrt(np.sum(x*x,-1)*np.sum(y*y,-1))
    for begin in range(0,len(boot),500):
        ii=boot[begin:begin+500];pa=a[ii];tb=b[ii]
        samples['delta'].append(corr(pa,tb))
        pa=pa-pa.mean(-1,keepdims=True);tb=tb-tb.mean(-1,keepdims=True)
        samples['eta_centered_delta'].append(corr(pa,tb))
        pa=pa-pa.mean(1,keepdims=True);tb=tb-tb.mean(1,keepdims=True)
        samples['state_eta_centered_delta'].append(corr(pa,tb))
    samples={k:np.concatenate(v) for k,v in samples.items()}
    lower=d['success']/4;upper=(d['success']+d['unknown'])/4
    selected=q.argmax(-1);row=np.arange(n)[:,None];controller=np.arange(2)[None,:]
    value=lower[row,controller,selected].mean(-1)
    conservative=value[boot].mean(-1)-upper[boot].mean(1).max(-1).mean(-1)
    samples['selected_lower_minus_reselected_controller_fixed_upper']=conservative
    intervals={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}
    write(folder/'root_uncertainty.json',dict(ci95=intervals,
        selected_unknown_seeds=int(d['unknown'][row,controller,selected].sum()),
        physical_states=n,bootstrap_replicates=len(boot),controllers_seen_in_training=2,
        interpretation='Physical-state generalization conditional on two fixed controller programs; no controller-population inference'))
    np.savez_compressed(folder/'root_uncertainty_bootstrap.npz',**samples)
    print(intervals,flush=True)


if __name__=='__main__':run()
