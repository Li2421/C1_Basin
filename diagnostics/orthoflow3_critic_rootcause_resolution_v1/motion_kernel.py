"""Source-only convex/local fit discriminator. No simulator or target reads."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from . import motion_factorial_train as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'motion_kernel_source'
RULE = ROOT/'motion_kernel_protocol.json'
CONDITIONS = ('correct', 'wrong_controller', 'joint_state_context_shuffle', 'state_shuffle')
read, write, sha = base.read, base.write, base.sha


def features(d, x, tr):
    states = np.unique(d['state_index'][tr])
    pieces, norms = [], {}
    for key in ('agents', 'pairs', 'obstacles', 'globals'):
        a = x[key].reshape(len(x[key]), -1).astype(float)
        c, s = a[states].mean(0), np.maximum(a[states].std(0), .05)
        norms[key] = dict(center=c.tolist(), scale=s.tolist())
        pieces.append((a-c)/s/np.sqrt(a.shape[1]))
    h = np.concatenate(pieces, -1)/2
    v = d['input_context'].astype(float)
    # Exactly the same agent-response mean as the existing neural forward.
    groups = (v[..., :24], v[..., 24:88].reshape(16, 160, 4, 16).mean(-2),
              v[..., 88:104], v[..., 104:120])
    c = np.concatenate([a/np.sqrt(a.shape[-1]) for a in groups], -1)/2
    e = d['normalized_eta'].astype(float)/np.sqrt(3)
    out = {}
    for condition in CONDITIONS:
        # Shift whole physical families, preserve exact eta position.
        shift = np.arange(160)
        for tag in ('train', 'validation'):
            ii = np.flatnonzero(d['split'] == tag)
            shift[ii] = np.roll(ii.reshape(-1, 2), -1, axis=0).ravel()
        hi = shift if condition in ('joint_state_context_shuffle', 'state_shuffle') else np.arange(160)
        cc = c[:, shift] if condition == 'joint_state_context_shuffle' else c
        if condition == 'wrong_controller':
            cc = c[(np.arange(16)+1) % 12]
        hh = np.broadcast_to(h[d['state_index'][hi]][None], (16, 160, h.shape[1]))
        ee = np.broadcast_to(e[None], (16, 160, 3))
        out[condition] = np.concatenate((hh, cc, ee), -1)/np.sqrt(3)
        assert np.isfinite(out[condition]).all()
    return out, norms


def prior(d, ci, ii):
    q = d['success'][ci, ii]/(d['success'][ci, ii]+d['failure'][ci, ii])
    e = d['normalized_eta']
    unique, inverse = np.unique(e[ii], axis=0, return_inverse=True)
    p = np.array([(q[inverse == j].sum()+.5)/(np.sum(inverse == j)+1) for j in range(len(unique))])
    weights = np.linalg.lstsq(np.column_stack((np.ones(len(unique)), unique)), logit(p), rcond=None)[0]
    return np.column_stack((np.ones(len(e)), e)) @ weights, weights


def objective(w, phi, offset, q, penalty):
    z = offset+phi@w
    loss = np.mean(np.logaddexp(0, z)-q*z)+penalty*np.dot(w, w)/2
    gradient = phi.T@(expit(z)-q)/len(q)+penalty*w
    return loss, gradient


def test_contracts():
    rng = np.random.default_rng(20261005)
    x = rng.normal(size=(12, 4));p = expit(rng.normal(size=12));w = rng.normal(size=4)
    off = rng.normal(size=12);v,g = objective(w,x,off,p,.01)
    fd = np.array([(objective(w+1e-6*np.eye(4)[i],x,off,p,.01)[0]-objective(w-1e-6*np.eye(4)[i],x,off,p,.01)[0])/2e-6 for i in range(4)])
    assert np.allclose(g,fd,atol=2e-8)
    k = np.exp(-cdist(x,x,'sqeuclidean')/2)
    eig,vec = eigh(k);keep = eig>1e-9
    phi = vec[:,keep]*np.sqrt(eig[keep])
    projected = k@(vec[:,keep]/np.sqrt(eig[keep]))
    assert np.allclose(phi,projected,atol=1e-7)
    print(dict(contracts='PASS',finite_difference_error=float(abs(g-fd).max())),flush=True)


def train(fold):
    held = base.folds()[fold]
    fit = [i for i in range(12) if i not in held]
    trainc = fit+([12,13,14,15] if 0 in fit and 1 in fit else [])
    d,x,tr,va,norm = base.load(fit)
    inputs,hnorm = features(d,x,tr)
    ii,ci = np.tile(tr,len(trainc)),np.repeat(trainc,len(tr))
    a = inputs['correct'][ci,ii]
    q = d['success'][ci,ii]/(d['success'][ci,ii]+d['failure'][ci,ii])
    offset,weights = prior(d,ci,ii)
    squared = cdist(a,a,'sqeuclidean')
    median = float(np.median(squared[np.triu_indices(len(a),1)]))
    assert median>0
    dest = OUT/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists(), 'Preserve completed fits'
    out,rows,convergence = {},[],[]
    def record(name,condition,z,**meta):
        out[f'{name}__{condition}'] = z
        rows.append(dict(model=name,condition=condition,fold=fold,**meta,**base.metrics(z,d,va,held)))
    priorz = np.broadcast_to(offset[va],(16,len(va)))
    for condition in CONDITIONS:record('eta_prior',condition,priorz)
    # Local baselines expose missing local support without a neural fit.
    query_dist = {key:cdist(v[:,va].reshape(-1,a.shape[1]),a,'sqeuclidean') for key,v in inputs.items()}
    for kk in (5,20):
        for condition,ds in query_dist.items():
            ids = np.argsort(ds,axis=1,kind='stable')[:,:kk]
            pp = (q[ids].sum(1)+.5)/(kk+1)
            record(f'knn{kk}',condition,logit(pp).reshape(16,len(va)))
    for width in (.5,1.,2.):
        variance = median*width**2
        kernel = np.exp(-squared/(2*variance))
        eigen,vec = eigh(kernel,check_finite=False,driver='evr')
        keep = eigen>max(float(eigen[-1])*1e-9,1e-10)
        phi = vec[:,keep]*np.sqrt(eigen[keep])
        projection = vec[:,keep]/np.sqrt(eigen[keep])
        assert np.max(abs(kernel@projection-phi))<1e-5
        for penalty in (.001,.01):
            name=f'kernel_w{width:g}_r{penalty:g}'
            result=minimize(objective,np.zeros(phi.shape[1]),args=(phi,offset[ii],q,penalty),jac=True,
                method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12,gtol=1e-6))
            assert result.success and np.isfinite(result.fun),(name,result.message)
            alpha = projection@result.x
            convergence.append(dict(model=name,converged=bool(result.success),iterations=int(result.nit),
                rank=int(keep.sum()),objective=float(result.fun),gradient_max=float(abs(result.jac).max())))
            for condition,ds in query_dist.items():
                zz = np.exp(-ds/(2*variance))@alpha+np.tile(offset[va],16)
                record(name,condition,zz.reshape(16,len(va)))
            np.savez_compressed(dest/f'{name}.npz',training_features=a,alpha=alpha,variance=variance,eta_prior_weights=weights)
    np.savez_compressed(dest/'predictions.npz',**out)
    base.csvwrite(dest/'metrics.csv',rows)
    write(dest/'normalization.json',dict(context=norm,physical=hnorm))
    write(dest/'complete.json',dict(fold=fold,fit=trainc,held=held,TRAIN_families=64,VAL_families=16,
        dataset_sha256=sha(base.OUT/'dataset.npz'),code_sha256=sha(__file__),protocol_sha256=sha(RULE),
        new_rollouts=0,target_labels_read=False,convergence=convergence,median_squared_TRAIN_distance=median))
    print(dict(fold=fold,completed=True,models=len(convergence),new_rollouts=0),flush=True)


def summarize():
    import csv
    rows=[]
    for fold in range(3):
        guard=read(OUT/f'fold{fold}/complete.json')
        assert guard['code_sha256']==sha(__file__) and guard['protocol_sha256']==sha(RULE)
        with (OUT/f'fold{fold}/metrics.csv').open() as f:rows.extend(list(csv.DictReader(f)))
    summary=[]
    for name in sorted({r['model'] for r in rows}):
        for condition in CONDITIONS:
            rr=[r for r in rows if r['model']==name and r['condition']==condition]
            summary.append(dict(model=name,condition=condition,NLL=float(np.mean([float(r['NLL']) for r in rr])),
                B15=sum(int(r['B15']) for r in rr),cases=sum(int(r['cases']) for r in rr),
                selected_Q=float(np.mean([float(r['selected_Q']) for r in rr])),
                centered_contrast_skill=float(np.mean([float(r['centered_contrast_skill']) for r in rr]))))
    best=min((r for r in summary if r['model'].startswith('kernel') and r['condition']=='correct'),key=lambda r:r['NLL'])
    baseline=next(r for r in summary if r['model']=='eta_prior' and r['condition']=='correct')
    base.csvwrite(OUT/'source_summary.csv',summary)
    write(OUT/'selection_frozen.json',dict(selected=best,eta_prior=baseline,criterion='source_held_controller_VAL_NLL',
        protocol_sha256=sha(RULE),code_sha256=sha(__file__),new_rollouts=0,target_labels_used=False,
        gate_vs_neural_trunk=best['NLL']<.5397345 and best['B15']>=123,
        limitation='Repeated source development CV is not independent confirmation; no target promoted.'))
    print(dict(selected=best,eta_prior=baseline),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('test','train','summarize'))
    parser.add_argument('--fold',type=int,default=0);args=parser.parse_args()
    if args.action=='test':test_contracts()
    elif args.action=='train':train(args.fold)
    else:summarize()
