"""Source-only, cross-controller AND cross-family low-dimensional calibration."""
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .motion_selection_headroom import stats, digest, write, csvwrite, read

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'motion_calibration_source'
RULE = ROOT/'motion_calibration_protocol.json'
DATA = ROOT/'motion_factorial_support/dataset.npz'
SEEDS = (17,23,41)
KINDS = ('identity','common_affine','eta_affine')


def fit(z, q, t, kind):
    if kind == 'identity':
        return np.array([1.,0.,0.])
    z,q,t = np.broadcast_arrays(z,q,t)
    z,q,t = z.ravel(),q.ravel(),t.ravel()
    assert np.isfinite(z).all() and np.isfinite(q).all()
    use_eta = kind == 'eta_affine'
    def objective(theta):
        a,b = theta[:2]
        g = theta[2] if use_eta else 0.
        transformed = a*z+b+g*t
        residual = expit(transformed)-q
        value = np.mean(np.logaddexp(0,transformed)-q*transformed)
        value += .01*((a-1)**2+b*b+g*g)
        gradient = [np.mean(residual*z)+.02*(a-1),np.mean(residual)+.02*b]
        if use_eta:
            gradient.append(np.mean(residual*t)+.02*g)
        return value,np.array(gradient)
    initial = np.array([1.,0.,0.] if use_eta else [1.,0.])
    result = minimize(objective,initial,jac=True,method='L-BFGS-B',
        bounds=[(.25,4.)]+[(None,None)]*(len(initial)-1),
        options={'ftol':1e-13,'gtol':1e-9,'maxiter':500})
    assert result.success or np.linalg.norm(result.jac)<1e-5, result.message
    return np.r_[result.x,0.] if not use_eta else result.x


def apply(theta,z,t):
    return theta[0]*z+theta[1]+theta[2]*t


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'selection_frozen.json').exists(), 'Do not overwrite a completed source gate'
    d = dict(np.load(DATA))
    va = np.flatnonzero(d['split']=='validation')
    assert len(va)==32 and len(np.unique(d['state_index'][va]))==16
    s,f = [d[k][:12,va].reshape(12,16,2) for k in ('success','failure')]
    assert np.all(s+f>0)
    q=s/(s+f)
    eta=d['eta'][va].reshape(16,2,3)
    assert np.array_equal(eta,np.broadcast_to(eta[0],eta.shape))
    center=eta[0].mean(0); vector=eta[0,1]-eta[0,0]
    t=2*((eta[0]-center)@vector)/(vector@vector)
    assert np.allclose(t,[-1,1])
    rows,parameters,summary,hashes = [],[],[],{str(DATA):digest(DATA),str(RULE):digest(RULE)}
    for seed in SEEDS:
        z=np.full((12,16,2),np.nan); folds=[]
        for fold in range(3):
            path=ROOT/f'motion_freeze_control/cv/fold{fold}/trunk_only/seed{seed}'
            done=read(path/'complete.json'); held=done['held']; folds.append(held)
            assert done['data_sha256']==digest(DATA)
            scores=np.load(path/'predictions.npz')['step400'].reshape(16,16,2)
            z[held]=scores[held]
            hashes[str(path/'predictions.npz')]=digest(path/'predictions.npz')
        assert np.isfinite(z).all() and sorted(sum(folds,[]))==list(range(12))
        for kind in KINDS:
            out=np.full_like(z,np.nan)
            for fold,held in enumerate(folds):
                train_controllers=np.array([c for c in range(12) if c not in held])
                for parity in (0,1):
                    train_families=np.arange(16)[np.arange(16)%2!=parity]
                    test_families=np.arange(16)[np.arange(16)%2==parity]
                    assert not set(train_families)&set(test_families)
                    train=np.ix_(train_controllers,train_families,np.arange(2))
                    test=np.ix_(held,test_families,np.arange(2))
                    theta=fit(z[train],q[train],t,kind)
                    out[test]=apply(theta,z[test],t)
                    parameters.append(dict(seed=seed,kind=kind,held_fold=fold,test_parity=parity,
                        a=float(theta[0]),b=float(theta[1]),gamma=float(theta[2])))
                    rows.append(dict(seed=seed,kind=kind,held_fold=fold,test_parity=parity,
                        **stats(out[test],q[test],s[test],f[test])))
            assert np.isfinite(out).all()
            if kind=='common_affine':
                assert np.array_equal(out.argmax(-1),z.argmax(-1)), 'Common monotone calibration must not change choices'
            summary.append(dict(seed=seed,kind=kind,**stats(out,q,s,f)))
            np.savez_compressed(OUT/f'crossfit_{kind}_seed{seed}.npz',logits=out)
    means={kind:{metric:float(np.mean([r[metric] for r in summary if r['kind']==kind]))
                 for metric in ('NLL','B15','selected_Q','state_centered_contrast_skill')}
           for kind in KINDS}
    selected=min(KINDS,key=lambda k:means[k]['NLL'])
    passed=selected!='identity' and means[selected]['B15']>=means['identity']['B15']
    csvwrite(OUT/'crossfit_metrics.csv',rows)
    csvwrite(OUT/'crossfit_parameters.csv',parameters)
    csvwrite(OUT/'seed_summary.csv',summary)
    write(OUT/'selection_frozen.json',dict(selected=selected,gate_passed=passed,source_means=means,
        seed_summaries=summary,input_sha256=hashes,code_sha256=digest(Path(__file__)),
        target_labels_used=False,new_rollouts=0,already_opened_targets_are_not_confirmation=True,
        eta_geometry=dict(center=center.tolist(),axis=vector.tolist(),scale=float(2/(vector@vector))),
        family_independence='Calibration fit/evaluation disjoint sourceVAL families and controllers; base model sourceCV checkpoint previously selected using sourceVAL. Not an independent final test.'))
    print(dict(selected=selected,gate_passed=passed,source_means=means),flush=True)


if __name__=='__main__':
    main()
