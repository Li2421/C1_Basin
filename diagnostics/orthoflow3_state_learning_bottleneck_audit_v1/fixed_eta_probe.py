"""Reuse the pre-existing fixed-eta kernel diagnostic on the expanded TRAIN.

Diagnostic only: known controllers and fixed eta heads are not a transferable
replacement model. Hyperparameters are chosen inside TRAIN, never on source VAL.
"""
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from diagnostics.orthoflow3_state_learning_bottleneck_audit_v1 import audit as a
from diagnostics.orthoflow3_state_context_adjudication_v1.audit import (
    canonical_h, fit_predict, choose_alpha, evaluate)


def main():
    d,rows,states,ix,tr,va,ys,s,n=a.load()
    x=dict(np.load(a.SRC/'variants/large_20/entities.npz'))
    h=canonical_h(x)
    hh=np.broadcast_to(h[None,:,None,:],(3,len(h),16,h.shape[-1]))
    cc=d['context'][:,ix].astype(float)
    ss=d['success'][:,ix].astype(float)
    nn=ss+d['failure'][:,ix]
    prior=(ss[:,tr].sum(1)+.5)/(nn[:,tr].sum(1)+1.)
    p0=np.broadcast_to(prior[:,None,:],(3,len(va),16))
    result={'scope':'Known-controller fixed-eta source-family predictability diagnostic; not LOSO',
        'training_counts':'Exactly the frozen large_20 successes/failures, including stronger historical evidence',
        'alpha_grid':[.03,.3,3.,30.],'alpha_selection':'6-fold TRAIN source-family CV, original pre-existing grid',
        'prior':evaluate(p0,s[:,va],n[:,va],va),
        'new_rollout':0,'new_target_labels_opened':False,'models':{}}
    saved={'prior':p0}
    for name,blocks in [('unified_h',[hh]),('H20_context',[cc]),('unified_h_plus_H20',[hh,cc])]:
        alpha,cv=choose_alpha(blocks,ss,nn,tr,result['alpha_grid'])
        p=fit_predict(blocks,ss,nn,tr,va,alpha)
        m=evaluate(p,s[:,va],n[:,va],va)
        m['selection_bounds']=a.selection(p,s[:,va],n[:,va])
        # Family-level uncertainty: controllers/eta from a state stay together.
        delta=np.asarray(result['prior']['per_family_NLL'])-m['per_family_NLL']
        rng=np.random.default_rng(2026100401)
        boot=np.mean(delta[rng.integers(0,len(va),(10000,len(va)))],axis=1)
        train_p=fit_predict(blocks,ss,nn,tr,tr,alpha)
        train_loss=a.nll(train_p,ss[:,tr],nn[:,tr])
        result['models'][name]={'alpha':alpha,'TRAIN_CV_NLL':cv,'TRAIN_fit_NLL':train_loss,
            'source_VAL':m,'prior_minus_model_NLL_family_bootstrap_95CI':np.quantile(boot,[.025,.975]).tolist(),
            'prior_minus_model_NLL_by_family':delta.tolist(),
            'caveat':'Only seven held-out source families; this is post-hoc diagnostic evidence, not a new confirmatory test.'}
        saved[name]=p
        a.write('expanded_fixed_eta_probe.json',result)
        print(name,alpha,round(train_loss,4),m['NLL'],m['selection_bounds'],m['state_reversals_correct'],flush=True)
    np.savez_compressed(a.OUT/'expanded_fixed_eta_predictions.npz',**saved)


if __name__=='__main__':main()
