"""Diagnostic orthogonal Brier/interaction decomposition, not a deployable oracle.

Held-out controller/eta outcome means are used only to describe variance;
they are never prediction inputs or model-selection criteria.
"""
import numpy as np
from scipy.special import expit
from .goal_response import OUT as SOURCE
from .function_support import read, write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite


def main():
    paths=[SOURCE.parent/n for n in ('goal_response_confirmation','goal_response_confirmation_b')]
    ds=[np.load(p/'dataset.npz') for p in paths]
    ps=[np.load(p/'predictions.npz') for p in paths]
    s=np.array([d['success'].reshape(64,2) for d in ds]);f=np.array([d['failure'].reshape(64,2) for d in ds]);n=s+f;q=s/n
    qm=q.mean(1,keepdims=True);qr=q-qm
    # Pointwise binomial variance estimate; seed outcomes may share RNG, so
    # this is descriptive and not used for formal contrast confidence.
    label_noise=q*(1-q)/np.maximum(n-1,1)
    tq=q[:,:,0]-q[:,:,1];tqc=tq-tq.mean(1,keepdims=True)
    rng=np.random.default_rng(202610048132);boot=rng.integers(64,size=(10000,64))
    rows=[]
    for e in read(SOURCE/'models_frozen.json')['models']:
        key=f"{e['variant']}__{e['seed']}__correct";p=expit(np.array([z[key] for z in ps]));pm=p.mean(1,keepdims=True);pr=p-pm
        brier=float(np.mean((q-p)**2));mean_term=float(np.mean((qm-pm)**2));res_term=float(np.mean((qr-pr)**2))
        assert np.isclose(brier,mean_term+res_term,atol=1e-6)
        dp=p[:,:,0]-p[:,:,1];dpc=dp-dp.mean(1,keepdims=True)
        # Positive residual skill means finer-than-controller/eta prediction.
        skill=qr**2-(qr-pr)**2;contrast_skill=tqc**2-(tqc-dpc)**2
        family_skill=skill.mean(2);replicates=(family_skill[0,boot].mean(1)+family_skill[1,boot].mean(1))/2
        rows.append(dict(variant=e['variant'],seed=e['seed'],Brier=brier,
            controller_eta_mean_calibration_term=mean_term,state_residual_error_term=res_term,
            zero_state_residual_error=float(np.mean(qr**2)),state_residual_skill=float(skill.mean()),
            state_residual_skill_CI=np.quantile(replicates,[.025,.975]).tolist(),
            centered_state_eta_contrast_skill=float(contrast_skill.mean()),
            centered_contrast_R2=float(1-np.mean((tqc-dpc)**2)/np.mean(tqc**2)),
            predicted_contrast_std=float(dpc.std()),true_contrast_std=float(tqc.std()),
            controller0_state_residual_skill=float(skill[0].mean()),controller1_state_residual_skill=float(skill[1].mean())))
    csvwrite(SOURCE/'independent_information_decomposition.csv',rows)
    write(SOURCE/'independent_information_decomposition.json',dict(results=rows,
        true_within_controller_eta_variance=float(np.mean(qr**2)),
        estimated_pointwise_binomial_noise=float(label_noise.mean()),
        note='Target means are retrospective variance references only; no training, calibration or selection uses them. Family-bootstrap CI conditional on2controllers. Paired-seed contrast noise not inferred from independent-binomial approximation.',
        new_rollouts=0,target_labels_used_for_model_selection=False))
    for r in rows:print(r)


if __name__=='__main__':main()
