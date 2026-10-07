"""Separate response-based controller identification from state information.

An intentionally privileged known-controller eta prior is diagnostic only.
It is fitted to source TRAIN, never to target labels, and is not deployable
on an unidentified unseen controller. No new models or rollouts.
"""
import numpy as np
from scipy.special import expit
from .function_support import OUT, write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite


def main():
    d = np.load(OUT / 'dataset.npz')
    tr, va = d['split']=='train', d['split']=='validation'
    q = d['success'] / (d['success']+d['failure'])
    p = np.clip(q[:, tr].reshape(12, 64, 2).mean(1), 1e-5, 1-1e-5)
    truth = q[:, va].reshape(12, 16, 2)
    good = d['success'][:, va].reshape(12, 16, 2)>=15
    prior_loss = -truth*np.log(p[:, None, :])-(1-truth)*np.log1p(-p[:, None, :])
    chosen = p.argmax(1)
    baseline = dict(NLL=float(prior_loss.mean()),
                    B15=int(sum(good[c, :, chosen[c]].sum() for c in range(12))),
                    oracle_B15=int(good.any(-1).sum()), cases=192,
                    source_TRAIN_controller_eta_priors=p.tolist())
    rng = np.random.default_rng(202610048130)
    boot = rng.integers(16, size=(10000, 16))
    rows = []
    for seed in (17,23,41):
        z = np.load(OUT/'models/twelve_controller/entity_response_mean'/f'seed{seed}'/'validation_predictions.npz')['correct'].reshape(12,16,2)
        losses = truth*np.logaddexp(0,-z)+(1-truth)*np.logaddexp(0,z)
        family_delta = (losses-prior_loss).mean((0,2))
        cc = z.argmax(-1)
        rows.append(dict(seed=seed,full_NLL=float(losses.mean()),
            full_B15=int(good[np.arange(12)[:,None],np.arange(16)[None,:],cc].sum()),
            NLL_minus_known_controller_prior=float(family_delta.mean()),
            conditional_family_CI=np.quantile(family_delta[boot].mean(1),[.025,.975]).tolist()))
    write(OUT/'known_controller_prior_audit.json',dict(baseline=baseline,full=rows,
        role='Post-hoc source-only diagnostic; not an unseen-controller baseline and not a model-selection criterion.',
        caution='A response-shuffle penalty alone need not equal incremental state information: mismatching h and C can also disrupt learned controller identification or create impossible inputs.',
        new_rollouts=0,target_labels_used=False))
    print(baseline);print(rows)


if __name__=='__main__':main()
