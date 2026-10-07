"""Summarize both preregistered confirmations without choosing a target/seed.

Bootstrap intervals resample physical families within each of the two fixed
controllers. They do not claim a population interval over new controllers.
"""
import csv
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from .function_support import OUT as SOURCE, read, write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite, sha


def main():
    targets = [SOURCE.parent / name for name in
               ('controller_function_confirmation', 'controller_function_confirmation_b')]
    datasets = [np.load(t / 'dataset.npz') for t in targets]
    predictions = [np.load(t / 'predictions.npz') for t in targets]
    audits = [read(t / 'evaluation_audit.json') for t in targets]
    assert all(a['models_frozen_sha256'] == sha(SOURCE / 'models_frozen.json') for a in audits)
    rng = np.random.default_rng(202610048129)
    boot = np.concatenate([rng.integers(64, size=(10000, 64)) + 64 * i for i in range(2)], axis=1)
    s = np.concatenate([d['success'].reshape(64, 2) for d in datasets])
    f = np.concatenate([d['failure'].reshape(64, 2) for d in datasets])
    q = s / (s + f)
    good, bad = s >= 15, f >= 2
    idx = np.arange(128)
    records, contrasts = [], []
    for arm in ('four_controller', 'reweighted_four_controller', 'twelve_controller'):
        for kind in ('eta_only', 'entity_response_mean'):
            for seed in (17, 23, 41):
                key = f'{arm}__{kind}__{seed}'
                z = np.concatenate([p[key + '__correct'] for p in predictions])
                chosen = z.argmax(1)
                nll = (s * np.logaddexp(0, -z) + f * np.logaddexp(0, z)).sum(1) / (s + f).sum(1)
                records.append(dict(arm=arm, kind=kind, seed=seed, cases=128,
                    B15=int(good[idx, chosen].sum()), unknown=int((~good[idx, chosen] & ~bad[idx, chosen]).sum()),
                    oracle_B15=int(good.any(1).sum()), selected_Q_observed=float(q[idx, chosen].mean()),
                    NLL=float(nll.mean()), MAE=float(abs(expit(z) - q).mean()),
                    per_target_B15=[int(good[j:j+64][np.arange(64), chosen[j:j+64]].sum()) for j in (0, 64)]))
                refs = {'eta_only': f'{arm}__eta_only__{seed}__correct',
                        'same_controller_wrong_state_context': key + '__context_state_shuffle',
                        'joint_state_context_shuffle': key + '__joint_state_context_shuffle',
                        **{f'wrong_{c}': key + f'__wrong_{c}' for c in ('alt', 'second', 'extra_a', 'extra_b')}}
                if arm == 'twelve_controller':
                    refs.update(matched_four_controller=f'four_controller__{kind}__{seed}__correct',
                                recipe_reweighted_four_controller=f'reweighted_four_controller__{kind}__{seed}__correct')
                for ref, rkey in refs.items():
                    zz = np.concatenate([p[rkey] for p in predictions]); other = zz.argmax(1)
                    ll = (s * np.logaddexp(0, -zz) + f * np.logaddexp(0, zz)).sum(1) / (s + f).sum(1)
                    dl = nll - ll; dq = q[idx, chosen] - q[idx, other]
                    ga, gb = good[idx, chosen], good[idx, other]
                    ba, bb = bad[idx, chosen], bad[idx, other]
                    r, b = int((ga & bb).sum()), int((ba & gb).sum())
                    contrasts.append(dict(arm=arm, kind=kind, seed=seed, reference=ref,
                        NLL_delta=float(dl.mean()), NLL_CI=np.quantile(dl[boot].mean(1), [.025, .975]).tolist(),
                        selected_Q_delta=float(dq.mean()), selected_Q_CI=np.quantile(dq[boot].mean(1), [.025, .975]).tolist(),
                        rescue=r, breaks=b, net_rescue=r-b,
                        paired_exact_p=float(binomtest(r, r+b, .5).pvalue) if r+b else 1.,
                        unknown_comparisons=int(((~ga & ~ba) | (~gb & ~bb)).sum()),
                        top1_changes=int((chosen != other).sum())))
    csvwrite(SOURCE / 'independent_confirmation_pooled_metrics.csv', records)
    csvwrite(SOURCE / 'independent_confirmation_pooled_comparisons.csv', contrasts)
    write(SOURCE / 'independent_confirmation_summary.json', dict(
        target_audits=audits, pooled_metrics=records,
        comparisons=[r for r in contrasts if r['arm']=='twelve_controller' and r['kind']=='entity_response_mean'],
        uncertainty_scope='Physical-family bootstrap conditional on these two fixed controllers; only two independent controller functions. Seeds are not independent data replicates.',
        observed_Q_note='s/(s+f) excludes numerical failures. B15 requires>=15 observed successes; non-B15 requires>=2 observed failures. Numerical omissions never become fabricated full Q16.',
        scope='Known Ring geometry, two seen exact eta; unseen physical families and two unseen controller parameters. Not strict cross-scene or unseen-eta validation.',
        target_labels_used_to_select_models=False, generator_modified=False,
        models_frozen_sha256=sha(SOURCE / 'models_frozen.json')))
    for r in records:
        print(r)


if __name__ == '__main__':
    main()
