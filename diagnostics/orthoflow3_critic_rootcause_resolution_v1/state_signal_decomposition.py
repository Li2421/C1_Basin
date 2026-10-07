"""Descriptive held-family decomposition; no model fitting or selection.

Separate state-dependent overall difficulty from eta-preference interaction.
Centering by controller here is an evaluation statistic, never an inference
preprocessing step. All predictions were frozen before confirmation labels.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
from .support_confirmation import OUT, read, write
from .state_support import OUT as SOURCE
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite


def main():
    d = np.load(OUT / 'dataset.npz')
    z = np.load(OUT / 'predictions.npz')
    q = (d['success'] / (d['success'] + d['failure'])).reshape(2, 64, 2)
    rng = np.random.default_rng(20261004092)
    bootstrap = rng.integers(64, size=(5000, 64))
    rows = []
    for entry in read(SOURCE / 'models_frozen.json')['models']:
        key = '__'.join(str(entry[k]) for k in ('size', 'kind', 'seed'))
        pred = expit(z[key + '__correct']).reshape(2, 64, 2)
        for name, true, estimate in (
            ('state_difficulty', q.mean(-1), pred.mean(-1)),
            ('eta_preference', q[..., 0] - q[..., 1], pred[..., 0] - pred[..., 1]),
        ):
            def center(a):
                return a - a.mean(1, keepdims=True)
            t, p = center(true), center(estimate)
            if p.std() < 1e-8:
                corr, low, high = None, None, None
            else:
                corr = float(pearsonr(t.ravel(), p.ravel()).statistic)
                bt = true[:, bootstrap]; bp = estimate[:, bootstrap]
                bt -= bt.mean(-1, keepdims=True); bp -= bp.mean(-1, keepdims=True)
                bcor = (bt * bp).sum((0, 2)) / np.sqrt((bt**2).sum((0, 2)) * (bp**2).sum((0, 2)))
                low, high = np.quantile(bcor, [.025, .975]).tolist()
            rows.append(dict(size=entry['size'], kind=entry['kind'], seed=entry['seed'], component=name,
                centered_correlation=corr, family_CI_low=low, family_CI_high=high,
                true_std=float(t.std()), predicted_std=float(p.std()),
                centered_MAE=float(abs(t-p).mean())))
    csvwrite(OUT / 'difficulty_vs_preference.csv', rows)
    write(OUT / 'difficulty_vs_preference_audit.json', dict(
        descriptive_only=True, new_model_selection=False, new_rollout=0,
        source_families=64, known_controllers=2, candidates=2,
        centering='Remove each controller mean for analysis only; not an inference transform',
        bootstrap='5000 resamples of source families jointly retaining both controllers and eta',
        caution='No finite-seed noise correction. Correlation of a predicted contrast is not B15 selection reliability.',
        mathematical_input_sufficiency_scope='Complete native Markov state plus exact identity of either known frozen source controller is an information-complete diagnostic within that fixed family; its held-family failure cannot be blamed on omitting raw physical variables alone.',
        H20_scope='The exact-prefix counterexample disproves sufficiency over the enlarged delayed-switch controller-program class, not exact aliasing among the natural stationary controllers.'))


if __name__ == '__main__':
    main()
