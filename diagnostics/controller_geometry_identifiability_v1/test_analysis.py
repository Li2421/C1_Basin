"""Unit test of analysis.py on synthetic labels (no experiment data are read).

A planted geometry-dependent ground truth: the rotate operators help only for
w > 1.3 and hurt below; eta preference flips at w = 1.1. Checks code paths,
shapes and that obvious planted structure is recovered by the interpolating
regime but not by extrapolation. Tiny learner settings; not a result.
"""
import numpy as np
from . import analysis as A


def synthetic():
    rng = np.random.default_rng(0)
    widths = [0.72, 0.78, 0.84, 0.90, 1.02, 1.08, 1.14, 1.20, 1.26, 1.32, 1.44, 1.50, 1.56, 1.62]
    S = 16
    Y = np.full((len(widths), S, 5, 4, 4), np.nan)
    for i, w in enumerate(widths):
        base = np.clip((w - 0.85) / 0.5, 0, 1)
        for c in range(5):
            for e in range(4):
                rot = (c in (2, 3)) * (0.4 if w > 1.3 else -0.4)
                eta = (0.3 if (e == 3) == (w < 1.1) else 0.0)
                p = np.clip(base * 0.6 + rot * base + eta * base, 0, 1)
                Y[i, :, c, e, :] = rng.random((S, 4)) < p
    roles = {w: 'pool' for w in widths}
    grid = dict(grid=widths, test=[0.90, 1.08, 1.26, 1.44],
                pool=[0.72, 0.78, 0.84, 1.02, 1.14, 1.20, 1.32, 1.50, 1.56, 1.62], few=[0.72, 1.20, 1.62])
    mech = {w: [('success', False, 100)] for w in widths}
    return dict(Y=Y, widths=widths, roles=roles, X_state=rng.normal(size=(S, 24)), eta=rng.normal(size=(4, 3)),
                grid=grid, mech=mech, numerical_failures=0, missing=0, n_cases=int(Y.size))


def main():
    A.STEPS, A.BOOT, A.LEARNER_SEEDS = 300, 30, [17]
    D = synthetic(); rng = np.random.default_rng(1)
    a = A.test_a(D, rng)
    print('A:', a['informative_widths'][:3], 'A1', a['A1_supported'], 'A2', a['A2_sign_reversal'],
          'neg', a['significant_negative'][:3], 'pos', a['significant_positive'][:3])
    b = A.test_b(D, rng)
    for k, v in b['results'].items():
        o = v['own_targets']
        print(f"{k:9s} B2 {o['B2']:+.2f} B1 {o['B1_skill']:+.2f} sel {o['sel_model']:.2f} oracle {o['sel_oracle']:.2f} prior {o['sel_prior']:.2f} n={v['train_rollouts']}")
    print({k: v for k, v in b['verdict'].items() if k != 'extrapolation_detail'})


if __name__ == '__main__':
    main()
