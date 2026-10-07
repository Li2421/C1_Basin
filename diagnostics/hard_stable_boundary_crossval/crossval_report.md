# Hard stable boundary source-group cross-validation

## Decision

**HARD_BOUNDARY_GENERALIZATION_FAILS**

No states or oracle rollouts were added. The pre-defined difficult stable pool contains 13 states from 6 source groups (6 zero / 7 nonzero). Six strict leave-one-source-group-out folds were used. Each fold excluded its outer group from training, normalization, inner validation, early stopping, and threshold selection.

## Pooled difficult out-of-fold result

- MLP 64x64 seed-mean: balanced accuracy 0.3929, AUROC 0.3333, AUPRC 0.6161.
- FPR 0.5000, FNR 0.7143, accuracy 0.3846 (5/13).
- Source-group bootstrap 95% CI: balanced accuracy [0.1000, 0.6333], AUROC [0.0000, 0.8000], accuracy [0.0909, 0.6429].
- Robust MLP mistakes across all three seeds: 6.
- Difficult ensemble errors by source group: {'anchor_D2_pair228': 2, 'anchor_D4_pair227': 0, 'baseline_r175': 2, 'baseline_r198': 2, 'qual_pair225': 1, 'qual_pair226': 1}.

## Linear versus nonlinear and global gap

- Linear difficult balanced accuracy / AUROC: 0.4762 / 0.3571.
- MLP difficult balanced accuracy / AUROC: 0.3929 / 0.3333.
- MLP all-stable held-out-group balanced accuracy / AUROC: 0.8751 / 0.8743 over 99 states.

## Interpretation

The pooled result, group bootstrap uncertainty, per-fold behavior, seed sensitivity, and recurrent errors determine the classification above. The smallest justified next step is: Audit the recurrently misclassified stable states against their existing full augmented snapshots; do not increase gate capacity.

No correction head was trained and no closed-loop control was run.
