# Group-DRO offline analysis

Read-only analysis of selected checkpoints. No model update, rollout, state, feature, label, or threshold retuning occurred. All ranking numbers below use fold-relative logit margin, never raw scores pooled across independently trained LOGO folds.

## OOF classification (BAcc / FPR / FNR)

| condition | all stable | recovery stable | difficult stable (N=13) |
|---|---:|---:|---:|
| source-balanced BCE | 0.9025 / 0.1200 / 0.0750 | 0.8844 / 0.1200 / 0.1111 | 0.3929 / 0.5000 / 0.7143 |
| Group-DRO | 0.8900 / 0.1200 / 0.1000 | 0.8511 / 0.1200 / 0.1778 | 0.3929 / 0.5000 / 0.7143 |

Group-DRO does not change the primary hard-boundary outcome: 5/13 correct, BAcc 0.3929, FPR 0.5000, FNR 0.7143. All three pre-registered robust zero clouds remain intervention on every seed-mean Flow variant.

## Group-DRO dynamics

Across 21 fold/seed runs, final effective group count has median 7.60 (min 6.17); median final max q is 0.2890. The final top-q group is baseline_r118 in every selected run. Its q mass is concentrated but not single-group collapse; epoch-to-epoch q movement is summarized in `q_trajectory_summary.csv`. First-to-final worst-group BCE falls in every run (see `group_dynamics_summary.csv`).

## Interpretation

**GROUP_DRO_DOES_NOT_HELP.** Worst-training-group weighting changes broad OOF behavior but does not repair the fixed source-held-out hard boundary or the stable-zero false-positive clouds. The smallest justified next experiment is not another generic reweighting sweep: audit a source-group-invariant training constraint/representation objective on the same frozen data under a fresh LOGO evaluation.
