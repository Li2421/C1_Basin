# Source-group Group-DRO gate training

This controlled experiment reused the frozen seven source-group-held-out folds,
214-D inputs, stable oracle labels, BCE per-example loss, SiLU 214→64→64→1
gate, train-only normalization, validation-only early stopping, and
validation-only threshold selection. No rollout, state, feature, oracle,
correction-head, or closed-loop work occurred.

`BASELINE_BCE_PRESERVED` remains commented in
`group_dro_gate_runner.py`; `source_balanced_bce` reproduction matched the
prior source-balanced condition before Group-DRO training began.

## Validation-only Group-DRO selection

The pre-registered candidates were eta_q = 0.01, 0.05, and 0.1. The selected
per-fold eta values were 0.1 for anchor_D2_pair228, anchor_D4_pair227,
baseline_r175, and anchor_D1_pair231; 0.05 for baseline_r198 and
qual_pair226; and 0.01 for qual_pair225. Selection did not access outer-test
data.

## Primary OOF results: balanced accuracy / FPR / FNR

| condition | all stable (N=130) | recovery stable (N=95) | difficult stable (N=13) |
|---|---:|---:|---:|
| Source-group-balanced BCE | 0.9025 / 0.1200 / 0.0750 | 0.8844 / 0.1200 / 0.1111 | 0.3929 / 0.5000 / 0.7143 |
| Group-DRO BCE | 0.8900 / 0.1200 / 0.1000 | 0.8511 / 0.1200 / 0.1778 | 0.3929 / 0.5000 / 0.7143 |

Any reported pooled ranking measure is computed in the fold-relative coordinate
`logit - validation_threshold_logit`; raw logits from different OOF models
are not pooled.

The difficult stable result remains 5/13 correct. The fixed stable-zero probes
p030, p073, and p050 remain intervention for all 64 seed-mean Flow variants.
Their mean fold-relative margins decline only from +1.445, +1.798, +1.891 to
+1.069, +1.591, +1.429, respectively, and none crosses to the correct side.

For difficult stable nonzero states, only two of seven entire Flow clouds are
fully intervention; their mean intervention fraction is 0.3504 and the
cloud-level FNR fraction is 0.7143.

## Dynamics and diagnostics

The final q distribution is concentrated but not a one-group collapse: median
effective group count is 7.60 of roughly 110 training groups and median maximum
q is 0.289. `baseline_r118` is the final top-q group in all 21 selected
fold/seed runs. Its q trajectory does not show severe epoch-scale oscillation
(4–9 top-group switches across 370–490 epochs; maximum adjacent-epoch q L1
change ≤0.069). Worst training-group BCE falls in every run (median change
−1.702), but this training improvement does not transfer to hard held-out
groups.

Fold-to-fold mean-margin spread for stable-zero states worsens from 0.3388 to
0.4718. Stable-nonzero spread changes slightly from 0.4944 to 0.4794 while its
mean error rises from 0.1144 to 0.1249. Thus Group-DRO does not improve
threshold transfer overall.

Integrated-gradient comparisons on the fixed stable-zero probes find no clear
shortcut reduction: mean wrong-direction geometry contribution rises
1.069→1.160, inter-agent-relative rises 0.168→0.308, and control/projection
rises 0.424→0.498. History/monitor and u_safe decrease modestly, but the
complete clouds remain false positives.

## Decision

**GROUP_DRO_DOES_NOT_HELP.** Explicit worst-training-source optimization lowers
training group losses but does not improve difficult source-group-held-out
boundary generalization and slightly degrades broad stable/recovery FNR.

The smallest justified next experiment is a single source-group-invariance
training constraint on the same frozen data, assessed with a fresh strict LOGO
evaluation—not another generic training-weight sweep.
