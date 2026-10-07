# Deterministic G_phi pilot training v2

## Coverage diagnosis

**STILL_DATA_LIMITED**

Closed-loop pilot readiness: **TRAINING_NEEDS_REVISION**.

The selected model predicts the four-dimensional instantaneous executed correction
`g*_exec = u*_exec - u_safe` directly.  It does not predict eta, a stochastic
distribution, or a trajectory.  No frozen controller, physics, event, oracle, or
dataset artifact was modified.

## Selected model

- Architecture: `214 -> 128 -> 128 -> 4`, SiLU hidden activation.
- Parameters: 44548.
- Optimizer: AdamW, learning rate 0.001, weight decay 1e-05.
- Selected training seed: 41 from seeds 17/23/41, using validation state-grouped MSE only.
- Checkpoint: `best_checkpoint.npz`.

## Held-out state results

| Split | state-grouped RMSE | state-grouped mean L2 | sample mean L2 |
|---|---:|---:|---:|
| Train | 0.00238312 | 0.00423081 | 0.00423081 |
| Validation | 0.0338005 | 0.0412853 | 0.0412853 |
| Test | 0.0276381 | 0.0305454 | 0.0305454 |

Test nonzero-label state-grouped mean L2 is 0.00779355 m/s.
Test zero-label false intervention mean norm is 0.0580872 m/s
(11.617% of vmax).

## Published-split V1 comparison

The V1 and V2 rows below use their respective published test splits.  V2 is a
materially broader and harder 42-state test set, so this table alone does not
isolate the effect of training coverage.

| Metric | V1 | V2 | V2 - V1 |
|---|---:|---:|---:|
| Train state-grouped mean L2 | 0.00301638 | 0.00423081 | +0.00121443 |
| Validation state-grouped mean L2 | 0.0458845 | 0.0412853 | -0.0045992 |
| Test state-grouped mean L2 | 0.0208517 | 0.0305454 | +0.00969373 |
| Zero-label false intervention | 0.0349482 | 0.0580872 | +0.023139 |
| Nonzero-label error | 0.0177192 | 0.00779355 | -0.00992565 |

The fixed-seed scaling diagnostic changed validation state-grouped mean L2 from
0.0704105 at 46
training states to 0.0419245 at
167 training states.

## Matched-state coverage comparison

On the identical expanded V2 test states, the frozen V1 checkpoint versus the
V2 checkpoint gives:

| Metric | V1 checkpoint | V2 checkpoint | reduction |
|---|---:|---:|---:|
| All-state mean L2 | 0.149017 | 0.0305529 | 79.50% |
| Zero-label false intervention | 0.303315 | 0.0580852 | 80.85% |
| Nonzero-label error | 0.0215543 | 0.00780886 | 63.77% |

On the 11 retained original V1 test states, all-state error improves from
0.0208517 to 0.00699629 and zero-label false intervention improves from
0.0349482 to 0.00651908 (81.35%).  Thus added coverage clearly helps on
matched states.  However, absolute V2 false intervention remains poor because
the expanded test exposes nine zero-label RECOVERY states: their mean false
intervention is 0.111549 m/s, versus 0.00997134 m/s on ten zero-label NORMAL
states.  Training contains only five zero-label RECOVERY states because exact
source-trajectory grouping leaves the three recovery geometry groups strongly
different in zero/nonzero composition.  Together with the still-decreasing
scaling curve, this supports **STILL_DATA_LIMITED**, not readiness.

## Required baselines on test

| Model | state-grouped mean L2 | state-grouped RMSE |
|---|---:|---:|
| ZERO | 0.0241345 | 0.0177741 |
| TRAIN-MEAN | 0.0350324 | 0.0193104 |
| Linear | 0.0401138 | 0.0303215 |
| Selected MLP | 0.0305454 | 0.0276381 |

## Projection replay

On test samples, mean post-projection executed-action error is
0.0300247 m/s.  The unchanged
second projection changes the raw predicted action by a mean of
0.00207249 m/s; the fraction
changed above 1e-6 is 48.586%.
Projection failures: 0.

## Generalization cautions

The 15,744 samples come from 246 independent augmented states.  Splits remain
source-state/source-trajectory grouped.  Train/validation and train/test
state-MSE ratios are 201 and 135.  Nearest-state distances
and output-collapse diagnostics are recorded in `test_metrics.json`; category,
zero/nonzero, and label-ambiguity results are in their dedicated CSV files.

This is an offline supervised pilot.  No large closed-loop benchmark was run.
