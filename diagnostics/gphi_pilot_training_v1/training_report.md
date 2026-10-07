# Deterministic G_phi pilot training v1

## Decision

**TRAINING_NEEDS_REVISION**

The selected deterministic model predicts the four-dimensional instantaneous
executed correction `g*_exec = u*_exec - u_safe`. It does not predict eta or a
trajectory. No frozen controller, physics, event, oracle, or dataset artifact
was modified. No large closed-loop benchmark was run.

## Selected model

- Architecture: `214 -> 128 -> 128 -> 4`, SiLU.
- Parameters: 44548.
- AdamW: learning rate 0.001, weight decay 1e-05.
- Validation-selected seed: 23 of 17/23/41; best epoch 1125.
- Checkpoint: `best_checkpoint.npz`.

## Held-out state results

| Split | state-grouped RMSE | state-grouped mean L2 | sample mean L2 |
|---|---:|---:|---:|
| Train | 0.00165015 | 0.00301638 | 0.00301638 |
| Validation | 0.0258548 | 0.0458845 | 0.0458845 |
| Test | 0.0128509 | 0.0208517 | 0.0208517 |

## Test baselines

| Model | state-grouped mean L2 | state-grouped RMSE |
|---|---:|---:|
| ZERO | 0.0338681 | 0.0199011 |
| TRAIN-MEAN | 0.033564 | 0.0183624 |
| Linear | 0.0228482 | 0.0141584 |
| Selected MLP | 0.0208517 | 0.0128509 |

## Category and label behavior on test

| Group | state-grouped mean L2 |
|---|---:|
| NORMAL | 0.0349482 |
| PRE_DEADLOCK | 0.00570195 |
| RECOVERY | 0.0237278 |
| zero label | 0.0349482 |
| nonzero label | 0.0177192 |
| LABEL_STABLE | 0.0270094 |
| LABEL_MILDLY_AMBIGUOUS | 0.0100758 |

Zero-label false intervention mean norm is 0.0349482 m/s
(6.990% of vmax and
84.427% of the test nonzero-target mean norm). This does
not support approximately-zero intervention on held-out zero-label states.

## Hard-projection replay

Test post-projection executed-action error is
0.0202031 m/s (mean). The second
projection rewrites the predicted action by 0.00256992
m/s on average; 69.460%
of samples change by more than 1e-6. Projection failures: 0.

## Generalization audit

The 4,288 Flow-seed samples represent only 67 independent augmented states.
The split is source-state/source-trajectory grouped. Validation/train and
test/train state-MSE ratios are 245 and 60.6.
Nearest-held-out-state distances and output-spread checks are in
`test_metrics.json`. This pilot warrants only a small closed-loop smoke check,
not a performance claim.
