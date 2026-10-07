# Deterministic G_phi targeted RECOVERY-zero retraining v3

## Diagnosis

**RECOVERY_ZERO_COVERAGE_FIXES_FAILURE**  
Closed-loop pilot status: **NOT_READY_FOR_CLOSED_LOOP_PILOT**.

The frozen `214 -> 128 -> 128 -> 4` SiLU model was retrained with direct MSE to `g*_exec`; no gate, auxiliary loss, reweighting, architecture change, controller change, or closed-loop benchmark was introduced.

## Selected model

- Seed: **41** (chosen by validation state-grouped MSE from 17/23/41).
- Parameters: **44548**.
- AdamW: lr `1e-3`, weight decay `1e-5`; train-only normalization.
- Checkpoint: `/home/zhihan/research/Basin_C1/diagnostics/gphi_pilot_training_v3/best_checkpoint.npz`.

| Split | state-grouped mean L2 (m/s) | state-grouped RMSE |
|---|---:|---:|
| Train | 0.003862 | 0.002647 |
| Validation | 0.034803 | 0.030543 |
| Test | 0.015429 | 0.014337 |

## Primary RECOVERY-zero result

- Held-out zero-label RECOVERY states/samples: 17/1088.
- False intervention mean / median / P95 / max: **0.025806 / 0.005647 / 0.105599 / 0.114649 m/s**.
- V2 published mean: **0.111549 m/s**; V3 change: **-0.085743 m/s (76.87% reduction)**.
- Exact matched nine V2 zero-label RECOVERY states: 0.111548 -> 0.044803 m/s (59.84% reduction).
- Test nonzero-label error: **0.009572 m/s** versus V2 **0.007794 m/s** (`+0.001778`, `+22.81%`); this is modest in absolute scale but is reported as a real relative tradeoff.

## Coverage scaling

With all old training data fixed, validation RECOVERY-zero false intervention changed from **0.057923** at +0 new states to **0.043825 m/s** at +36 new states. The sequence was `0.057923 -> 0.041993 -> 0.032478 -> 0.043825` m/s for `+0/+15/+30/+36`: an overall 24.34% endpoint improvement, but not a monotonic curve; the best fixed-seed point was +30.

## Projection replay and scope

Test projected executed-action error is 0.014828 m/s; mean projection rewrite is 0.001744 m/s. Solver failures: 0; invalid projected actions: 0; nonfinite predictions: 0.

This was an offline state-coverage experiment only. No formal closed-loop benchmark was run.

Residual failure concentration: four retained, close-geometry active-recovery states (inter-agent distance below 0.55 m) average 0.093185 m/s false intervention. This concentrated tail is why the checkpoint is not ready for a closed-loop pilot despite the large aggregate and matched-state gains.

Residual failure concentration: four retained, close-geometry active-recovery states (inter-agent distance below 0.55 m) average 0.093185 m/s false intervention. This concentrated tail is why the checkpoint is not ready for a closed-loop pilot despite the large aggregate and matched-state gains.
