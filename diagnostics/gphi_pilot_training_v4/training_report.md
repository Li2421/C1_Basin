# Deterministic G_phi recovery-boundary retraining V4

## Decision

**BOUNDARY_NOT_LEARNABLE_WITH_CURRENT_REGRESSION**  
**NOT_READY_FOR_CLOSED_LOOP_PILOT**

The deterministic `214 -> 128 -> 128 -> 4` SiLU MLP, direct MSE target, FlowBC, both hard projections, and all oracle/controller semantics remained unchanged. No gate, deadband, classifier, weighting, auxiliary loss, or closed-loop benchmark was introduced.

## Selected model

- Seed **23**, best epoch 35, 44,548 parameters.
- AdamW, lr `1e-3`, weight decay `1e-5`, train-only normalization.
- Checkpoint: `/home/zhihan/research/Basin_C1/diagnostics/gphi_pilot_training_v4/best_checkpoint.npz`.

| Split | state-grouped mean L2 (m/s) | state-grouped RMSE |
|---|---:|---:|
| Train | 0.017591 | 0.011160 |
| Validation | 0.054905 | 0.040856 |
| Test | 0.049594 | 0.040321 |

## Boundary results

- New held-out close-range zero states: 4; false intervention mean/median/P95/max = **0.125815/0.116143/0.191909/0.193380 m/s**.
- V3 on those exact states: 0.096932; matched reduction: -29.80%.
- Old four hard states: **0.093185 -> 0.079257 m/s** (14.95% reduction).
- New held-out close nonzero states: error 0.033337 m/s, relative error 27.58%, target norm 0.124909, predicted norm 0.100023, cosine 0.9803.
- Overall nonzero test error: **0.009572 -> 0.023697 m/s** (+147.58%).

## Learnability and projection

Deployment-feature 1-NN leave-one-out boundary accuracy is 50.00%; the current 214-D input appears sufficient: **False**. Detailed overlap, matched-pair, and feature-group diagnostics are stored without changing the feature schema.

Test projected executed-action error is 0.040383 m/s and mean rewrite is 0.016359 m/s. Solver failures: 0; invalid actions: 0.

No formal closed-loop benchmark was run.
