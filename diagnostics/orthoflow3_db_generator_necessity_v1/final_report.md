# OrthoFlow3 DB continuous-generator necessity

Classification: **FIXED_MODE_CODEBOOK_SUFFICIENT**

The outcome-blind hard cohort contains 48 source-isolated Double-Bottleneck true-t0 states. The prior cancelled batch of 1,128 rollout records was physically isolated and never used.

| Method | B63 states | coverage | mean Q64 | success/trials | deadlock | timeout | collision | J_def | episode length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 0/48 | 0.000 | 0.677 | 2079/3072 | 0 | 993 | 0 | 0.0000 | 838.5 |
| Fixed transformed mode | 48/48 | 1.000 | 1.000 | 3072/3072 | 0 | 0 | 0 | 0.7398 | 718.7 |
| Frozen selector | 48/48 | 1.000 | 1.000 | 3072/3072 | 0 | 0 | 0 | 0.6363 | 724.6 |
| 12-mode oracle | 48/48 | 1.000 | 1.000 | 3072/3072 | 0 | 0 | 0 | 0.7398 | 718.7 |
| Continuous oracle | 48/48 | 1.000 | — | — | — | — | — | — | — |

Continuous-minus-codebook coverage gap: **0.0 percentage points**. Generator-necessity states: **0**. Continuous-search unresolved states: **0**.

Nearest transformed-mode distances for generator-necessity eta: none.

Hard DB best-single B63 prevalence is 1.000, versus 1.000 on ordinary DB. Mean feasible modes/state is 9.33, versus 9.12. Hard DB is not sparser on both preregistered measures.

`READY_TO_TRAIN_CONTINUOUS_GENERATOR = NO`

No generator was trained.
