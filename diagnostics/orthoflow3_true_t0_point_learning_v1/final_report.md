# OrthoFlow3 true-t0 robust eta dataset and point learning v1

## Dataset

The frozen outcome-blind search attempted **32** new independent true-t0 states and obtained **32** new exact-B63 labels. The final source-isolated split is **24/8/8**, with 40 independent source groups and no leakage. Target quality counts are `{'ROBUST_POINT_ONLY': 2, 'ROBUST_INTERIOR_STRONG': 37, 'ROBUST_INTERIOR_MODERATE': 1}`. Every held-out target itself is exact B63, so later prediction failure is not label failure. Mean search cost per new usable label was 53.3 eta queries, 3.97 Q64 promotions, 648.8 new continuations, and 391.5 worker-seconds.

Total label construction used 20,760 new continuations and 8,849,692 physical steps in 35.4 critical-path minutes.

## Training and VAL selection

All three runs retained epoch 0 by VAL MSE. Closed-loop VAL results were: seed 23: 77/128, 4/8 states 16/16, VAL eta MSE 0.5713; seed 17: 56/128, 3/8 states 16/16, VAL eta MSE 0.6093; seed 41: 36/128, 2/8 states 16/16, VAL eta MSE 0.5339. The lexicographic selector chose seed **23**, checkpoint SHA256 `e2ffd0b60a3be40e7897026eecbb634f6df1178a602379a907b8d71d1982d97d`.

## Held-out true-t0 TEST

G_POINT_T0 achieved **2/8 B63**, mean Q64 **0.5566** (285/512 successes), 130 deadlocks, 97 timeouts, and 0 collisions. Its successful-rollout J_def mean/median were 0.6261/0.6464. Matched Safety achieved 3/8 B63, mean Q64 0.7207, and 369/512 successes. Frozen G_LOWJ achieved 4/8 B63, mean Q64 0.7949, and 407/512 successes. Pairwise seed outcomes contained 78 rescues but 162 breaks relative to Safety.

Normalized eta error versus Q64 had Spearman rho 0.252 (p=0.548, n=8), so ordinary target error did not explain real reliability in this small test.

## Decision

**TRUE_T0_POINT_LEARNING_FAILS**. True-t0 robust eta prediction is **not established**: the labels are robust, but point regression generalized poorly and broke far more Safety successes than it rescued. The project should **not proceed directly to Basin/set-valued learning yet**. The single smallest justified next experiment is a nearest-h0 target-consistency/cross-transfer audit to distinguish arbitrary multimodal target assignment from insufficient h0 support before changing the learning objective.

The valid pipeline used six GPU shards at peak, about 614 MiB per worker (3,650 MiB aggregate observed), 12 CPU threads, and 42.2 minutes observed wall time from first valid label rollout through final TEST.
