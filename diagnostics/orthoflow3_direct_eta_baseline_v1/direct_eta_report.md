# OrthoFlow3 Direct-eta baseline v1

## Result

**DIRECT_ETA_PROMISING_BUT_IMPERFECT**

The target is `eta*_24-robust-lowJ`: zero when zero is B63, otherwise the minimum mean-successful full-horizon J_def among confirmed B63 candidates in the frozen 24-point cloud. It is **not** an exact global continuous canonical eta.

## Target and model

- Target resolution: `{"status": "READY_FOR_TRAINING", "test": {"ACTIVE": 4, "TARGET_SINGLE_ROBUST": 0, "TARGET_UNRESOLVED": 0, "ZERO": 16}, "train": {"ACTIVE": 30, "TARGET_SINGLE_ROBUST": 0, "TARGET_UNRESOLVED": 0, "ZERO": 50}, "train_resolved_fraction": 1.0, "val": {"ACTIVE": 6, "TARGET_SINGLE_ROBUST": 0, "TARGET_UNRESOLVED": 0, "ZERO": 14}}`
- Source-group leakage: none (exact Q-v2 80/20/20 split).
- Model: 214->128->128->3, 44419 parameters, selected seed 23.
- Checkpoint SHA256: `bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7`.

## Held-out TEST

- Pointwise metrics: `{"active": {"normalized_l2_mean": 0.1831219643354416, "normalized_l2_median": 0.15988963842391968, "per_dimension_mae": [0.12420706450939178, 0.06094777211546898, 0.018700670450925827], "physical_l2_mean": 0.1445484459400177, "physical_l2_median": 0.12105896323919296, "states": 4}, "all": {"normalized_l2_mean": 0.19294825196266174, "normalized_l2_median": 0.08761380612850189, "per_dimension_mae": [0.11910077184438705, 0.058369409292936325, 0.013778885826468468], "physical_l2_mean": 0.1549796164035797, "physical_l2_median": 0.08344335108995438, "states": 20}, "constant_normalized_mse": 0.21164774894714355, "ridge_normalized_mse": 8.677248887726325, "zero": {"normalized_l2_mean": 0.19540482759475708, "normalized_l2_median": 0.07722186297178268, "per_dimension_mae": [0.11782419681549072, 0.057724811136722565, 0.01254843920469284], "physical_l2_mean": 0.1575874239206314, "physical_l2_median": 0.07484456896781921, "states": 16}}`
- Predicted B63: 16/20; mean Q64 0.9180.
- Target B63: 20/20; zero B63: 16/20.
- Zero harmful false activations: 4; harmless nonzero: 12.
- ACTIVE successes: 4; basin misses: 0; near-zero misses: 0.
- Read-only frozen-Q mean on Direct-eta successes/misses: 0.9965179041028023 / 0.19988055221438117.

## Fresh WIDE 200

- Safety: 140/200 success, 4 deadlock, 56 timeout, 0 collisions.
- Direct-eta: 150/200 success, 29 deadlock, 21 timeout, 0 collisions.
- Rescue / break / net: 53 / 43 / 10.
- Successful mean J_def, Safety / Direct-eta: 0.0 / 0.29051247935021624.
- Eta norm median/mean/max: 0.4549 / 0.5860 / 1.3292; clipping 0/200.
- Hard safety: zero collisions and invalid actions; the second projection remained authoritative.

Historical P0 context (different cohort, not paired): 90/200 success, 33 rescues, and 86 breaks versus its 143/200 Safety reference.

## Interpretation

OrthoFlow3 Direct-eta is judged substantially more viable than the historical P0 step-0 controller: **True**. Q-guided training status: **promising and justified to test, but not yet proven necessary**.

Smallest justified next experiment: Train one otherwise identical G with frozen-Q feasibility loss as a controlled ablation, keeping split and target data fixed.

No Q, J, Direct-g, gate, mixture, flow, recovery, or cadence model was trained or used to alter Direct-eta behavior.
