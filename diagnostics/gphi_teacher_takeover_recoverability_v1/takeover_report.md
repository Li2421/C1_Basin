# G_phi teacher-takeover recoverability audit

## Result

**K1_MODEL_EXITS_BASIN_EARLIER**

Both frozen learners were evaluated without training, fine-tuning, eta search, a gate, or cadence changes. Each continuation used dense learned H1 for exactly `k` transitions, then permanently switched to the source case's already-frozen dense fixed eta*.

## Primary recoverability

| learner | learned prefix k | success | Q_takeover | strict deadlock | timeout | B63 states |
|---|---:|---:|---:|---:|---:|---:|
| coverage | 0 | 1088/1088 | 1.0000 | 0 | 0 | 17/17 |
| coverage | 1 | 1088/1088 | 1.0000 | 0 | 0 | 17/17 |
| coverage | 2 | 1088/1088 | 1.0000 | 0 | 0 | 17/17 |
| coverage | 4 | 1086/1088 | 0.9982 | 0 | 2 | 16/17 |
| coverage | 8 | 1086/1088 | 0.9982 | 2 | 0 | 17/17 |
| coverage | 16 | 1084/1088 | 0.9963 | 4 | 0 | 16/17 |
| coverage | 32 | 1079/1088 | 0.9917 | 9 | 0 | 15/17 |
| k1 | 0 | 1088/1088 | 1.0000 | 0 | 0 | 17/17 |
| k1 | 1 | 1087/1088 | 0.9991 | 0 | 1 | 17/17 |
| k1 | 2 | 1088/1088 | 1.0000 | 0 | 0 | 17/17 |
| k1 | 4 | 1085/1088 | 0.9972 | 2 | 1 | 16/17 |
| k1 | 8 | 1074/1088 | 0.9871 | 14 | 0 | 16/17 |
| k1 | 16 | 1049/1088 | 0.9642 | 37 | 2 | 15/17 |
| k1 | 32 | 904/1088 | 0.8309 | 184 | 0 | 11/17 |

The paired `k1 - coverage` Q differences are: {"0": 0.0, "1": -0.000919, "16": -0.032169, "2": 0.0, "32": -0.160846, "4": -0.000919, "8": -0.011029}.

First loss of B63 by state (where `33` denotes `>32` internally): coverage `{'>32': 14, '32': 1, '4': 1, '16': 1}`; k1 `{'>32': 11, '32': 4, '4': 1, '16': 1}`. The k1 model loses B63 earlier in 6/17 states and later in 2/17.

## Local imitation versus recoverability

| learner | k | executed L2 mean | cosine mean | position deviation mean (m) | L2-success correlation | position-success correlation |
|---|---:|---:|---:|---:|---:|---:|
| coverage | 0 | 0.012644 | 0.998695 | 0.000000 | None | None |
| coverage | 1 | 0.246272 | 0.570635 | 0.000632 | None | None |
| coverage | 2 | 0.212428 | 0.782640 | 0.012772 | None | None |
| coverage | 4 | 0.204430 | 0.832371 | 0.033971 | 0.013065247481584509 | -0.0545167468988095 |
| coverage | 8 | 0.187603 | 0.826530 | 0.071410 | -0.04491025261754645 | -0.03445115778032658 |
| coverage | 16 | 0.163908 | 0.848984 | 0.124568 | -0.06101291978528408 | -0.06652455285772327 |
| coverage | 32 | 0.121602 | 0.667882 | 0.182546 | -0.04677424726078721 | -0.15762168218843656 |
| k1 | 0 | 0.013349 | 0.997782 | 0.000000 | None | None |
| k1 | 1 | 0.013830 | 0.998897 | 0.000667 | -0.022251658774742045 | -0.0670064899261253 |
| k1 | 2 | 0.015916 | 0.998006 | 0.001253 | None | None |
| k1 | 4 | 0.021056 | 0.990499 | 0.002723 | -0.15469955285723505 | -0.14278959592793192 |
| k1 | 8 | 0.032824 | 0.949307 | 0.007438 | -0.4221125174402921 | -0.4268557262304075 |
| k1 | 16 | 0.046089 | 0.918472 | 0.020366 | -0.6141339180157416 | -0.6704849933742223 |
| k1 | 32 | 0.078226 | 0.864053 | 0.060124 | -0.4922975442626266 | -0.5106056085616949 |


Instantaneous action error is descriptive, not causal. Geometric deviations are measured against the matched dense-eta trajectory at the same elapsed step; neither metric is used by control.

## Cohorts and safety

Historical and fresh-diagnostic cohort results are reported independently in `historical_vs_fresh.csv`. Hard-safety status: **PASS**; collision=0, execution error=0, invalid/NaN/solver failure=0.

## Interpretation and next controlled experiment

The evidence supports **direct-g vs fixed-D structured eta predictor** as the next direction. The single smallest experiment is a frozen, validation-controlled comparison of the existing direct 4-D executed-correction predictor against a fixed-D 3-D eta predictor on identical oracle data and identical dense takeover states; no Basis-as-set machinery should be added yet.

## Resources

Three GPU shards, six requested CPU cores, and 43 GiB requested memory (34.4% of 125 GiB). Maximum shard wall time: 28.66 min. The audit executed 15,232 takeover continuations and 5,299,648 physical steps.
