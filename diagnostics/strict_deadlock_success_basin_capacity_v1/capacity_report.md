# Strict-deadlock success-basin capacity audit

**Aggregate classification: ETA_CAPACITY_PRESENT_FOR_STRICT_DEADLOCK.** Dominant remaining bottleneck: **learned G_phi approximation/deployment**.

## Integrity and frozen semantics

- All 17 authoritative Safety strict deadlocks reproduced exactly; maximum differences for state, Flow key, projected action, and monitor fields were all zero.
- 102 complete augmented query states were restored: S0, 8 s, 4 s, 2 s, 1 s, and immediately pre-detector for every episode.
- Eta remains fixed for the continuation; goal/safe/relative bases are recomputed at every physical step.
- Continuation horizon is the remaining global frozen episode horizon through step 850, matching the G_phi oracle-label pipeline.
- Search is confined to the authoritative Phase-A envelope goal=[0.5,1.25], safe=[-0.5,0.5], relative=[0,0.75].

## Episode-level capacity

| Cohort | Episodes | Any exact success | Robust >=63/64 | Early robust | Late robust | Fragile only | No basin |
|---|---:|---:|---:|---:|---:|---:|---:|
| historical | 11 | 11 | 11 | 11 | 0 | 0 | 0 |
| fresh_unseen | 6 | 6 | 6 | 6 | 0 | 0 | 0 |
| combined | 17 | 17 | 17 | 17 | 0 | 0 | 0 |

Robust capacity before detector onset: **17/17 (100.0%)**.
Earliest robust-basin lead-time statistics (seconds): `{"count": 17, "max": 33.15, "mean": 26.232352941176472, "median": 27.1, "min": 8.0, "p90": 31.85}`.
Robust candidate success-count statistics: `{"count": 249, "max": 64.0, "mean": 63.967871485943775, "median": 64.0, "min": 63.0, "p90": 64.0}`.
Exact-flow successful-cell fraction across rescuable queried states: `{"count": 102, "max": 0.9671052631578947, "mean": 0.7384526960928625, "median": 0.7804054054054054, "min": 0.30952380952380953, "p90": 0.8843117408906882}`.

## Projection

Projection is classified as **not the primary limiting factor** by the predeclared descriptive comparison of rewrite fractions for exact successes versus non-successes.
Successful exact mean rewrite/raw ratio: 0.8284821883279052; non-success ratio: 0.8929056901156193.

## Smallest next experiment

Add only the audited early robust-deadlock states and their minimum-J_def B63 targets to the unified supervised set, retrain one unchanged G_phi, then repeat frozen H=8 evaluation on a new held-out wide cohort.
