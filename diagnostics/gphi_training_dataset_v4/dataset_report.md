# G_phi Dataset V4: recovery intervention-boundary coverage

## Result

- Appended **30** unique close-range RECOVERY boundary states to the bitwise-preserved V3 prefix: **15 zero / 15 nonzero**.
- These states use **8** independent root source groups and 29 distinct trajectories.
- Explicit matched zero/nonzero pairs: **15**.
- Final states/samples: **324 / 20736**; categories: {'NORMAL': 120, 'PRE_DEADLOCK': 60, 'RECOVERY': 144}.
- Final zero/nonzero states: **194 / 130**.
- B_63-empty selected states: **0**; multivalued quarantines: **0**.

## Boundary split

New close-range zero/nonzero counts by split: `{'train': {'zero': 9, 'nonzero': 9}, 'validation': {'zero': 2, 'nonzero': 2}, 'test': {'zero': 4, 'nonzero': 4}}`. All Flow variants remain with their augmented state and root source group. No source-group, state, sample, or trajectory leakage was found.

## Integrity and learnability

- Exact restoration audit: **PASS**.
- V3 sample prefix preserved exactly for every array: **True**.
- Distance range: 0.320189–0.547293 m.
- Nearest-neighbor cross-label disagreement: 50.00%; matched and feature-group diagnostics are in `boundary_diversity_metrics.json`.
- Target construction is unchanged: B_63 -> minimum J_def -> E_near -> hard-projected executed correction. Exact zero labels arise only from eta=0 membership, never thresholding.

## Runtime

Eta-zero/new-candidate oracle used two nonoverlapping GPU shards. New rollout counts were 2999 eta-zero plus 7680 nonzero-candidate evaluations. No valid tuple was rerun.
