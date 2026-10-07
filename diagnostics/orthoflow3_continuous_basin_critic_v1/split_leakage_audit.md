# Split leakage audit

## State groups

Frozen source-group splits from the prior selector datasets were reused. Pairwise source-group overlaps are `{"DB": {"train_test": 0, "train_val": 0, "val_test": 0}, "Toy": {"train_test": 0, "train_val": 0, "val_test": 0}}`; all are zero.

## Eta groups

Exact eta IDs were not independently randomized. Eta points were joined into connected components whenever normalized distance was <=0.05, then entire components were assigned together using outcome-blind availability balancing.

- Unique eta: 4,649; components: 708.
- Test→train nearest distance: minimum 0.0500, median 0.1096, mean 0.1074.
- No eta UID appears in multiple splits, and no cross-split pair is closer than the grouping radius apart from floating-point equality at the 0.05 boundary.

The main simultaneous holdout contains Toy 268 and DB 64 pairs. DB coverage is scientifically limited and is reported as such.
