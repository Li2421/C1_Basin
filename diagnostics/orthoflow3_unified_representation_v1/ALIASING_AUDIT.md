# Aliasing audit

No exact learned-embedding aliases were found among 245 original train/development states at the registered tolerance. This is a finite-corpus check, not an injectivity theorem. The remaining-horizon field from the contract repair is retained.

Nearest-neighbor statistics:

```json
{
  "double_bottleneck": {
    "h_distance": 0.029357410967350006,
    "robust_jaccard_common_tested": 1.0,
    "Q_mean_absolute_difference": 0.026041666666666668,
    "Q_ranking_spearman": 0.9999999999999999
  },
  "four_way_intersection": {
    "h_distance": 0.20397348701953888,
    "robust_jaccard_common_tested": 0.9672131147540983,
    "Q_mean_absolute_difference": 0.0,
    "Q_ranking_spearman": null
  },
  "ring_exchange": {
    "h_distance": 0.02935820911079645,
    "robust_jaccard_common_tested": 0.6746684350132626,
    "Q_mean_absolute_difference": 0.0,
    "Q_ranking_spearman": null
  }
}
```

Jaccard and Q comparisons use common evaluated evidence only; missing eta are not negatives. See aliasing.json for common-evidence counts, physical-summary distances and ranking correlation, and nearest_neighbor_diagnostics.svg. Near-neighbor differences are not by themselves aliasing defects. Monitor-history support for intermediate Double states remains explicitly out of scope: the parser rejects them rather than resetting history.
