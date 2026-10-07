# Global rollout database migration audit

Scanned 132 experiment directories and 4634 candidate files.
Imported 923,092 unique seed-level rollouts and 48,353 unique aggregate records; 370,846 duplicate observations were deduplicated.
Conflicts quarantined: 2,084. Ambiguous/incompatible records remain indexed but are excluded from automatic reuse.

## Scenario coverage

| scenario | state-eta pairs | >=16 compatible seeds | coverage | observed B15 |
|---|---:|---:|---:|---:|
| ToyGiveWay | 20283 | 6644 | 0.328 | 5456 |
| DoubleBottleneck_4A | 13916 | 3393 | 0.244 | 2698 |

## Dry-run validation

| case | requested | exact reused | missing | ambiguous |
|---|---:|---:|---:|---:|
| toy | 68528 | 68528 | 0 | 0 |
| db | 53069 | 53036 | 1 | 0 |
| recent | 27616 | 27616 | 0 | 0 |

No rollout was executed. Each dry run reconstructed exact historical request keys, then queried the global index. Semantic mismatches remain non-reusable.
