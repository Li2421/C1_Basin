## Dry-run validation

| case | requested | exact reused | missing | ambiguous |
|---|---:|---:|---:|---:|
| toy | 68528 | 68528 | 0 | 0 |
| db | 53069 | 53036 | 1 | 0 |
| recent | 27616 | 27616 | 0 | 0 |

No rollout was executed. Each dry run reconstructed exact historical request keys, then queried the global index. Semantic mismatches remain non-reusable.
