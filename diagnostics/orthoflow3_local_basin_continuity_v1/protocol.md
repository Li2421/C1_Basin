# OrthoFlow3 local basin continuity protocol

Anchors are selected by a seeded permutation from the prior uniformly selected 32-state migration subset before attaching eta labels. Neighbors are exact replayed augmented states at t-4, t-1, t+1, and t+4 on the same original trajectory. The fixed shared probe cloud is eta=0 plus 23 points from the frozen authoritative 256-point Sobol design. Each state/probe uses eight matched future Flow identities; 8/8 is screening only.

## Frozen 64-seed cross-transfer subset

Before screening outcomes, the 64-seed directional cross-transfer subset was frozen as: all 24 `t±1` pairs, plus the 21 `t±4` pairs with the smallest remaining frozen horizon (ties: anchor rank, then offset). Each neighbor is tested with its anchor's already B63-confirmed, budgeted-migration canonical eta. Zero eta reuses the first eight screening streams and adds the remaining 56 streams; active canonical eta adds all 64. This totals 14,176 new continuations and 7,852,064 worst-case physical steps jointly with screening, inside the 15,000 / 8,000,000 hard caps.

## Execution-integrity amendment

After the user authorized six idle-server shards, cancelling two original
four-way screen shards raced with batch completion and yielded 1,120 duplicate
raw tuples. Exact tuple keys and terminal outcomes agreed, so duplicates are
excluded from all statistics. The missing unique screen tuples were completed
without re-running any already-recorded tuple. To retain the 15,000-new-
continuation cap including the unavoidable duplicate computation, six of the
preselected `t±4` directional transfers with the greatest remaining horizon
were removed **before any screening outcome was analyzed**. All `t±1` pairs
remain. See `resource_amendment.json` and the immutable amended plan.
