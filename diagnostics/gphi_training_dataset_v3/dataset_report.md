# G_phi training dataset v3: targeted RECOVERY-zero coverage

## Result

- Dataset V2 was retained exactly and **48** independently sourced, oracle-confirmed zero-label RECOVERY states were appended.
- Final unique states / samples: **294 / 18816**.
- Categories: {'NORMAL': 120, 'PRE_DEADLOCK': 60, 'RECOVERY': 114}.
- Zero/nonzero states: **179 / 115**.
- New states use **12** root recovery source groups and 48 distinct corrected source trajectories.
- B_63-empty states: **0**; multivalued states: **0**.  Every new state has singleton eta-zero E_near and is LABEL_STABLE.

## Group-balanced split

- State counts: {'train': 203, 'validation': 41, 'test': 50}.
- Zero-label RECOVERY train/validation/test: **41 / 17 / 17**.
- Exact category/label counts are in `split_manifest.json`.
- V2 memberships and all V2 sample arrays were preserved bitwise; no source-group, state, sample, or near-adjacent trajectory leakage was found.

## Restoration, oracle, and diversity

- Exact augmented-state restoration, matched u_Flow/u_safe/u_exec, and one-step transition reconstruction: **PASS (96/96)**.
- Candidate eta-zero B_63 pass/reject: **78 / 18**; only pass states entered V3.
- New selected source trajectories are one-state-per-trajectory. Feature-space and duplicate audits are recorded in `diversity_metrics.json`.
- Oracle work: 5312 new continuations, 345727 physical steps, 0 reused; 228.3 s on one GPU shard.

## Frozen semantics

Feature dimension is **214**, target dimension is **4**, and every new target is exactly `g*_exec = 0` because `eta=(0,0,0)` satisfies the unchanged 63/64 rule. Physics, FlowBC, both hard projections, monitor/event definitions, oracle semantics, feature schema, and deterministic target are unchanged.
