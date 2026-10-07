# OrthoFlow3 local basin continuity audit

## Result

**Classification: `BASIN_LOCALLY_STABLE_CANONICAL_DISCONTINUOUS`.** Across 12 uniformly selected provenance-valid anchors and 48 exact same-trajectory neighbors (`t±1`, `t±4`), the 24-point common eta Q-fields were almost unchanged. The conclusion is empirical: it supports local stability of the sampled success field, not mathematical basin connectedness.

| separation | pairs | mean / median `|ΔQ8|` | mean / median S8 Jaccard | mean / median Spearman (available pairs) | directional B63 retained |
|---|---:|---:|---:|---:|---:|
| `±1` | 24 | 0.002604 / 0.000000 | 1.000000 / 1.000000 | 0.999458 / 1.000000 | 24/24 |
| `±4` | 24 | 0.005859 / 0.000000 | 0.995288 / 1.000000 | 0.996619 / 1.000000 | 14/15 |
| all | 48 | 0.004232 / 0.000000 | 0.997644 / 1.000000 | 0.998038 / 1.000000 | 38/39 |

The selected directional robust transfers have mean `Q64=0.995994`. All `±1` transfer tests retained B63; `14/15` selected `±4` tests did. The one non-retention was an anchor zero eta at `t-4`, `58/64` success (six timeouts); its common-cloud mean `|ΔQ8|` was only `0.005208`. It is not bidirectional evidence of a genuine basin switch.

## Valid nearby-state construction

Neighbors are exact replay states on the same frozen trajectory, preserving physical state, Flow continuation identity, history, monitor, timers, latches and absolute timestep. No feature perturbation or reconstructed physical-only state was used. Across all pairs aggregate position displacement had mean/median `0.021619/0.011235`, relative-geometry change `0.023470/0.011451`, and normalized 214-D feature distance `2.293711/1.597751`. No sampled pair crossed a discrete monitor transition or changed its stuck timer.

## Zero boundary and canonical selection

`eta=0` changed its 8-seed status on only `1/48` pairs. The promoted zero transition above is a local feasibility boundary candidate, but not a dominant pattern and not a confirmed far-away active-basin relocation. Canonical neighbor targets were not searched in this strictly bounded continuity audit; consequently direct per-pair canonical-jump vs set-jump and bidirectional neighbor-canonical transfer remain unavailable. The prior documented non-smooth canonical representative together with the present stable local Q-fields supports the interpretation that canonical selection is the more likely source of the apparent non-smoothness, while this audit does not prove it at every pair.

## Set distance and J_def

The screening `S8` / `S7` overlaps are reported, but robust sampled Hausdorff/centroid distances are marked `UNDERRESOLVED`: neighbor robust sets were not re-searched and 8/8 is not B63. J_def is secondary; directional canonical-transfer J_def is stored, but a changing neighbor minimum-J_def point cannot be inferred without neighbor canonical optimization.

## Safety and integrity

There were zero agent/wall collisions, invalid actions, numerical errors, or projection failures in all retained screening and cross-transfer rollouts. Of 12,640 raw screening records, 1,120 deterministic cancellation-race duplicates were verified identical on success/outcome/terminal step and excluded, leaving exactly 11,520 required unique screening tuples. Full details are in `integrity_checks.json` and `resource_amendment.json`.

## Learning implication

The evidence supports `BASIN_AWARE_DIRECT_ETA` over treating canonical-point MSE as a sufficient representation of the local feasible set. It does **not** justify a generative/multimodal model or a claim of true local basin switching. The smallest justified next experiment is a bounded, source-uniform neighbor-oracle confirmation on a small predeclared subset of these pairs, producing both neighbor canonical representatives and two-way B63 cross-transfer; no model training should start automatically.
