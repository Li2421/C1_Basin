# Flow-variant representation collision audit

## Scope and integrity

This audit used only exact saved V4 `h(z, xi)` inputs. It trained no model, generated no state or rollout, and changed neither the 214-D schema nor oracle labels. Distances are normalized RMS distances using the exact outer-fold TRAIN-derived normalization; binary dimensions remain literal, matching the prior gate pipeline.

The two collision neighbors were frozen from the prior local audit. The control was frozen before inspecting these cloud metrics as oracle-audit `boundary_pair_10`, an already `CLEARLY_SEPARATED` stable pair involving a non-collision zero state.

## Results

### collision_pair_1

- Fixed pair: `RBV_Q_pair228_m080_s95401003_p030` (gate=0) vs `R_D4_s95101006_p60` (gate=1).
- Saved variants: 64 / 64; exact matched seeds: 0.
- Common state-static / union Flow-dependent dimensions: 178 / 36.
- Mean within-cloud distance: 0.130272; mean cross-cloud distance: 0.413623; cross/within: 3.175064.
- Opposite-state nearest-neighbor rate: 0.00%.
- Leave-one-out 1/3/5-NN accuracy: 100.00% / 100.00% / 100.00%.
- Energy distance 0.570772 (permutation p=0.0005); MMD² 1.132086 (p=0.0005).
- Separating feature groups: geometry_observation|goal_relative|inter_agent_relative|history_monitor|episode_time|u_flow|u_safe|control_projection.
- Classification: **STATE_SUMMARY_COLLISION_ONLY**.

### collision_pair_2

- Fixed pair: `RB_Q_pair226_m080_s95400802_p073` (gate=0) vs `R_D1_s95106004_p40` (gate=1).
- Saved variants: 64 / 64; exact matched seeds: 0.
- Common state-static / union Flow-dependent dimensions: 181 / 33.
- Mean within-cloud distance: 0.055718; mean cross-cloud distance: 0.256874; cross/within: 4.610256.
- Opposite-state nearest-neighbor rate: 0.00%.
- Leave-one-out 1/3/5-NN accuracy: 100.00% / 100.00% / 100.00%.
- Energy distance 0.404053 (permutation p=0.0005); MMD² 0.764422 (p=0.0005).
- Separating feature groups: geometry_observation|goal_relative|inter_agent_relative|history_monitor|episode_time|u_flow|u_safe|control_projection.
- Classification: **STATE_SUMMARY_COLLISION_ONLY**.

### control_pair_boundary_10

- Fixed pair: `RB_Q_pair228_m080_s95401001_p050` (gate=0) vs `RBV_Q_pair228_m080_s95401004_p036` (gate=1).
- Saved variants: 64 / 64; exact matched seeds: 64.
- Common state-static / union Flow-dependent dimensions: 176 / 38.
- Mean within-cloud distance: 0.112553; mean cross-cloud distance: 0.528054; cross/within: 4.691597.
- Opposite-state nearest-neighbor rate: 0.00%.
- Leave-one-out 1/3/5-NN accuracy: 100.00% / 100.00% / 100.00%.
- Energy distance 0.834520 (permutation p=0.0005); MMD² 1.070475 (p=0.0005).
- Separating feature groups: geometry_observation|goal_relative|inter_agent_relative|history_monitor|episode_time|u_flow|u_safe|control_projection.
- Classification: **CONTROL_CLEANLY_SEPARATED**.

## Matched-seed interpretation

The two primary collision pairs have no identical saved Flow seed IDs across their members, so an exact matched-seed comparison is unavailable and no index-based pseudo-pairing was used. Their valid comparison is distributional (all 64×64 cross distances versus within-cloud Flow variation). The control pair has 64 exact matched seeds; its detailed matched distances are saved separately.

## Decision

Overall classification: **REPRESENTATION_INFORMATION_PRESENT**.

This conclusion is restricted to the two audited states and does not establish class-wide source-group generalization.

Smallest justified next experiment: Replay all 64 saved variants of these two states through their existing source-group-held-out gate folds and report per-variant probability distributions; this directly tests whether the robust gate mistakes persist across xi despite the available input separation.

Permutation p-values are cloud-identity diagnostics, not independent-state evidence: the 214 coordinates are correlated and all 64 variants within a cloud share the same augmented state/static coordinates.
