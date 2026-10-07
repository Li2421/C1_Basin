# Stable-oracle feature audit

## Decision

**CURRENT_214D_APPEARS_SUFFICIENT**

Primary analysis reused the prior audit's stored stability judgment and excluded every oracle-ambiguous state. No rollout, controller, oracle, B_63, feature, G_phi, or closed-loop behavior was changed.

## Cohort

- Audited states: **21 stable / 21 ambiguous**.
- Stable labels: {'0': 10, '1': 11}; stable difficult states: 13.
- Fully stable matched zero/nonzero boundary pairs: **2 / 15**. Both were correctly separated by the existing gate.
- Existing gate correct on stable states: **19/21**. The two stable errors (one train boundary nonzero and one old-hard test zero) were not explained by an omitted varying snapshot field.

## Representation audit

- Stable-train primary matrix has 16 constant and 1 near-constant dimensions. The broader V4-train state-mean cross-check has 7 constants.
- Across V4-train state means, 147 dimensions participate in 18 |r|>=0.995 clusters; 129 dimensions can be removed by retaining one representative per cluster. This confirms substantial overcompleteness, not that it harms prediction.
- Stable-train state-mean numerical/effective rank: 6/3.02 (only 7 independent states). Flow-expanded rank/effective rank: 33/3.32. Contextual all-V4-train state means need only 13 PCs for 95% variance, although PCA was not used as an input.
- Stable 214-D nearest-neighbor accuracy: 0.857; source-group-aware: 0.857; difficult-stable only: 0.846. Opposite-label neighbors are closer than same-label neighbors for 3/21 stable states, including 2/13 difficult stable states.

## Lightweight source-group-held-out diagnostics

- Full 214-D, all stable: AUROC 0.991, balanced accuracy 0.950.
- Full 214-D, difficult stable only: AUROC 0.762, balanced accuracy 0.833.
- Strong-correlation pruned (78 D), all stable: AUROC 0.991, balanced accuracy 0.850.
- Strong-correlation pruned, difficult stable: AUROC 0.881, balanced accuracy 0.750.

Pruning changes the small-sample metrics but does not consistently improve both AUROC and balanced accuracy. Current-control/projection and combined geometry/history blocks carry the strongest stable-label signal; geometry alone is weaker. Treat all classifier numbers as diagnostic because there are only 21 stable states and 10 source groups.

## Missing-state check

The ordered 41x2 goal-error history, last executed velocity, candidate timing, current/max stuck timers, historical deadlock latch, current active CBF identities, residual margins, barriers, u_Flow, and u_safe are already represented. The only fields in the exact restorable snapshot but absent from 214-D are four first-terminal-event latches and `done`; all are constant in these pre-terminal inputs. Adding them leaves the diagnostic result unchanged. Richer action/position/correction histories are not present in the current restorable augmented state and were therefore not used as evidence of missing features.

## Interpretation

The feature vector is overcomplete, but the evidence does **not** show that redundancy causes the stable-boundary failure, nor that a varying deployment-available restorable field is missing. Stable matched pairs are separated and the full input retains meaningful source-group-held-out signal. The remaining issue is more consistent with the very small number of stable boundary states and model fitting/generalization. This is separate from the 21/42 audited states already known to have oracle-label instability.

Smallest justified next experiment: Repeat the same oracle-stable, source-group-held-out gate fit after collecting a small number of additional statistically stable close-range pairs from new source groups; keep oracle-ambiguous states excluded.
