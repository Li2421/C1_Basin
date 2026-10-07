# OrthoFlow3 current-pipeline MUST DO evidence gaps

The independent Double-Bottleneck hard panel contains 48 source-distinct true-t0 states, with zero source-group overlap with DB generator/critic TRAIN, VAL, or TEST. All methods used the same 16 frozen continuation indices and the same authoritative OrthoFlow3 basis, Flow checkpoint, hard-safety projections, horizon, and true-t0 conditioning. The mode-free generator checkpoint and DB-adapted continuous-Q checkpoints were frozen before these outcomes. No model was trained, no manual mode ID was used, and no unrestricted critic optimization was performed.

| DB hard panel method | B15 states / 48 | Mean Q16 | Mean J_def | Collision | Numerical failure |
|---|---:|---:|---:|---:|---:|
| Safety | 7 | 0.6823 | 0.0000 | 0 | 0 |
| Fixed common eta | 48 | 1.0000 | 0.7399 | 0 | 0 |
| Mode-free generator mean | 44 | 0.9492 | 0.5897 | 0 | 0 |
| Generator K=4 oracle | 47 | 0.9961 | 0.5322 | 0 | 0 |
| K=4, DB-adapted critic | 44 | 0.9154 | 0.5463 | 0 | 0 |
| Generator K=16 oracle | 48 | 1.0000 | 0.5183 | 0 | 0 |
| K=16, DB-adapted critic | 46 | 0.9570 | 0.5071 | 0 | 0 |

The oracle rows are diagnostic upper bounds, not deployable selection. K=4 proposal coverage is 47/48; K=16 coverage is 48/48. The critic was **not Toy-only**: it is the pre-existing Toy W1 critic fully fine-tuned on DB TRAIN with DB's 80-D state adapter, used here as a frozen three-seed logit ensemble. Its DB TRAIN/VAL/TEST source groups do not overlap this hard panel.

At the B15 state level, generator mean versus Safety has 37 rescues and 0 breaks; versus fixed common eta it has 0 rescues and 4 breaks. At matched-seed level, mean versus Safety has 227 rescues and 22 breaks; versus fixed it has 0 rescues and 39 breaks. All 4 mean-failure states (`DB_HARD_011`, `040`, `042`, `047`) have B15 proposals, and both K=4 and K=16 critic selections recover them. Yet K=4 breaks 4 states on which mean was B15; K=16 breaks 2 (`DB_HARD_034`, `044`). On those two K=16 failures, the selected eta is 0/16 despite 7/16 and 11/16 proposals, respectively, being B15. Thus finite-proposal ranking has some real rescue value relative to mean, but it is not reliable enough to match the fixed common eta. The paired net improvement of K=16 versus mean is only 2/48 states (4 rescues, 2 breaks; two-sided exact paired p=0.6875), not a stable advantage claim. Increasing K from 4 to 16 closes the proposal-coverage gap but exposes additional high-score bad proposals.

The compatible global DB search found no existing fixed-fail/adaptive-success cohort: 180 complete standard-16 fixed-common profiles are all B15, and another 8 are already 15/15 success, which certifies B15 even if the 16th seed fails. Nineteen superficially plausible historical profiles were excluded because their controller fingerprint lacks the authoritative OrthoFlow3 basis/conditioning. A [minimum discriminative plan](minimum_discriminative_plan.md) is recorded; it was not executed.

The Toy K=16 offline audit contains a [full 13×16 proposal table](toy_critic_13_proposals.csv). All 13 are within-state ranking errors by definition. Descriptive, overlapping indicators: 8 boundary-overestimation proxies, 6 selected-proposal probability overpredictions of at least 0.2, 4 eta-distribution-shift proxies (nearest critic TRAIN eta >0.2 normalized), and 3 severe overconfidence cases. A B15 solution is within the critic top-2 in 4/13 states and top-3 in 6/13. These flags identify association, not proven causal mechanisms; no critic was changed.

The initial mean preflight requested 768 new continuations and the proposal preflight requested 12,288. All 13,056 were executed, written through append-only journals, merged into the global rollout database, and are now exact-reusable: mean postflight 768/768, proposal postflight 12,288/12,288, missing/ambiguous/conflict/collision/numerical-failure 0.

Remaining evidence gaps before a paper-level claim are: (1) an independent source-isolated DB cohort that actually contains fixed-fail yet eta-recoverable states, without changing this frozen panel; (2) fresh-seed Q64 confirmation of a finally chosen mode-free deployment policy; and (3) stronger DB off-anchor boundary/narrow-basin evidence if claiming general state-conditioned generation. This experiment does **not** establish that the current generator/critic beats the fixed common eta in DB. No new model or larger basin search was started.
