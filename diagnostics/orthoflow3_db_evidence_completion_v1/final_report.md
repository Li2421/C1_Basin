# OrthoFlow3 DB mode-free pipeline: fresh-seed Q64 evidence

This extends the frozen 48-state, source-isolated Double-Bottleneck hard panel. The mode-free generator mean and the already-frozen DB-adapted W1 critic's K=16 selected proposal were each evaluated on future indices 16–63, after their indices 0–15 had been evaluated previously. No model, proposal, score, eta, controller, or threshold was changed. The critic is Toy-pretrained and full-finetuned on DB TRAIN, not an unadapted Toy-only critic. Safety and the fixed-common transformed eta use their existing compatible Q64 records.

## Cache and integrity

The preflight requested 4,608 fresh continuations: exact 0, partial 0, aggregate 0, genuinely missing 4,608. Safety/fixed baseline preflight requested 6,144 and found 6,144 exact reusable. Four CPU workers ran the missing requests and wrote 144 append-only journals, merged atomically into `shared_rollout_db/rollout.sqlite`. Postflight is 4,608/4,608 exact reuse, zero missing, ambiguous, or incompatible. The database associates 4,608 new rollout rows with this experiment, with zero quarantined conflicts, collisions, or numerical failures. True-t0 conditioning SHA checks passed for every rollout. The frozen panel has 48 distinct source groups and zero overlap with the DB critic split as audited previously.

| Method | B63 states / 48 | Mean Q64 | Success / 3072 | Mean J_def | Collision |
|:--|--:|--:|--:|--:|--:|
| Safety | 0 | 0.6768 | 2079 | 0 | 0 |
| Fixed common eta | 48 | 1.0000 | 3072 | 0.7398 | 0 |
| Mode-free generator mean | 43 | 0.9544 | 2932 | 0.5874 | 0 |
| Generator K=16 + DB-adapted critic | 46 | 0.9580 | 2943 | 0.5053 | 0 |

The mean rescues 43 B63 states relative to Safety and breaks none relative to Safety, but breaks five relative to fixed common eta. The critic rescues five mean failures but breaks two mean successes: `DB_HARD_034` and `DB_HARD_044`. Both critic-selected eta remain **0/64 success** after fresh seeds; their generator means are 64/64. Critic versus mean mean-Q64 gain is only +0.0036 (state-paired bootstrap 95% CI −0.0736 to +0.0732). Critic versus fixed breaks two B63 states and rescues none. The reduction in J_def is real on this panel, but cannot outweigh the two catastrophic selection errors for a robustness-first deployment decision.

DB off-anchor proposal evidence is informative about what is missing in TRAIN: 40/48 hard states have both successful and failed K=16 proposals; 29/48 have an observed B15/non-B15 pair separated by at most 0.10 in normalized eta coordinates. The DB generator TRAIN source used only 12 old anchors on 64 states, with zero off-anchor pairs. At the same time, the hard-panel proposal Basin is not generally narrow: median 13/16 proposals are B15, and only two states have four or fewer B15 proposals. This is boundary-evidence deficiency, not proof that a richer or multimodal generator is needed.

The compatible database still has no fixed-fail/adaptive-success DB cohort: 180 profiles with complete standard-16 fixed-common outcomes are all B15; eight more have 15/15 successes and are already B15-certified. Historical superficially plausible exceptions under a different controller fingerprint remain excluded. Thus this Q64 experiment confirms that the current mode-free generator helps markedly over Safety and that the critic can rescue some mean failures, **but it does not show any advantage over fixed common eta**.

## Remaining evidence before stronger claims

1. A new source-isolated, outcome-blind DB panel containing verified fixed-fail/adaptive-success states is necessary for a claim that adaptive eta generation beats fixed eta. Existing compatible data do not provide such a panel; no new basin search was started here.
2. A genuinely narrow-basin DB panel and proposal-matched critic validation are needed for a claim that the critic selects reliably across DB difficulty regimes. The present wide-basin panel exposes two stable catastrophic misrankings.
3. For a final two-scenario paper-level robustness claim, independently confirm the selected Toy pipeline at Q64. This task did not run Toy rollouts.

Key artifacts: `frozen_q64_manifest.json`, `cache_preflight.json`, `cache_postflight.json`, `q64_per_state.csv`, `q64_summary.json`, `db_offanchor_boundary_audit.csv`, `db_offanchor_boundary_summary.json`.
