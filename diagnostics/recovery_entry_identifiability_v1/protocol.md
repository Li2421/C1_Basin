# Recovery-entry identifiability audit protocol

Status: **frozen before new outcome evaluation**.

## Objective

Determine whether the existing deployment feature `h_t in R^214` contains
enough information to predict the matched complete-continuation advantage of
entering the already-frozen persistent structured-eta recovery option now,
relative to continuing Safety.  This is a representation/probe audit, not
production-gate training.

## Frozen branches

- `N`: continue authoritative Safety to the original 850-step horizon.
- `R`: at the queried state predict eta once with checkpoint
  `2481028b...`, latch it, run dense structured-eta recovery with the exact
  frozen learned exit rule from `single_segment_recovery_training_v1`, then
  Safety forever.  No re-entry.
- The current Flow realization and every future random stream are paired
  exactly between N and R.  Switching never resets physical state, history,
  monitor state, timers, latches, absolute time, or RNG continuation.

No Direct-g, H8, fixed burst, online oracle, controller retraining, new input
feature, or future outcome enters either branch or any probe input.

## Development sources and states

- 120 new authoritative unconditioned WIDE roots.
- IC root seed `2026102601`; Flow root seed `2026102602`.
- Two deterministic source-ID-derived uniform absolute transition indices in
  `[0,849]`, without replacement, are frozen per root before rollout.
- A requested state is materialized only if Safety is nonterminal immediately
  before that physical transition; unavailable requests are not replaced.
- Selection does not use terminal time/type, geometry, prior errors, or branch
  outcomes.  All derived states from one root remain grouped.

## Branch evidence and budget

- Initial block: 16 matched futures per state and branch.
- Paired future-stream root seed: `2026102603`.
- Global maximum: 12,000 new branch continuations.
- If an additional block is needed, add exactly 16 fresh matched futures to at
  most 135 states.  Eligibility is predeclared as an initial paired 90% CI for
  `DeltaQ` crossing either `-0.05`, `0`, or `+0.05`; rank eligible states by CI
  width, then deterministic state hash.  Freeze the escalation manifest before
  executing it.  Do not repeatedly inspect after the confirmation block.
- Official horizon remains absolute step 850 at `dt=0.05`; there is no horizon
  restart.

Primary statewise quantities are `Q_N`, `Q_R`, `DeltaQ`, paired rescue, and
paired break.  Hard labels never replace continuous advantage in regression.

## Probe protocol

- Primary input is exactly the frozen 214-D deployment feature.
- Repeated 5-fold source-grouped evaluation is primary; one random-state split
  is a shortcut diagnostic only.
- Ridge regression and one small `214 -> 64 -> 64 -> 1` SiLU MLP are allowed.
  MLP seeds are 17, 23, and 41; selection is internal-validation only.
- Binary diagnostic labels are frozen at beneficial `DeltaQ >= 0.05`, harmful
  `DeltaQ <= -0.05`, ambiguous otherwise.
- Utility sensitivities use fixed break weights `{1,2,4}`.  Thresholds are
  selected without held-out-fold labels.
- Nearest-neighbor and documented feature-block ablations are diagnostic only.
- A secondary augmented-state probe is permitted only if 214-D grouped probes
  are weak and only for deployment-available omitted fields.

No production entry checkpoint will be selected or deployed.  No final test is
created or consumed.
