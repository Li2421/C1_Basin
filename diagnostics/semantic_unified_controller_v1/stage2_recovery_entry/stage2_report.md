# Stage 2 — State-driven recovery entry

## Decision

**RECOVERY_ENTRY_NOT_PREDICTABLE — STOP AFTER STAGE 2.**

Persistent eta showed real counterfactual benefit at some generic NORMAL states,
but the predeclared conservative labeling procedure did not produce a usable
three-way selector.  Of 60 queried states, only 9 had a strictly supported
action and all 9 selected `ENTER_RECOVERY_NOW`; 51 remained unresolved and no
state supplied a reliable Safety/Local counterexample.  Training the proposed
three-way head on this dataset would therefore manufacture an enter-almost-
everywhere policy, a predeclared degeneracy.  No mode head was trained.

Stages 3--5 and the pristine final test remain blocked and unused.

## Generic sources and integrity

- 160 new authoritative unconditioned WIDE roots were frozen before outcomes:
  80 train, 40 validation, 40 calibration.
- One deterministic hash-uniform absolute-time anchor was predeclared per root;
  terminal-before-anchor cases were not replaced.
- Safety outcomes: 102 success, 53 timeout, 5 strict deadlock.
- 126/160 anchors were available: 59 train, 31 validation, 36 calibration.
- Overlap with audited prior assets: zero exact IC/seed/source-ID overlap.
- Source rollouts: 160/160 complete, 106,962 physical transitions, zero hard-
  safety or numerical events.

## Matched S/L/R evidence

The frozen branch cohort used 60 generic source-diverse anchors (40 train,
20 validation), 32 matched future streams, and three actions, for 5,760 complete
continuations.  `S` and `L` differed for one physical step and then shared the
same frozen Stage-1 local policy.  `R` predicted eta once and used it densely to
terminal.  There was no periodic timing, future failure label, oracle search,
or recovery exit.

| Action | Success / 1920 | Q | Timeout | Strict deadlock |
|---|---:|---:|---:|---:|
| Safety now, then semantic local | 1251 | 0.6516 | 668 | 1 |
| Local now, then semantic local | 1278 | 0.6656 | 641 | 1 |
| Enter persistent eta now | 1434 | 0.7469 | 414 | 72 |

Paired continuation outcomes were:

| Comparison | Rescue | Break | Paired delta Q |
|---|---:|---:|---:|
| Local vs Safety | 30 | 3 | +0.0141 |
| Recovery vs Safety | 271 | 88 | +0.0953 |
| Recovery vs Local | 243 | 87 | +0.0813 |

These 1,920 future continuations are clustered within 60 source states and are
not reported as 1,920 independent WIDE episodes.  Recovery's aggregate gain
coexists with substantial paired breakage and 72 deadlock terminals, so it
cannot justify indiscriminate entry.

## Label sufficiency

The frozen rule required one action's paired 90% confidence lower bound to
exceed both alternatives by the predeclared `epsilon_Q=0.02`.  Unsupported
comparisons were not converted into hard labels.

- Strictly supported recovery: 4 train, 5 validation.
- Strictly supported Safety: 0.
- Strictly supported Local: 0.
- Unresolved: 36 train, 15 validation.

This is insufficient for learning when *not* to enter recovery.  Validation-
only checkpoint selection cannot repair a one-class decision dataset.

## Implementation audit

- 5,760/5,760 branch rows complete; zero execution errors.
- All 1,920 R branches predicted eta exactly once, kept it fixed, and had
  recovery transitions equal to physical/Flow transitions.
- Current Flow replay was exact; paired branches shared the current and future
  Flow streams.
- Zero collisions, invalid actions, NaN/Inf events, or projection failures.
- Independent review found an inert constant-false exit query in the branch
  harness.  It was read-only, consumed no RNG, changed no action/state, and all
  rows completed; outcomes therefore match no-exit dense eta.  The deployment
  evaluator itself contains no exit query.
- The full-loop runner/analyzer were hardened to authenticate the actual frozen
  source manifest, exact roots/seeds/ICs, and validation/calibration allowlist.
  Eight main tests plus four source-freeze tests pass.

## Consequence

Stage 1 remains the only usable semantic controller: state-driven local
correction, with no H/L clock, achieved 178/200 on the inspected development
cohort, rescued 43/59 Safety timeouts, and introduced zero breaks.  It remains
below the frozen H8 reference (188/200, 59/59 timeout rescue).

Persistent eta remains an established strict-deadlock recovery primitive, but
this generic Stage-2 experiment did not establish a deployment-available entry
decision.  A learned exit, shared multi-head consolidation, and pristine final
test would be scientifically premature.

The smallest justified next experiment is a separately predeclared partial-
preference mode-head pilot using the already-frozen paired evidence: supervise
`R` where it is strictly supported and supervise the set `{S,L}` where recovery
is strictly disfavored, without forcing an S-vs-L label.  It must be specified
before training and validated on new development roots; it is not run here.
