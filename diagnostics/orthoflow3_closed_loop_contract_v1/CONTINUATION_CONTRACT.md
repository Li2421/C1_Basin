# Continuation contract, before targeted tests

## Target

`Q(z,eta)` is task-success probability from a complete continuation snapshot,
under the frozen MACFlow and both safety projections, with eta held fixed
until success/collision/timeout. The outcome is averaged over future MACFlow
sampling randomness. The empirical B15 decision uses 15 valid successes among
the canonical 16 seed positions. It is not a population-probability guarantee.

`z` comprises physical positions, velocities, associated goals, geometry,
environment config, elapsed/remaining time, valid monitor flags, the frozen
policy/action-preprocessing convention, a reference Flow probe where used,
and replay RNG identifiers. `h=C(z)` in Phase B is canonical observation plus
canonical reference Flow; it currently omits elapsed/remaining time. Context
is physical geometry/config scalars, globally normalized on train only.

## Actual schedule and timing

Four/Ring dataset generation samples a reference Flow probe from
`CONDITIONING_FLOW_ROOT=2026100106` for conditioning. TrainingRuntime rollout
independently samples its first executed Flow using `FUTURE_ROOT=2026093011`,
physical state token, future_index, and local step0. Therefore the suffix is
an auxiliary current-policy probe, not a committed first action. Q marginalizes
the later independent Flow stream. The generator/critic select once before a
continuation; no code path reselects eta in the dynamics loop.

DoubleTrainingRuntime differs: FixedCurrent uses its conditioning action key
at step0, then future samples from step1. This distinction is explicit in its
controller identity and must be preserved by any common representation as
an action-commitment field. It is not a coordination mode.

Four/Ring reset restores `env.step_count` from the physical snapshot.
Environment termination checks the same fixed max_steps, so the remaining
horizon is preserved even though the outer Python loop allows max_steps
iterations. Live snapshots have no prior collision/success flag because these
events terminate the episode. Goal occupancy is derived from positions/goals;
the environments do not freeze individually finished agents. No runtime strict
deadlock monitor is active. There is no hidden recurrent policy state.

## Variables

### A: required or conservatively retained

- Physical positions/velocities, goal association, geometry and safety config.
- Remaining horizon; the same spatial state can have different reachability
  with one remaining step versus a full episode.
- Policy reference-frame relationship where the frozen policy is anisotropic.
  Four active rotations/relabels are not certified continuation symmetries.
- First-action commitment and action value when the continuation commits it
  (Double). Four/Ring's reference probe may be retained as a current-policy
  response feature, but must be marked uncommitted.
- Valid monitor/finished status when causally relevant; current Four/Ring
  live-state flags are derivable or false, but full snapshots retain them.

### B: replay-only or legitimately marginalized

Future RNG seed/position is needed for exact replay but is marginalized by Q.
Proposal base noise is a generator nuisance. Scenario name and database IDs
are bookkeeping, not learned targets. Four/Ring reference-probe randomness
does not determine their first continuation action.

### C: prohibited learner information

Future seeds, future actions/outcomes, oracle proposal rank, coordination-mode
IDs, test identity and DB keys may not enter h.

## Coordinate reencoding versus scene transformation

A passive chart change represents the same policy and noise in a new frame.
An active rotation uses unchanged learned weights on a different scene.
Equal Q requires equivariance of the entire eta-conditioned transition kernel,
as well as geometry, reward/termination, horizon and schedule. Physical
dynamics and projections alone are insufficient. Ring rotations have strong
tested support; Four rotations do not. Reflection remains uncertified.

Original physical rollout labels remain valid after reencoding their own
snapshots. Equal canonical encodings do not authorize transferring outcomes
to another physical snapshot. Neural h hashes are never physical DB IDs.

The critic's implemented target is an empirical conditional average given h
and eta. If h aliases different horizons or policy-frame relationships, that
average exists but need not provide reliable state-specific rankings.

## Sufficient kernel condition

Let K_eta(z,A) include MACFlow sampling, both projections, integration and
monitor update. A transformation T is continuation-value preserving if
K_eta(Tz,T A)=K_eta(z,A), terminal-success indicators agree, remaining step
counts agree, and the eta holding/selection schedule commutes with T.
Backward induction on remaining steps then yields Q(Tz,eta)=Q(z,eta).
Equivariant dynamics, projections and bases are necessary components of this
argument, not substitutes for policy-distribution equivariance. A passive
change transforms the policy's reference chart and noise/action mapping too;
an active scene change need not satisfy the kernel condition.

## Repair delta

The waiting probe confirms missing-time aliasing. The isolated repaired h
appends `1-normalized_episode_time`, sourced from the actual snapshot and
scaled physically without fitted statistics. No future value is exposed.
No physical label changes meaning, and eta remains held until termination.
The original Phase-B encoding is kept as a versioned comparator. Four's
active-policy orientation sensitivity remains a scope limitation; this repair
does not assert kernel equivariance or transfer any rotated label.
