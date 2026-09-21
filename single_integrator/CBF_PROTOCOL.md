# Frozen paired post-hoc safety evaluation

Run from the repository root:

```bash
python scripts/plan_baseline_400.py --out-dir results/reproduced_400
```

The available policy is **Stage-I joint Flow-BC**, not the complete MAC algorithm.
This command uses the two selected 25k checkpoints and the fixed 200-start
suite underlying MAC-only 309/400 and Safety 314/400. It also applies the
six-class stalled-timeout reclassification after paired evaluation.
No data, sampling weights, checkpoint, policy sampling, plant, initial conditions,
or deadlock thresholds are changed. Saved dataset initial states are materialized
once for both arms. The existing evaluator's rollout is reused; randomness is
indexed by experiment seed, episode index, and timestep independently of episode
length. Executed velocity changes subsequent observations, so nominal actions
need only agree until the first intervention, not throughout divergent episodes.

The protocol records source/checkpoint hashes, complete environment configuration,
initial positions, and fixed CBF constants before executing either arm. A result
is complete only when `complete.json` exists. Partial summaries after any failure
must not be treated as a completed experiment.

## Hard projection

The target is the existing speed-bounded nominal velocity, with objective
`0.5 * ||u - u_nom||²`. Pair and wall barriers use the supplied linear CBF
inequalities. Both gains are fixed at 1 s^-1 before evaluation. Pair separation
is `2 * agent_radius + agent_collision_margin + 0.0001 m`. Wall clearance is
distance to the finite segment minus robot radius, wall radius, the **unchanged**
5 mm collision-reporting margin, and 0.1 mm numerical separation. This last fixed
buffer separates the barrier boundary from the evaluator's inclusive collision
test. It is explicitly recorded, not a changed collision threshold.

All actual finite wall segments are constrained, including both branches at a
nearest-segment tie. Segment endpoint gradients use the nearest endpoint rather
than an infinite supporting line, preserving the bay opening. `min_wall_h` is the
minimum of these inflated-segment signed clearances. The evaluator starts inside
the valid workspace and stops at the first collision; no outside recovery is
introduced.

The plant's Euclidean per-agent speed limits are retained. Exact circular speed
constraints make the full projection a convex quadratically constrained problem.
The implementation solves hard linear-constraint QPs using SLSQP and adaptive
outer supporting planes for those circles. A QP optimum satisfying the original
circles is also optimal for the full projection to the declared numerical
tolerances. Each candidate is checked for feasibility and KKT residuals. There
is no inscribed speed polygon, post-solve clipping, slack, PID, priority rule,
or substitute control on solver failure. Failure aborts the experiment and saves
`experiment_error.json`; it does not become an episode timeout.

The filter requires `max(gamma, gamma_wall) * dt <= 1`. On initially nonnegative
barriers, convexity of squared pair distance and distance to a finite segment
provides a lower bound along each held-velocity interval. The prescribed CBF
inequality keeps that lower bound nonnegative in exact arithmetic. Numerical
tolerances still apply; actual swept collision checks remain authoritative.

## Outcomes and artifacts

The first terminal timestep determines exactly one of success, wall_collision,
agent_collision, safe_deadlock, or other_timeout. Any collision overrides success
and deadlock on that timestep. Simultaneous pair/wall collision retains both flags
but uses agent_collision; its count is debug-only. No solver failure is classified.
The unchanged window-progress/low-speed deadlock detector remains in the plant.

Each trace adds `u_nom`, `u_safe`, `intervention_norm`, `pairwise_h`, `min_wall_h`,
and `qp_status` to existing fields. Barrier trace values are pre-action. Episode
barrier minima include swept intervals and initial state. Completion time is null
unless the exclusive outcome is success. Intervention means norm greater than
the fixed 1e-6 m/s reporting tolerance; this does not affect control decisions.

Per-arm aggregates contain only the five outcome rates, timestep-weighted CBF
intervention rate, and successful-episode mean completion time (null if none).
`comparison.json` contains the 5-by-5 paired transition counts and conditional
row rates; unobserved rows are null. Rows are MAC-only, columns MAC+CBF. Read the
success→safe_deadlock and collision→safe_deadlock/success cells without assuming
that a particular transition must occur. Legacy evaluator reports retain their
old schema; this new protocol is explicitly versioned separately.
