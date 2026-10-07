# Phase-B conditioning transform

The derived dataset keeps all original physical state IDs and eta/Q evidence.
Only the neural encoding and its train-fitted normalization change. The fixed
MACFlow, plant, hard-safety projections, eta domain and three fields are used
as before. No label is transferred to a rotated physical state.

## Ring

For agent position p, use outward radial p/|p| and its CCW tangent (-y,x).
Express the stored reference Flow vector in that same chart used by Ring's
native observation. Sort agents by the rotation-invariant tuple (radius,
local goal displacement, local velocity), rounded to 10 decimals only for
deterministic ordering. Rebuild all relative-agent blocks in that order.
The tangent convention is a coordinate chart, with no circulation label.

## Four-Way

Enumerate the four quarter-turn rotations about the origin. For each, sort
agents by their rotated goal, position and velocity. Select the lexicographically
smallest flattened (goals,positions,velocities) key, rounded to 10 decimals.
Rebuild the native 4x18 observation and rotate/reorder the actual reference
Flow vectors with the same transform. This uses present physical data and
contains no crossing order or priority. Store the inverse chart metadata for
provenance, while eta remains three scalar coefficients.

The frozen Four-Way MACFlow is orientation-sensitive. A passive transformation
of a given Flow vector commutes exactly with this encoding. Recomputing Flow
after actively rotating the scene can produce another vector and another
encoded suffix. Accordingly passive input equality is not a claim of equality
of active closed-loop transition kernels. Active tests and held-out replay
must retain and report that distinction.

## Normalization and context

Four/Ring normalization pools the same feature across all four agent slots;
the action suffix pools across four agents as well. Double retains its native
flattening and serves as a control. All statistics are fitted on train states
only. Physical descriptor scalars use global train-only context statistics;
the redundant explicit scenario one-hot is removed. Existing input adapters
remain because native dimensions differ. No branch is selected from a
coordination label.

## Training and evidence reuse

Generator architecture, robust-point likelihood objective, bounded variance,
optimizer and budget are inherited. Critic architecture and empirical-Q BCE
are inherited from Phase A; the same v2 full-Q and Phase-A proposal-aligned
train/dev evidence is re-encoded. Numerical-uncertified Q targets remain
excluded. This keeps physical evidence fixed for the representation comparison.
The new generator's proposals receive their own cache-first development
evaluation; labels on old proposals are never asserted to apply to new eta.

Passive tests: all 160 primary-scenario v2 states, four rotations and two
consistent permutations, 1,280 cases; maximum encoding discrepancy 0.
