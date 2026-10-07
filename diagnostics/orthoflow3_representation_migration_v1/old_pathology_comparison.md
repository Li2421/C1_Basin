# P0 versus OrthoFlow3 learning-geometry check

The frozen 32 states were selected uniformly before new outcomes.  They are a
migration sanity set, not evidence for a global superiority claim.

## Capacity and target structure

- P0 robust basin: **32/32**; OrthoFlow3 robust basin: **32/32**.
- Paired basin counts: both `32`, P0-only
  `0`, OrthoFlow3-only `0`,
  neither `0`.
- OrthoFlow3 zero/active/no-basin: `18` /
  `14` / `0`.
- Distinct canonical values: P0 `7`; OrthoFlow3
  `7` (active only `6`).
- OrthoFlow3 active canonical boundary saturation: exact `0`,
  within 1% of a domain boundary `0`.
- Confirmed OrthoFlow3 robust candidate counts are lower bounds because the
  budgeted protocol promotes at most two active candidates per state.  The
  reported canonical target is therefore provisional within the promoted
  robust set, not an exhaustive minimum over every 16-stream screen survivor.

## Basis redundancy

- Median per-agent max `|cos(B_goal, middle)|`: P0 `0.998083`;
  OrthoFlow3 `4.93108e-11`.

## Interpolation

- Tested `48` interpolants between confirmed B63 endpoints
  on 8 matched streams each; `0` had at least two failures
  and therefore cannot be B63, while `48` were 8/8.
- An 8/8 staged screen is not relabeled as confirmed B63.  Sparse unresolved
  topology is reported as such.

OrthoFlow3 clearly removes the specific P0 goal/Safety collinearity in this
subset.  Whether it also simplifies canonical-target learning remains bounded
by the 32-state sample and intentionally incomplete robust-candidate coverage.
