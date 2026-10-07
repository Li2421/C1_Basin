# Versioned basis-family interface

Future eta-oracle and dataset jobs must carry an explicit `basis_family` in
their plan, arm, result, cache key, and manifest.  The two supported values are
`p0` and `orthoflow3`; missing values are rejected in new pipelines.  Legacy
artifacts may be interpreted as P0 only when their enclosing frozen schema and
hash identify them as historical P0 data.

The implementation is `shared_control/basis_families.py` and returns three
per-agent `[N,2]` fields plus immutable metadata (`family`, `version`, names,
dimension, normalization/clipping description, source hash).  Both families
have `D=3`.

## P0 (`p0_legacy_v1`)

The historical source files remain byte-unchanged.  The selected fields are:

1. `B_goal = bounded_rows(goals - positions, max_speed)`
2. `u_safe`
3. `B_rel = mean_{j != i} bounded_rows(p_i - p_j, max_speed)`

Each relative pair is bounded before averaging.  The migrated selector is
tested by exact array comparison against both historical P0 implementations.

## OrthoFlow3 (`p1_orthoflow3_v1`)

The authoritative implementation is the archived
`diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py` function set
`raw_ortho_flow`, `basis_terms`, and `correction`, SHA256
`51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`.

The interface reproduces its fields exactly:

1. `B_goal = bounded_rows(goals - positions, max_speed)`
2. `B_flow_perp_scaled = 3.303687238760696 * B_flow_perp_raw`, where the raw
   field is evaluated by the archived source implementation.
3. `B_rel = mean_{j != i} bounded_rows(p_i - p_j, max_speed)`

There is no per-state normalization and no additional clipping of the middle
field.  The global scale is the archived RMS calibration from 96 baseline
anchors (`P1_SCALE.json`, SHA256
`ecdca9e63e5d345ec45ae264bcdbc1006f9281cc5a32c3fdfec901e55609a133`).

The eta correction is the coefficient-weighted sum in field order.  Eta is
fixed for a continuation, while all three fields are recomputed from the
current state and current `u_safe` every physical step.

## Control chain

For either explicitly selected family:

`u_flow -> u_safe = Pi_safe(u_flow) -> g = sum eta_j B_j ->
u_exec = Pi_safe(u_safe + g) -> environment`.

Basis family is controller/oracle metadata, not environment configuration; it
does not alter environment or checkpoint fingerprints.  Eta zero is exactly
the zero correction and is verified against Safety after the unchanged second
projection.

