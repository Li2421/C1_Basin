# OrthoFlow3 representation migration report

## Decision

**ORTHOFLOW3_FULL_REBUILD_READY_BUT_REQUIRES_APPROVAL**

Stage A passed, the explicit versioned basis selector is integrated without
changing P0, and the frozen 32-state migration audit completed within budget.
The full 424-state rebuild was not launched because its exact 256-point screen
alone requires `108,544` continuations, above the automatic
15,000 limit.

## Authoritative implementation

- Source: `diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py`
- SHA256: `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`
- Symbols: `raw_ortho_flow`, `basis_terms`, `correction`
- Frozen scale: `3.303687238760696` from `P1_SCALE.json` SHA256
  `ecdca9e63e5d345ec45ae264bcdbc1006f9281cc5a32c3fdfec901e55609a133`
- Candidate design: 256 frozen Sobol points, SHA256
  `6a0732c5ad35bc2000e61be6c2825034058b30aad09458cf2cf5975cc501f258`

The exact basis equations and code references are recorded in
`basis_interface_spec.md`; no formula was reconstructed from prompt prose.

## Integrity

- Archived P0 and OrthoFlow3 fields reproduced with maximum absolute error 0
  on 12 archived anchors.
- Three archived selected-center continuations reproduced exact basis fields
  and successful outcomes.
- Eta zero produced exactly zero correction and exactly the Safety action on
  64 state/Flow/family comparisons.
- Historical P0 source files were not modified; new jobs require explicit
  `basis_family = p0 | orthoflow3`.

## Frozen 32-state audit

| Metric | P0 | OrthoFlow3 |
|---|---:|---:|
| robust basin found | 32/32 | 32/32 |
| median observed Q_max | 1.0000 | 1.0000 |
| median best J_def | 0 | 0 |
| median best J_def, active-required 14 | 0.221035 | 0.349371 |

Paired basin existence: both `32`, P0-only
`0`, OrthoFlow3-only `0`,
neither `0`.  OrthoFlow3 states classify as
ZERO_SUFFICIENT `18`, ACTIVE_REQUIRED
`14`, NO_BASIN_FOUND
`0`.

The median per-agent maximum absolute goal/middle cosine fell from
`0.998083` under P0 to `4.93108e-11` under OrthoFlow3,
confirming that the old redundancy is materially reduced.

The audit confirmed at least one robust candidate in `32/32` states.
It confirmed at least two robust points in
`31/32`
states; candidate counts and the reported canonical target are lower-bound /
provisional results within the budgeted promoted set.  Of
`48` eight-stream interpolation screens,
`0` had enough observed failures to reject B63 and
`48` were 8/8 but remain explicitly unconfirmed at B63.

Across the 14,624 Stage-B/C continuation records there were zero collisions,
zero execution errors, and zero unrecovered projection failures.  Sixteen
first-projection and six second-projection retries were recovered by the
unchanged frozen solver path.

## Migration scope and next step

All future eta/basin oracle and dataset-learning work should default to
**OrthoFlow3** with an explicit basis version/hash.  P0 remains available only
as a reproducible historical baseline or declared ablation.

The smallest justified next experiment is the already-prepared full-424
OrthoFlow3 oracle/data rebuild after explicit approval of the reported compute
cost.  No neural model should be trained before that rebuilt candidate evidence
is inspected.
