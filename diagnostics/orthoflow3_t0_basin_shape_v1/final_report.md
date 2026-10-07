# OrthoFlow3 t0 basin shape and cross-state morphology

## Scope

Cached exact Q64 labels were aggregated before the frozen targeted probe batch. No model was trained. Every genuinely new probe used 64 matched future seeds.

| State | interpolation B63 | interpolation non-B63 | normal lower bound | tangent/normal anisotropy | morphology |
|---|---:|---:|---:|---:|---|
| T0_WIDE_perm00_ep0082 | 0.67 | 0.33 | 0.025 | 3.82 | HOLED_OR_INTERLEAVED_REGION |
| T0_WIDE_perm01_ep0217 | 1.00 | 0.00 | 0.000 | 2.38 | THIN_FILLED_SLAB |
| T0_WIDE_perm02_ep0182 | 1.00 | 0.00 | 0.075 | 2.21 | THIN_FILLED_SLAB |
| T0_WIDE_perm03_ep0205 | 1.00 | 0.00 | 0.075 | 2.82 | THIN_FILLED_SLAB |
| T0_WIDE_perm04_ep0195 | 1.00 | 0.00 | 0.025 | 6.91 | THIN_FILLED_SLAB |
| T0_WIDE_perm05_ep0179 | 1.00 | 0.00 | 0.000 | 1.87 | THIN_FILLED_SLAB |
| T0_WIDE_perm06_ep0074 | 0.75 | 0.25 | 0.050 | 2.59 | UNDERRESOLVED |
| T0_WIDE_perm07_ep0139 | 0.92 | 0.08 | 0.050 | 1.75 | THIN_FILLED_SLAB |

## Cross-state conclusion

**SHARED_MORPHOLOGY_STATE_DEPENDENT_DEFORMATION**. Morphology counts: `{'HOLED_OR_INTERLEAVED_REGION': 1, 'THIN_FILLED_SLAB': 6, 'UNDERRESOLVED': 1}`. The recommendation is **state-conditioned anisotropic slab/tube, contingent on future false-inclusion validation**. Alignment removes raw center/orientation differences but does not erase observed variation in tangential filling, normal thickness, or holes/boundary behavior.

## Theory

Observed basins are empirical samples of `B(h)`, not proven sets. Exact equality across h is not expected; a shared representation family with state-conditioned deformation is more plausible under smoothness/regularity assumptions, while topology can change near coordination or projection-active-set events. No theorem is claimed.

## Limitation

Observed labels establish only the sampled points. Unobserved space is unknown; this experiment cannot establish connectedness, interpolation safety outside tested trajectories, absence of holes, or a certified false-inclusion rate.

## Next step

Frozen targeted validation of interpolation, normal offsets, and candidate gaps around the empirically favored representation; no learning until false-inclusion rate is measured.
