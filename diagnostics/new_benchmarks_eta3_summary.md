# New benchmark OrthoFlow3 summary

All new-scenario values use the present frozen 60-state formal safety evaluation and the fixed ≥15/16 definition. The Double-Bottleneck row is a historical OrthoFlow3 reference with a different safe-failure population and earlier screening/local protocols; its single-seed 60/61 shared-point observation must not be read as Q16 cross-state coverage.

| Benchmark | Mechanism | Hard-safety success | Safe failures | Robust-eta existence | Best robust eta state coverage | Median local robust fraction | 3D classification |
|---|---|---:|---:|---:|---:|---:|---|
| Double-Bottleneck | ordering/shared resource | historical frozen reference | 61 | 61/61 OrthoFlow3 reference | 60/61 under original episode seeds; not present Q16 protocol | 1.000 at Q≥0.50 in its 8-seed local study; not present 15/16 rule | historical `USE-ORTHOFLOW3`; not reclassified here |
| Four-Way | cyclic crossing | 32/60 | 28 | 28/28 | 100.00% | 1.0000 | `3D_ROBUST_SHARED` |
| Ring Exchange | circulation/lateral | 47/60 | 13 | 13/13 | 100.00% | 1.0000 | `3D_ROBUST_SHARED` |

The new rows are directly comparable to each other because they use the same eta domain, both common Sobol designs, seed policy, promotion rule, local design, coverage rule, and robust threshold. The historical Double-Bottleneck evidence is included only as requested context and is explicitly not numerically harmonized with this protocol.

No `G_phi` was trained, no eta dimension/domain was expanded, and OrthoFlow3 and hard safety were not redesigned.

## Completion audit

- Shared/single-integrator regression: 58/58 passed.
- Toy Give-Way regression: 2/2 passed.
- Double-Bottleneck regression: 19/19 passed.
- New shared/Four-Way/Ring/eta3 regression: 49 passed plus 3 subtests.
- Shared rollout database integrity: PASS for both scenarios, with zero foreign-key violations and zero duplicate exact keys.
- Final canonical hashes equal the recorded pre-work hashes: Toy Give-Way `6c88e06bd4f167ad5ce6b75ff670247cb916310913c7e37d1459e93edb0f9bf9`; Double-Bottleneck `1e0e98a3ddb91dc123ba2bdeb75476fa41583382e34da1febd159fc307de39ab`; single-integrator `f26814a9616d9bb7d17a731656e93721031570fb5e271315d3607ea62f728437`; shared-control `3accf5da696f083a684694beb668797f28f660f3d2d942790c41cacd3cd983c9`.
- The final regression ran on CPU; the emitted CUDA initialization warning was the known CPU fallback and not a test failure.
