# New deadlock benchmark summary

This comparison is documentary: no controller, OrthoFlow3, eta, basin, or
`G_phi` experiment was run while preparing it.  Four-Way and Ring values are
their already-frozen untouched-test reports. Four-Way's formal decision is
[`PASS_STAGE1`](four_way_intersection_stage1/REPORT.md).

| Benchmark | Primary mechanism | Expert | Frozen raw Stage-I success | Collision | Timeout | Main remaining failure |
|---|---|---:|---:|---:|---:|---|
| Double-Bottleneck | ordering over repeated shared narrow resources | centralized 8-order trajectory expert | 28.13% (27/96) | 52.08% (7 wall + 43 agent, fresh untouched set) | 19.79% (19/96) | agent collisions and terminal/goal timeouts; baseline was **not** accepted as mature in its authority report |
| Four-Way Intersection | cyclic crossing conflict in an open central region | centralized crossing-order hypothesis search | 48.33% (29/60) | 10.00% (1 wall + 5 agent) | 41.67% (25/60) | timeout/liveness failure under cyclic crossing interactions |
| Ring Exchange | lateral circulation consistency in a wide annulus | centralized CW/CCW spatiotemporal search | 78.33% (47/60) | 1.67% (1 agent; 0 obstacle/outer wall) | 20.00% (12/60) | late convergence/drift timeout |

## Sources and interpretation

**Double-Bottleneck.** The authority is
[`double_bottleneck_sxl_baseline_maturation/REPORT.md`](double_bottleneck_sxl_baseline_maturation/REPORT.md),
especially its frozen fresh untouched-set outcome and maturity decision, with
machine-readable values in
[`summary.json`](double_bottleneck_sxl_baseline_maturation/summary.json).
This is a raw conventional joint MACFlow result, not an eta/basin/safety
result.  Its report explicitly concludes that the S-XL candidate is not yet a
credible mature task-completion baseline; the row is included for mechanism
comparison, not to overwrite that decision.

**Four-Way Intersection.** Values are from the master-authorized frozen run
[`final_diagnostics_v13_frozen_test/summary.json`](four_way_intersection_stage1/final_diagnostics_v13_frozen_test/summary.json).
The open central conflict disk admits multiple physically valid permutations;
the environment has no right-of-way or fixed crossing priority. Its residual
is principally timeout rather than a hidden wall bottleneck. Its formal
[`REPORT.md`](four_way_intersection_stage1/REPORT.md) is `PASS_STAGE1` and
contains the dataset, expert, OOD, mode, and safety-smoke provenance.

**Ring Exchange.** Values and full provenance are in
[`ring_exchange_stage1/REPORT.md`](ring_exchange_stage1/REPORT.md) and its
frozen artifact
[`final_diagnostics_v10_frozen_test/summary.json`](ring_exchange_stage1/final_diagnostics_v10_frozen_test/summary.json).
The 2.35-wide radial annulus is over six agent radii wide; compatible CW or
CCW movement allows simultaneous exchange.  Policy modes were CW 32/60 and
CCW 28/60, while no direction label is supplied to MACFlow.

## Mechanism-distinctness assessment

The three benchmarks are genuinely different coordination/liveness probes.
Double-Bottleneck tests sequencing through serial, repeated narrow resources.
Four-Way tests mutually incompatible crossing attempts and cyclic yielding in
an open shared conflict region, with many crossing orders.  Ring tests whether
lateral decisions remain circulation-consistent around a broad obstacle; it is
not one-at-a-time passage allocation.  Thus geometry is not merely cosmetic:
the relevant coordination variable changes from resource order, to crossing
cycle resolution, to distributed circulation convention.

This table does **not** claim that all three have the same Stage-I maturity.
Ring v10 and Four-Way v13 are `PASS_STAGE1`; the cited Double-Bottleneck S-XL authority remains
`REVISE_STAGE1`/not mature.  No frozen test was used to collect data or change
any of these baselines.
