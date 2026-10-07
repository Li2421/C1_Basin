# OrthoFlow3 true-t0 PACT encoding and training readiness v1

## Outcome

**PACT_NOT_READY_AND_T0_DATA_INSUFFICIENT.** No rollout or network training was launched after the deterministic gates failed.

## PACT cached geometry

The audit used 877 exact-Q64 eta records from eight true-t0 states: 762 B63 and 115 non-B63. Independent `coverage64`/Sobol B63 points were excluded from triangulation and used as held-out cached recall.

| State | selected L/d5 | cells | independent B63 recall | cached false inclusions | median delta-/delta+ |
|---|---:|---:|---:|---:|---:|
| T0_WIDE_perm00_ep0082 | 1.5 | 72 | 0.0000 | 0 | 0.100/0.100 |
| T0_WIDE_perm01_ep0217 | 1.5 | 63 | 0.0000 | 0 | 0.100/0.100 |
| T0_WIDE_perm02_ep0182 | 1.5 | 65 | 0.0000 | 0 | 0.100/0.100 |
| T0_WIDE_perm03_ep0205 | 1.5 | 70 | 0.0000 | 0 | 0.100/0.100 |
| T0_WIDE_perm04_ep0195 | 1.5 | 69 | 0.0000 | 0 | 0.100/0.100 |
| T0_WIDE_perm05_ep0179 | 2.5 | 106 | 0.0625 | 0 | 0.100/0.100 |
| T0_WIDE_perm06_ep0074 | 1.5 | 78 | 0.0000 | 0 | 0.100/0.100 |
| T0_WIDE_perm07_ep0139 | 2.5 | 95 | 0.0625 | 0 | 0.100/0.100 |

All selected PACT candidates have zero cached false inclusions, but six states have zero independent recall and two have recall 1/16=0.0625. Median recall is 0. This fails the per-state 0.30 gate for all eight states. Consequently there were zero SHAPE-USABLE states, and the preregistered fresh validation stage was not run. The 0.10 cached thickness values are therefore not evidence of transferable normal thickness; they only reflect absence of cached negatives inside the accepted local cells.

## State-level readiness

The repository contains 312 deduplicated true-t0 state records. Of these, 301 have an exact frozen 214-D h0 vector, but only eight states have compatible exact-Q64 clouds and defensible independently verified robust-center targets. The remaining records are single-episode fresh-WIDE states or historical state snapshots/lower-seed evidence—not B63 supervision.

- SET-USABLE: 0; achievable split 0/0/0 versus required 24/8/8.
- POINT-USABLE: 8; frozen candidate split 5/1/2, missing 19/7/6.

Thus even if PACT encoding had failed completely—which it did under the recall criterion—the existing true-t0 point supervision alone would **not** be sufficient to train. Hundreds of eta evaluations at eight h0 values remain eight state-level examples.

## Interpretation

This is a representation failure and a state-count shortfall, not evidence that basin supervision itself is invalid. PACT interpolates the structured construction cloud without cached false inclusions, yet its surfaces fail to cover the independent robust observations. Training was prohibited by both dataset gates.

## Next step

Acquire 32 additional source-diverse true-t0 robust-interior point labels in the frozen 19/7/6 split deficit, then train the clean G_POINT_T0 baseline before revisiting set learning.
