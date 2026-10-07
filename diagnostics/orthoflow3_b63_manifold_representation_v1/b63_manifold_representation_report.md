# Existing B63 low-complexity manifold representation audit

## Scope

This audit ran **zero** new rollouts, submitted **zero** GPU jobs, queried **zero** new eta values, and trained **zero** networks. It used only the exact B63 point clouds and their archived conditioning/provenance manifest.

## Held-out protocol

Independent Sobol B63 points are the primary held-out geometry set for 6/8 states. `ep0082` (6 points) and `ep0195` (4 points) use an explicitly secondary deterministic construction-point split because they do not meet the preregistered minimum of 8 independent B63 points.

| State | train | independent B63 | regime | R0 median | R1 median | R2 selected K / median | R3 median | R0 tangential coverage |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| T0_WIDE_perm00_ep0082 | 48 | 6 | SECONDARY_DETERMINISTIC_CONSTRUCTION_SPLIT | 0.0401 | 0.0222 | 2 / 0.0241 | 0.0220 | 0.62 |
| T0_WIDE_perm01_ep0217 | 64 | 24 | PRIMARY_INDEPENDENT_SOBOL | 0.1651 | 0.1947 | 2 / 0.1913 | 0.2453 | 0.88 |
| T0_WIDE_perm02_ep0182 | 62 | 20 | PRIMARY_INDEPENDENT_SOBOL | 0.1092 | 0.1150 | 3 / 0.1922 | 0.2067 | 0.65 |
| T0_WIDE_perm03_ep0205 | 61 | 17 | PRIMARY_INDEPENDENT_SOBOL | 0.0757 | 0.0673 | 3 / 0.1133 | 0.1641 | 0.71 |
| T0_WIDE_perm04_ep0195 | 49 | 4 | SECONDARY_DETERMINISTIC_CONSTRUCTION_SPLIT | 0.0387 | 0.0187 | 2 / 0.0173 | 0.0308 | 0.85 |
| T0_WIDE_perm05_ep0179 | 62 | 16 | PRIMARY_INDEPENDENT_SOBOL | 0.1259 | 0.1348 | 3 / 0.1788 | 0.1235 | 0.44 |
| T0_WIDE_perm06_ep0074 | 62 | 12 | PRIMARY_INDEPENDENT_SOBOL | 0.1068 | 0.0649 | 2 / 0.1688 | 0.2389 | 0.75 |
| T0_WIDE_perm07_ep0139 | 64 | 16 | PRIMARY_INDEPENDENT_SOBOL | 0.0947 | 0.1045 | 2 / 0.1382 | 0.1282 | 0.56 |

## Representation comparison

Primary independent-held-out median residuals are R0 affine **0.1080**, R1 quadratic **0.1097**, R2 selected patches **0.1738**, and R3 local kNN ceiling **0.1854**. Median relative gains are R1/R0 **-6.2%**, R2/best(R0,R1) **-57.2%**, and R3/R2 **-17.9%**.

R0 has the lowest complexity: a center and oriented 2-D plane. R1 adds six normal-graph coefficients. R2 uses (2, 3, 4) local planes and train-only BIC selection. R3 is nonparametric and is only a local-reconstruction ceiling. Empirical `delta90/delta95/delta_max` values are support thicknesses around B63 samples; they are **not certified success tubes**.

## Decision

**B63_GEOMETRY_TOO_SPARSE_OR_INCONSISTENT**

The decision uses only primary independent-held-out states where possible. A common class with state-dependent parameters can be used across states, but no fitted surface certifies interpolation or thickness safety.

## Required next validation

The next experiment should use a small, frozen rollout set targeted to: (1) interpolation within fitted tangential supports, (2) normal offsets near empirical `delta95`, and (3) candidate gaps between nearby successful samples. It must measure both B63 success and false inclusion before any learning target is adopted.

## Limitation

This task cannot establish success-set connectedness, interpolation safety, absence of holes, or a tube false-inclusion rate, because it used success points only and ran no new rollout.
