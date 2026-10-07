# Existing true-t0 B63 geometry audit

## Scope and integrity

This is a strictly offline audit. It launched **0 new rollouts**, submitted **0 GPU jobs**, queried **0 new eta values**, and trained **0 networks**. Exact evidence was accepted only when state ID, `h/xi0` conditioning identifier, source group/RNG namespace, authoritative OrthoFlow3 stack, and at least 64 unique future seeds matched. The invalid feature-replay directory and all 8/8-only points were excluded. Exact eta coordinates were deduplicated in float64 representation.

## Per-state summary

| State | unique B63 | balanced PCA EV1/2/3 | shape | MST max edge | all-cache ball coverage | independent-cloud coverage | radial morphology |
|---|---:|---:|---|---:|---:|---:|---|
| T0_WIDE_perm00_ep0082 | 67 | 0.736/0.248/0.016 | 2-D sheet-like | 0.405 | 41.8% | 0.0% | multiple angular sectors |
| T0_WIDE_perm01_ep0217 | 88 | 0.580/0.306/0.113 | 3-D volumetric | 0.354 | 34.1% | 8.3% | multiple angular sectors |
| T0_WIDE_perm02_ep0182 | 82 | 0.634/0.302/0.064 | 2-D sheet-like | 0.311 | 36.6% | 10.0% | multiple angular sectors |
| T0_WIDE_perm03_ep0205 | 78 | 0.643/0.314/0.042 | 2-D sheet-like | 0.316 | 35.9% | 0.0% | multiple angular sectors |
| T0_WIDE_perm04_ep0195 | 66 | 0.668/0.318/0.014 | 2-D sheet-like | 0.860 | 42.4% | 0.0% | multiple angular sectors |
| T0_WIDE_perm05_ep0179 | 78 | 0.528/0.394/0.078 | 2-D sheet-like | 0.503 | 35.9% | 0.0% | multiple angular sectors |
| T0_WIDE_perm06_ep0074 | 74 | 0.594/0.356/0.050 | 2-D sheet-like | 0.405 | 37.8% | 0.0% | multiple angular sectors |
| T0_WIDE_perm07_ep0139 | 80 | 0.559/0.369/0.072 | 2-D sheet-like | 0.405 | 35.0% | 0.0% | multiple angular sectors |

B63 count range is **66–88**, median **78.0**. Density-balanced PCA uses one representative per normalized 0.05 voxel to reduce the bias from dense ray/boundary sampling. Shape counts are `{"approximately 2-D sheet-like": 7, "approximately 3-D volumetric": 1}`. The median pairwise absolute alignment of the dominant PCA axis is **0.768**; the median sheet-normal alignment is **0.580**.

The disjoint uniform coverage cloud provides a second, less density-biased check: B63 counts are 4–24/state. Among the six states with at least 10 such points, four are sheet-like and two are volumetric under the same rule; the two remaining clouds are explicitly too sparse for that standalone classification.

## Graph and cluster geometry

The kNN, radius-graph, MST, and DBSCAN tables show scale-dependent empirical B63 point-cloud connectivity. DBSCAN cluster counts over eps 0.075/0.10/0.15/0.20 are:

- `T0_WIDE_perm00_ep0082`: [4, 4, 4, 4]
- `T0_WIDE_perm01_ep0217`: [3, 4, 7, 5]
- `T0_WIDE_perm02_ep0182`: [3, 4, 5, 5]
- `T0_WIDE_perm03_ep0205`: [4, 4, 4, 4]
- `T0_WIDE_perm04_ep0195`: [4, 4, 4, 4]
- `T0_WIDE_perm05_ep0179`: [3, 6, 5, 5]
- `T0_WIDE_perm06_ep0074`: [5, 4, 4, 6]
- `T0_WIDE_perm07_ep0139`: [4, 5, 4, 4]

Apparent components persist across several fixed-radius scales, but they align strongly with the deliberately sampled original/new verified balls: roughly 57 points/state come from local ball construction/validation. After accounting for this bias, k=5 kNN is connected in 6/8 states and six MSTs have no dominant adjacent-edge gap ratio above 1.5. Thus the cloud does not support a stable universal count of true components. MST maximum merge scales and full distributions are archived. H0 Vietoris–Rips merge scales were computed exactly from each Euclidean MST. H1 was skipped because `ripser/gudhi` are unavailable; no dependency was installed.

## Why verified balls cover little

Across every existing B63 record, verified-ball coverage has median **36.2%**; this number is sampling-biased upward because ball-construction points are included. The disjoint independent-cloud coverage remains the authoritative diagnostic and has median **0.0%**. The component centers span large union diameters, but most component radii are limited by a thin local direction. Confirmed B63 samples extend far along the two dominant tangential PCA directions and occupy multiple angular sectors. Isotropic balls therefore spend radius in the narrow normal direction and miss long sheet-like extensions. Importantly, the audit does not assert that unsampled intervals between them succeed.

## Cross-state interpretation

Radial morphology counts are `{"multiple angular sectors": 8}`. The recurring pattern is a two-dimensional, state-dependent sheet with multiple angular sectors, scale-sensitive empirical clustering, and low conservative-ball coverage. The first PCA axis is fairly consistent and is usually eta1-dominated, but the second tangent and sheet normal vary enough that one fixed plane is not adequate. No stable small component count is consistent across all eight states.

## Representation decision

**LOW_DIMENSIONAL_MANIFOLD_OR_TUBE_PROMISING**

The best-supported next representation is a conservative low-dimensional manifold/tube candidate, most naturally a state-conditioned curved 2-D sheet with an explicitly verified local thickness. This is a hypothesis about representation class, not proof of topology. It must not fill between sampled successes without future failure/interpolation checks. A directional-radius model would require ray evidence, while a larger multi-component union would need to overcome the observed component-count and coverage scaling problem.

## Critical limitation

No new failure/interpolation/midpoint rollout was run. Therefore this audit cannot establish true connectedness, true convexity, star-convexity, or absence of hidden failure gaps. It identifies only the most plausible next representation from finite, nonuniform, confirmed-success samples.
