# OrthoFlow3 true-t0 multiball basin-learning audit

## Outcome

**T0_MULTIBALL_GEOMETRY_INADEQUATE.** The fixed-eight geometry gate failed, so the mandated hard stop was applied before dataset construction or neural training.

| State | K | largest r | union diameter | false inclusion | independent robust coverage | usable |
|---|---:|---:|---:|---:|---:|---:|
| T0_WIDE_perm00_ep0082 | 4 | 0.04416 | 1.3577 | 0 | 0/6 (0.0%) | no |
| T0_WIDE_perm01_ep0217 | 4 | 0.24969 | 1.2134 | 0 | 2/24 (8.3%) | yes |
| T0_WIDE_perm02_ep0182 | 4 | 0.20187 | 1.1656 | 0 | 2/20 (10.0%) | yes |
| T0_WIDE_perm03_ep0205 | 4 | 0.08500 | 1.5762 | 0 | 0/17 (0.0%) | no |
| T0_WIDE_perm04_ep0195 | 4 | 0.04416 | 1.2471 | 0 | 0/4 (0.0%) | no |
| T0_WIDE_perm05_ep0179 | 4 | 0.17000 | 1.1831 | 0 | 0/16 (0.0%) | yes |
| T0_WIDE_perm06_ep0074 | 4 | 0.08500 | 1.3291 | 0 | 0/12 (0.0%) | no |
| T0_WIDE_perm07_ep0139 | 4 | 0.17000 | 1.3418 | 0 | 0/16 (0.0%) | yes |

All eight states reached four verified components. Independent union validation produced **0 confirmed false inclusions**. The sets were therefore reliable where claimed, but not sufficiently representative of independently discovered robust success geometry.

## Frozen gate

- Training-usable states: **4/8** (required at least 6/8).
- Median largest-component radius: **0.12750** (required at least 0.15).
- Median union diameter: **1.2881** (required at least 0.30).
- Median independent robust coverage: **0.000** (required at least 0.50).
- Retained-set maximum common coverage: **5/8 = 0.625** (must not exceed 0.75).
- Confirmed union false inclusions: **0** (required zero).

Failed criteria: `training_usable_at_least_6_of_8, median_largest_component_radius_at_least_0.15, median_independent_robust_coverage_at_least_0.50`.

The failure is not a reliability failure: every accepted component remained conservative under the frozen validation protocol. It is a size/coverage failure. With only four local spheres, the union captured a median of 0% of independently discovered B63 cloud points, despite large center-to-center union diameters.

## Learning decision

The verified multi-ball representation did **not** provide a sufficiently large learning target under the preregistered criteria. Consequently no source-diverse dataset was built, neither `G_CENTER` nor `G_SET` was trained, and no held-out or fresh-WIDE controller comparison was run. Set supervision therefore cannot be compared with center regression in this experiment.

## Runtime

- New continuations: **58,424**
- Physical steps: **25,512,209**
- Longest per-state rollout wall time: **1.64 h**
- Maximum GPU shards: **6**

## Smallest justified next step

Audit why sparse independently robust eta points lie outside the verified four-ball union, then test a conservative representation capable of retaining disconnected/nonconvex regions without asserting success in unverified gaps. Do not begin learning until that representation passes an independent coverage gate.
