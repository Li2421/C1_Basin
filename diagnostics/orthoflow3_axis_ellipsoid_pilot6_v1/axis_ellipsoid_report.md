# OrthoFlow3 axis-aligned ellipsoid pilot6 v1

## Decision

**REJECT — AXIS_ELLIPSOID_NOT_RELIABLE**

Centers, OrthoFlow3, E_bridge, normalization, safety projection, horizon, RNG, and B63 semantics were frozen. No model was trained. The globally frozen shrink factor was 0.85 and the shell scale was 1.05.

| State | Frozen center | Ball r | Ball volume | Ellipsoid (r1,r2,r3) | Ellipsoid volume | E/B ratio | E_bridge fraction | Interior 64-seed | Shell 8/8 | Axis anisotropy |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| N_r104_s125 | (0.898438, -0.156250, 0.445312) | 0.015938 | 1.6957e-05 | (0.116875, 0.015938, 0.328047) | 0.00255957 | 150.944x | 0.209% | 8/8 B63 | 24/24 | 20.583 |
| ZR_P_r052_m080_s95400004_p253 | (0.625000, 0.000000, 0.375000) | 0.353621 | 0.185227 | (0.403750, 0.403750, 0.403750) | 0.275693 | 1.488x | 22.557% | 8/8 B63 | 24/24 | 1.000 |
| S_r043_p02 | (0.625000, 0.000000, 0.375000) | 0.233750 | 0.0534987 | (0.297500, 0.233750, 0.324062) | 0.0943965 | 1.764x | 7.723% | 7/8 B63 | 22/24 | 1.386 |
| N_r076_s119 | (0.625000, 0.000000, 0.375000) | 0.249688 | 0.0652047 | (0.310781, 0.265625, 0.337344) | 0.11665 | 1.789x | 9.544% | 7/8 B63 | 19/24 | 1.270 |
| R_D2_s95101009_p112 | (0.625000, 0.000000, 0.375000) | 0.265625 | 0.0785047 | (0.403750, 0.265625, 0.403750) | 0.181377 | 2.310x | 14.840% | 8/8 B63 | 24/24 | 1.520 |
| N_r004_m120 | (0.625000, 0.000000, 0.375000) | 0.353621 | 0.185227 | (0.403750, 0.403750, 0.403750) | 0.275693 | 1.488x | 22.557% | 8/8 B63 | 24/24 | 1.000 |

## Validation

- Independent interior screening: 191/192 were 8/8.
- Predeclared promotion: 46/48 were B63; confirmed false inclusions: 2.
- Shell: 137/144 points were 8/8 at ellipsoid radius 1.05.
- Median volume increased from 0.07185469 to 0.14901363; 6/6 states exceeded the predeclared 1.25x meaningful-gain threshold.
- Collisions: 0; invalid/numerical/projection evaluations: 0.

### Confirmed false inclusions

- `S_r043_p02` / `I31`: eta=(0.551305958, -0.015356688, 0.595501048), ellipsoid radius 0.967719, true success 61/64, outcomes {"safe_deadlock": 3, "success": 61}.
- `N_r076_s119` / `I31`: eta=(0.548016046, -0.017450781, 0.604537976), ellipsoid radius 0.967719, true success 25/64, outcomes {"safe_deadlock": 39, "success": 25}.

## Interpretation

The proposed axis-aligned ellipsoid would have supplied `(c1,c2,c3,r1,r2,r3)`, but the confirmed false inclusions mean it is not a validated conservative learning label. A later direct-center baseline could use `L_center=||G_phi(x)-c(x)||^2`; a set objective would use `d_E^2=sum_j((G_j-c_j)/r_j)^2`, with zero set loss inside `d_E<=1` and positive penalty outside. Neither objective was implemented here.

A later deformation study may compare robust-center-like outputs against minimum intervention inside the verified ellipsoid, using either `||eta||^2` or rollout `J_def`. No choice was made here.

## Regression and integrity

The frozen repository regression suite completed with 139 tests and 13 subtests passing, zero failures. Frozen canonical hashes are checked separately in `frozen_hash_regression.json`.

## Stop condition

No G_phi, Q, J, H, gate, rotated ellipsoid, polytope, or nonconvex set model was trained or constructed.
