# Four-Way / Ring symmetry and conditioning audit

## Executive conclusion

No exact physical implementation violation was found. Dynamics, swept collision distances, hard-safety constraints, the first projection, all three OrthoFlow3 fields, the full correction, and the mandatory second projection passed every pre-registered rigid-transform and consistent-relabel test.

The learned pipeline is nevertheless not convention-neutral. Four-Way has strong world-frame and agent-slot dependence. Ring Stage-I is exactly rotation-equivariant under 90/180/270-degree rotations because of its radial/CCW-tangent chart, but its downstream basin conditioning appends a world-coordinate Flow action to a local-coordinate observation. That makes generator/critic output weakly rotation-dependent again. Ring also shows strong reflection/chirality bias and moderate downstream slot dependence.

Final classifications:

- Four-Way Intersection: `REPRESENTATION_SHORTCUT_RISK`
- Ring Exchange: `REPRESENTATION_SHORTCUT_RISK`
- Joint generator/critic: `REPRESENTATION_SHORTCUT_RISK`

This is not `IMPLEMENTATION_BUG_FOUND`: the exact physical controller is correct. The risks are avoidable tensor conventions exposed to non-equivariant learned networks plus finite-data learned asymmetry.

## 1. Executable conditioning map

The exact machine-readable map is in `conditioning_pipeline_map.json`.

### Four-Way

`positions/velocities/goals in world frame -> observation[4,18] -> flatten[72] -> append reference Flow world action[8] -> h[80] -> per-scenario z-score -> Four adapter -> shared generator/critic trunk`.

Each row is own `(p_x,p_y,v_x,v_y,goal_x-p_x,goal_y-p_y)`, followed by relative position/velocity blocks for the other slots in ascending A/B/C/D order. Stage-I emits an eight-dimensional world action.

### Ring

`world physical state -> per-agent radial/CCW-tangent observation[4,23] -> flatten[92] -> append reference Flow WORLD action[8] -> h[100] -> per-scenario z-score -> Ring adapter -> shared generator/critic trunk`.

The first 92 values are rotation-invariant local quantities. The last eight are world x/y velocities after converting Stage-I local actions back to world coordinates. This mixed chart is the main downstream symmetry risk.

For both scenarios, `c[33]` is 30 mechanically serialized descriptor scalars plus a three-way scenario one-hot. All descriptors are constant within each scenario, so per-scenario normalization maps canonical c to numerical zero (max residual below 6e-7). Scenario identity still selects a separate input adapter. There is no history; timestep and normalized episode time are stored but not fed into h.

## 2. Arbitrary conventions

The complete PHYSICAL/REPRESENTATIONAL table is in `arbitrary_conventions.json`. Important points:

- World x/y orientation, A/B/C/D labels, flatten order, relative-block order, Ring cardinal index and CCW-positive tangent are representational.
- Goal assignment and geometry dimensions are physical; consistently moving an agent with its goal is only relabeling.
- Four-Way's q-offset layout has a canonical 180-degree symmetry. A 90-degree test is physically equivalent only when the full goal/layout chart is rotated; it is outside the frozen canonical lane-offset distribution.
- Ring reflection is physically valid, but it reverses the representational tangent sign.

## 3. Exact-component metamorphic tests

| Scenario | Cases | Exact failures | Max first projection error | Max second projection error | Max basis/correction error |
|---|---:|---:|---:|---:|---:|
| Four-Way | 30 | 0 | 8.38e-15 | 2.72e-08 | 1.62e-14 |
| Ring | 36 | 0 | 0 | 1.44e-08 | 1.11e-16 |

The same eta produces the correspondingly transformed correction. Collision outcome and sorted wall/pair distances were invariant. No tolerance was relaxed after execution.

## 4. MACFlow symmetry

Paired action RMSE is normalized by max speed:

| Transform | Four-Way | Ring |
|---|---:|---:|
| rotate 90 | 0.650 | 0 |
| rotate 180 | 0.376 | 0 |
| rotate 270 | 0.579 | 0 |
| cyclic relabel | 0.563 | 0.018 |
| reflection | N/A | 0.480 |

Ring rotation errors are exactly zero at sampled precision. Four-Way 90-degree full-scene rotation is severe; even its canonical 180-degree relabel has normalized paired RMSE 0.199. Ring cyclic relabel is mild, while reflection is severe (0.480), showing learned chirality sensitivity.

In full-horizon hard-safety-only replays, Ring matched outcomes in 100% of four pairs; its two pure rotations have effectively zero back-transformed trajectory error. Four-Way matched 50%: both 90-degree pairs changed success to timeout, while both canonical 180-degree pairs preserved the terminal outcome.

## 5. Generator proposal distribution

No Q16/basin experiment was repeated. “Robust hit” in the diagnostic files means critic-predicted score >=0.9375 only, not a new robust certificate.

| Transform | Four mean eta delta | Four cloud match | Ring mean eta delta | Ring cloud match |
|---|---:|---:|---:|---:|
| rotate 90 | 1.210 | 1.210 | 0.067 | 0.067 |
| rotate 180 | 0.590 | 0.575 | 0.089 | 0.091 |
| cyclic relabel | 1.337 | 1.353 | 0.244 | 0.236 |

Ring rotation is good but not exact (mean delta 0.067–0.093 in normalized eta coordinates), precisely unlike its exact Stage-I rotation behavior. Ring reflection delta is 0.926. Four-Way pure rotations and cyclic relabeling change the proposal distribution substantially; its canonical 180-degree relabel is much closer (0.165).

## 6. Critic symmetry

| Transform | Four score abs diff | Four top-1 match | Ring score abs diff | Ring top-1 match |
|---|---:|---:|---:|---:|
| rotate 90 | 0.719 | 0% | 0.0069 | 75% |
| rotate 180 | 0.084 | 0% | 0.0062 | 75% |
| cyclic relabel | 0.923 | 0% | 0.0116 | 50% |

Four score shifts are severe for 90/270 rotation and pure relabeling. The canonical 180-degree relabel is much cleaner: mean absolute score change 0.0005, top-1 match 75%. Ring rotations preserve ranking well (Spearman 0.956–0.969), while cyclic relabeling lowers it to 0.624; reflection produces mean score change 0.126.

## 7. Agent ordering and normalization

Exact components pass consistent permutation tests, so no physical module treats a slot as priority. Learned networks are ordinary flattened MLPs and are not architecturally permutation-equivariant. All Stage-I and basin normalization statistics are feature/slot specific.

- Four-Way: pure cyclic relabel MACFlow RMSE 0.563; generator normalized mean delta 1.337; critic top-1 agreement 0%.
- Ring: corresponding values 0.018, 0.244, and 50%.

This is slot dependence, not a fixed environment right-of-way rule.

## 8. Scenario-ID shortcut audit

Canonical c is effectively zero in both scenarios after per-scenario normalization. Therefore the physical descriptor and one-hot do not vary over canonical states; scenario separation primarily occurs through separate Four/Ring input adapters.

Changing only the nonphysical one-hot while keeping the branch and all physical inputs fixed triggered the pre-registered critic threshold on 1/8 Four-Way and 1/8 Ring states; generator mean never crossed its threshold. `SCENARIO_ID_SHORTCUT` is therefore flagged for both, with an important qualification: this counterfactual is outside training support and does not prove canonical predictions actively read the zeroed one-hot. The explicit branch adapter remains real bookkeeping-based routing.

## 9. Closed-loop metamorphic replay

The learned finite proposal pipeline used canonical mean+4 proposals; no continuous eta search was performed.

- Four-Way: canonical 180-degree relabel preserved success in 2/2 pairs with back-transformed trajectory RMSE 0.016–0.024. Full-scene 90-degree rotations changed success to timeout in 2/2 pairs and had trajectory RMSE above 1.0.
- Ring: all 4 pairs remained successful and collision-free. Pure 90-degree rotations selected the same proposal and stayed close (trajectory RMSE 0.0035 and 0.0203). Rotation+cyclic relabel remained successful but showed larger path/eta variation.

These are deterministic train/dev diagnostics only, not generalization estimates.

## 10. Discrepancy classification

### EXACT_IMPLEMENTATION_VIOLATION

None.

### REPRESENTATION_CONVENTION_DEPENDENCE

- Four-Way world Cartesian flattening and semantic slot order.
- Ring's mixed local/world h serialization.
- Slot-indexed relative blocks and feature-wise slot normalization.
- Scenario-specific adapter routing and redundant zeroed scenario context.

### LEARNED_STATISTICAL_ASYMMETRY

- Strong Four-Way rotation/relabel bias.
- Ring reflection/chirality bias.
- Smaller Ring generator/critic rotation and slot bias downstream of an exactly rotation-equivariant Stage-I policy.

### PHYSICALLY_JUSTIFIED_ASYMMETRY

Four-Way's canonical offset chart only has an internal 180-degree symmetry. A full 90-degree physical rotation is valid but outside the frozen canonical training chart.

## 11. Recommended interpretation, without changes

No canonical artifact was modified. If future work chooses to address the risk, the minimal candidates to test separately are: encode Ring's reference Flow suffix in the same local chart as its observation; use permutation/rotation-consistent normalization/encoders; and remove or explicitly ablate redundant scenario bookkeeping. Those are future experiments, not changes made here.
