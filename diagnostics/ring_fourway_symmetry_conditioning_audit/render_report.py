#!/usr/bin/env python3
from pathlib import Path
import json
import hashlib

OUT = Path(__file__).resolve().parent


def load(name): return json.loads((OUT / name).read_text())
def dump(name, value): (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


exact = load("exact_component_test_results.json")
mac = load("macflow_symmetry_metrics.json")
gen = load("generator_proposal_distribution_metrics.json")
critic = load("critic_ranking_symmetry_metrics.json")
shortcut = load("scenario_id_shortcut_audit.json")
serial = load("serialization_audit.json")
closed = load("closed_loop_metamorphic_replay_results.json")
norm = load("../orthoflow3_generator_critic_v1/normalization.json")
provenance = load("provenance_hashes.json")

current_hashes = {relative: hashlib.sha256((OUT.parents[1] / relative).read_bytes()).hexdigest()
                  for relative in provenance["files"]}
dump("hash_verification.json", {
    "matches": {key: current_hashes[key] == value for key, value in provenance["files"].items()},
    "all_canonical_hashes_preserved": all(current_hashes[key] == value for key, value in provenance["files"].items()),
    "frozen_test_states_used": 0,
})

pipeline = {
  "global": {
    "agent_order": ["A", "B", "C", "D"],
    "pair_order": [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],
    "history": "none",
    "conditioning_schema": "native_observation_plus_reference_current_flow_v1",
    "context": {"raw_dimension": 33, "environment_scalar_union_dimension": 30,
                "scenario_one_hot_dimension": 3, "environment_keys": norm["environment_keys"],
                "normalization": "per-scenario train-only elementwise z-score"},
    "generator": "scenario-specific native-input Dense(96) adapter + shared context Dense(32), MLP 128/64, six outputs",
    "generator_distribution": "three unconstrained means and sigma=0.025+0.275*sigmoid(raw[3:]); tanh into frozen eta box",
    "critic": "scenario-specific native-input Dense(96) adapter + shared context/eta encoders and shared MLP; sigmoid Q score",
    "eta": "three scalar coefficients [eta_g, eta_perp, eta_rel] invariant under rigid physical symmetry",
    "execution": "MACFlow world velocity -> first hard projection -> OrthoFlow3 world correction -> second hard projection -> single-integrator step"
  },
  "four_way_intersection": {
    "physical_state": "positions[4,2], last-applied velocities[4,2], fixed/per-state goals[4,2] in world metres and m/s",
    "stage1_observation_shape": [4,18], "flat_stage1_dimension": 72,
    "row_fields": ["own world position[2]", "own world velocity[2]", "goal-own position[2]",
                   "for j in ascending A/B/C/D excluding self: relative world position[2], relative world velocity[2]"],
    "stage1_action": "world Cartesian velocity [4,2], flattened A.x,A.y,...,D.x,D.y",
    "basin_h_shape": [80], "basin_h": "flattened world observation[72] + bounded reference MACFlow world velocity[8]",
    "goals": "A N->S, B E->W, C S->N, D W->E; goal displacement is observed",
    "geometry": "square boundary not present explicitly in h; fixed through scenario/checkpoint and c descriptor world_half_extent",
    "time": "timestep and normalized episode time are retained in structured state but absent from h"
  },
  "ring_exchange": {
    "physical_state": "positions[4,2], last-applied velocities[4,2], per-agent goals[4,2] in world metres and m/s",
    "stage1_observation_shape": [4,23], "flat_stage1_dimension": 92,
    "local_frame": "each row uses outward radial axis and CCW tangent axis at that agent's current position",
    "row_fields": ["own [radius,0]", "own local [radial,tangent] velocity", "local goal displacement",
                   "for j in ascending A/B/C/D excluding self: observer-frame relative position[2], velocity[2]",
                   "[-radius,0,obstacle_radius,outer_radius,central-obstacle clearance]"],
    "stage1_action": "MACFlow emits local radial/tangent velocity [4,2], then it is mapped to world velocity",
    "basin_h_shape": [100],
    "basin_h": "flattened local observation[92] + bounded reference MACFlow WORLD velocity[8]",
    "mixed_frame_warning": "the 92-D prefix is rotation-invariant local data while the 8-D suffix is arbitrary world x/y",
    "geometry": "central and outer radii are directly visible in every observation row; descriptor repeats them in c",
    "time": "timestep and normalized episode time are retained in structured state but absent from h"
  }
}
dump("conditioning_pipeline_map.json", pipeline)

conventions = {
  "world_xy_orientation": {"classification": "REPRESENTATIONAL", "scope": "both",
    "qualification": "only when the complete physical scene, goals and geometry are rigidly transformed"},
  "four_agent_names_ABCD": {"classification": "REPRESENTATIONAL",
    "qualification": "a consistent permutation must move positions, velocities, goals and actions together"},
  "four_cardinal_route_order": {"classification": "REPRESENTATIONAL",
    "qualification": "route/goal itself is physical; which tensor slot names that complete route is not"},
  "four_offset_lane_handedness": {"classification": "PHYSICAL",
    "qualification": "the frozen q-offset layout has an exact canonical 180-degree symmetry, not an internal 90-degree one; 90 degrees is equivalent only after rotating the complete goals/layout"},
  "ring_cardinal_start_index": {"classification": "REPRESENTATIONAL",
    "qualification": "global start angle is randomized and agent labels carry no priority"},
  "ring_CCW_tangent_positive": {"classification": "REPRESENTATIONAL",
    "qualification": "reflection flips tangent sign even though the circular physical problem remains equivalent"},
  "goal_assignment": {"classification": "PHYSICAL",
    "qualification": "changing a goal alone changes the task; permuting each agent with its own goal is only relabeling"},
  "scenario_one_hot_and_branch": {"classification": "REPRESENTATIONAL",
    "qualification": "bookkeeping/routing rather than a physical field"},
  "environment_radii_extents": {"classification": "PHYSICAL"},
  "flatten_and_relative_block_order": {"classification": "REPRESENTATIONAL"},
  "eta_coordinates": {"classification": "PHYSICAL_SCALAR_ON_EQUIVARIANT_BASES",
    "qualification": "the same eta must be used under rigid rotation, reflection or consistent relabeling"}
}
dump("arbitrary_conventions.json", conventions)

agent = {
 "canonical_order": ["A","B","C","D"],
 "architectural_permutation_equivariance": {"macflow": False, "generator": False, "critic": False},
 "exact_physical_components_permutation_test": {
   s: {"failures": len(exact[s]["exact_failures"])} for s in exact},
 "four_way": {
   "macflow_cyclic_relabel_paired_rmse_over_speed": mac["four_way_intersection"]["initial_action"]["cyclic_relabel"]["paired_action_rmse_over_speed"],
   "generator_cyclic_relabel_normalized_mean_delta": gen["four_way_intersection"]["by_transform"]["cyclic_relabel"]["normalized_mean_delta"],
   "critic_cyclic_relabel_top1_match": critic["four_way_intersection"]["by_transform"]["cyclic_relabel"]["top1_match"],
   "normalization_is_agent_index_dependent": True,
   "assessment": "strong learned slot dependence; no hard-coded environment priority found"
 },
 "ring_exchange": {
   "macflow_cyclic_relabel_paired_rmse_over_speed": mac["ring_exchange"]["initial_action"]["cyclic_relabel"]["paired_action_rmse_over_speed"],
   "generator_cyclic_relabel_normalized_mean_delta": gen["ring_exchange"]["by_transform"]["cyclic_relabel"]["normalized_mean_delta"],
   "critic_cyclic_relabel_top1_match": critic["ring_exchange"]["by_transform"]["cyclic_relabel"]["top1_match"],
   "normalization_is_agent_index_dependent": True,
   "assessment": "Stage-I learned near-permutation symmetry despite no architectural guarantee; downstream eta models retain moderate slot dependence"
 },
 "module_priority_search": "no environment/policy right-of-way or fixed-priority feature found; positional slots remain available as a statistical shortcut"
}
dump("agent_order_audit.json", agent)

classification = {
 "four_way_intersection": "REPRESENTATION_SHORTCUT_RISK",
 "ring_exchange": "REPRESENTATION_SHORTCUT_RISK",
 "joint_generator_critic": "REPRESENTATION_SHORTCUT_RISK",
 "exact_implementation_bug_found": False,
 "discrepancy_categories": {
   "EXACT_IMPLEMENTATION_VIOLATION": [],
   "REPRESENTATION_CONVENTION_DEPENDENCE": [
     "Four-Way world-coordinate and fixed-slot flattening",
     "Ring mixed local-observation/world-flow conditioning",
     "ascending slot-index relative blocks",
     "scenario-specific adapters plus redundant scenario one-hot/context serialization"
   ],
   "LEARNED_STATISTICAL_ASYMMETRY": [
     "Four-Way MACFlow/generator/critic world-rotation and slot bias",
     "Ring reflection/chirality bias",
     "residual Ring downstream rotation and cyclic-relabel bias"
   ],
   "PHYSICALLY_JUSTIFIED_ASYMMETRY": [
     "Four-Way frozen offset layout is internally 180-degree but not 90-degree canonical; full-scene 90-degree tests rotate goals/layout out of the training chart"
   ]
 }}
dump("final_classification.json", classification)

f = mac["four_way_intersection"]["initial_action"]
r = mac["ring_exchange"]["initial_action"]
fg = gen["four_way_intersection"]["by_transform"]
rg = gen["ring_exchange"]["by_transform"]
fc = critic["four_way_intersection"]["by_transform"]
rc = critic["ring_exchange"]["by_transform"]

report = f"""# Four-Way / Ring symmetry and conditioning audit

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
| Four-Way | {exact['four_way_intersection']['cases']} | {len(exact['four_way_intersection']['exact_failures'])} | {exact['four_way_intersection']['max_errors']['projection_error']:.3g} | {exact['four_way_intersection']['max_errors']['second_projection_error']:.3g} | {max(exact['four_way_intersection']['max_errors'][k] for k in ('basis_goal_error','basis_flow_perp_error','basis_rel_error','full_correction_error')):.3g} |
| Ring | {exact['ring_exchange']['cases']} | {len(exact['ring_exchange']['exact_failures'])} | {exact['ring_exchange']['max_errors']['projection_error']:.3g} | {exact['ring_exchange']['max_errors']['second_projection_error']:.3g} | {max(exact['ring_exchange']['max_errors'][k] for k in ('basis_goal_error','basis_flow_perp_error','basis_rel_error','full_correction_error')):.3g} |

The same eta produces the correspondingly transformed correction. Collision outcome and sorted wall/pair distances were invariant. No tolerance was relaxed after execution.

## 4. MACFlow symmetry

Paired action RMSE is normalized by max speed:

| Transform | Four-Way | Ring |
|---|---:|---:|
| rotate 90 | {f['rotate_90']['paired_action_rmse_over_speed']['mean']:.3f} | {r['rotate_90']['paired_action_rmse_over_speed']['mean']:.3g} |
| rotate 180 | {f['rotate_180']['paired_action_rmse_over_speed']['mean']:.3f} | {r['rotate_180']['paired_action_rmse_over_speed']['mean']:.3g} |
| rotate 270 | {f['rotate_270']['paired_action_rmse_over_speed']['mean']:.3f} | {r['rotate_270']['paired_action_rmse_over_speed']['mean']:.3g} |
| cyclic relabel | {f['cyclic_relabel']['paired_action_rmse_over_speed']['mean']:.3f} | {r['cyclic_relabel']['paired_action_rmse_over_speed']['mean']:.3f} |
| reflection | N/A | {r['reflect_x_axis']['paired_action_rmse_over_speed']['mean']:.3f} |

Ring rotation errors are exactly zero at sampled precision. Four-Way 90-degree full-scene rotation is severe; even its canonical 180-degree relabel has normalized paired RMSE {f['canonical_rotate_180_relabel']['paired_action_rmse_over_speed']['mean']:.3f}. Ring cyclic relabel is mild, while reflection is severe ({r['reflect_x_axis']['paired_action_rmse_over_speed']['mean']:.3f}), showing learned chirality sensitivity.

In full-horizon hard-safety-only replays, Ring matched outcomes in {mac['ring_exchange']['full_horizon_hard_safety']['outcome_match_fraction']:.0%} of four pairs; its two pure rotations have effectively zero back-transformed trajectory error. Four-Way matched {mac['four_way_intersection']['full_horizon_hard_safety']['outcome_match_fraction']:.0%}: both 90-degree pairs changed success to timeout, while both canonical 180-degree pairs preserved the terminal outcome.

## 5. Generator proposal distribution

No Q16/basin experiment was repeated. “Robust hit” in the diagnostic files means critic-predicted score >=0.9375 only, not a new robust certificate.

| Transform | Four mean eta delta | Four cloud match | Ring mean eta delta | Ring cloud match |
|---|---:|---:|---:|---:|
| rotate 90 | {fg['rotate_90']['normalized_mean_delta']['mean']:.3f} | {fg['rotate_90']['cloud_hungarian_mean']['mean']:.3f} | {rg['rotate_90']['normalized_mean_delta']['mean']:.3f} | {rg['rotate_90']['cloud_hungarian_mean']['mean']:.3f} |
| rotate 180 | {fg['rotate_180']['normalized_mean_delta']['mean']:.3f} | {fg['rotate_180']['cloud_hungarian_mean']['mean']:.3f} | {rg['rotate_180']['normalized_mean_delta']['mean']:.3f} | {rg['rotate_180']['cloud_hungarian_mean']['mean']:.3f} |
| cyclic relabel | {fg['cyclic_relabel']['normalized_mean_delta']['mean']:.3f} | {fg['cyclic_relabel']['cloud_hungarian_mean']['mean']:.3f} | {rg['cyclic_relabel']['normalized_mean_delta']['mean']:.3f} | {rg['cyclic_relabel']['cloud_hungarian_mean']['mean']:.3f} |

Ring rotation is good but not exact (mean delta 0.067–0.093 in normalized eta coordinates), precisely unlike its exact Stage-I rotation behavior. Ring reflection delta is {rg['reflect_x_axis']['normalized_mean_delta']['mean']:.3f}. Four-Way pure rotations and cyclic relabeling change the proposal distribution substantially; its canonical 180-degree relabel is much closer ({fg['canonical_rotate_180_relabel']['normalized_mean_delta']['mean']:.3f}).

## 6. Critic symmetry

| Transform | Four score abs diff | Four top-1 match | Ring score abs diff | Ring top-1 match |
|---|---:|---:|---:|---:|
| rotate 90 | {fc['rotate_90']['mean_abs_score_delta']['mean']:.3f} | {fc['rotate_90']['top1_match']['mean']:.0%} | {rc['rotate_90']['mean_abs_score_delta']['mean']:.4f} | {rc['rotate_90']['top1_match']['mean']:.0%} |
| rotate 180 | {fc['rotate_180']['mean_abs_score_delta']['mean']:.3f} | {fc['rotate_180']['top1_match']['mean']:.0%} | {rc['rotate_180']['mean_abs_score_delta']['mean']:.4f} | {rc['rotate_180']['top1_match']['mean']:.0%} |
| cyclic relabel | {fc['cyclic_relabel']['mean_abs_score_delta']['mean']:.3f} | {fc['cyclic_relabel']['top1_match']['mean']:.0%} | {rc['cyclic_relabel']['mean_abs_score_delta']['mean']:.4f} | {rc['cyclic_relabel']['top1_match']['mean']:.0%} |

Four score shifts are severe for 90/270 rotation and pure relabeling. The canonical 180-degree relabel is much cleaner: mean absolute score change {fc['canonical_rotate_180_relabel']['mean_abs_score_delta']['mean']:.4f}, top-1 match {fc['canonical_rotate_180_relabel']['top1_match']['mean']:.0%}. Ring rotations preserve ranking well (Spearman 0.956–0.969), while cyclic relabeling lowers it to {rc['cyclic_relabel']['spearman']['mean']:.3f}; reflection produces mean score change {rc['reflect_x_axis']['mean_abs_score_delta']['mean']:.3f}.

## 7. Agent ordering and normalization

Exact components pass consistent permutation tests, so no physical module treats a slot as priority. Learned networks are ordinary flattened MLPs and are not architecturally permutation-equivariant. All Stage-I and basin normalization statistics are feature/slot specific.

- Four-Way: pure cyclic relabel MACFlow RMSE {f['cyclic_relabel']['paired_action_rmse_over_speed']['mean']:.3f}; generator normalized mean delta {fg['cyclic_relabel']['normalized_mean_delta']['mean']:.3f}; critic top-1 agreement {fc['cyclic_relabel']['top1_match']['mean']:.0%}.
- Ring: corresponding values {r['cyclic_relabel']['paired_action_rmse_over_speed']['mean']:.3f}, {rg['cyclic_relabel']['normalized_mean_delta']['mean']:.3f}, and {rc['cyclic_relabel']['top1_match']['mean']:.0%}.

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
"""
(OUT / "REPORT.md").write_text(report)
print(json.dumps({"report": str(OUT / 'REPORT.md'), "classification": classification}, indent=2))
