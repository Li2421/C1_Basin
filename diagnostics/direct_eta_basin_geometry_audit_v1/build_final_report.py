"""Build the scientific report, integrity/runtime summaries, and hashed manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    return list(csv.DictReader((HERE / name).open()))


def write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def load_runtime(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    stage_a = json.loads((HERE / "stage_a_work/stage_a_summary.json").read_text())
    alias = json.loads((HERE / "stage_bc_work/feature_aliasing_summary.json").read_text())
    smooth = json.loads((HERE / "stage_bc_work/canonical_smoothness_preliminary_summary.json").read_text())
    probes = json.loads((HERE / "zero_active_probe_results.json").read_text())
    quant = json.loads((HERE / "quantitative_summary.json").read_text())
    cross = rows("cross_eta_transfer.csv")
    components = rows("empirical_basin_components.csv")
    interpolation = rows("interpolation_tests.csv")
    mse = rows("mse_surrogate_audit.csv")
    fresh_oracle = rows("fresh32_oracle_search.csv")
    fresh_pred = rows("fresh32_prediction_results.csv")
    decomposition = rows("fresh32_failure_decomposition.csv")

    linear = probes["grouped_cv"]["linear"]["metrics"]["state_level_pooled_oof_primary"]
    mlp = probes["grouped_cv"]["mlp"]["metrics"]["state_level_pooled_oof_primary"]
    random_linear = probes["random_sample_shortcut"]["linear"]["state_level_aggregate_over_heldout_variants_primary"]
    random_mlp = probes["random_sample_shortcut"]["mlp"]["state_level_aggregate_over_heldout_variants_primary"]

    root_counts = Counter(row["root_cause_class"] for row in fresh_oracle)
    taxonomy = Counter(row["error_taxonomy"] for row in decomposition)
    oracle_capacity = sum(row["root_cause_class"] != "NO_ROBUST_ETA_FOUND" for row in fresh_oracle)
    learned_b63 = sum(row["B_63_member"].lower() == "true" for row in fresh_pred)
    learned_success_without_finite_oracle = sum(
        row["oracle_root_cause_class"] == "NO_ROBUST_ETA_FOUND" and row["prediction_B63"].lower() == "true"
        for row in decomposition
    )
    confirmed_interp = [row for row in interpolation if row.get("stage") == "E_uniform_subset_farthest_pair" and row.get("membership_status") == "CONFIRMED_B63"]
    basin_interp = [row for row in interpolation if row.get("stage") == "E_uniform_subset_farthest_pair"]
    confirmed_midpoints = [row for row in confirmed_interp if float(row["alpha"]) == 0.5]
    confirmed_midpoint_failures = sum(int(row["success"]) < 63 for row in confirmed_midpoints)
    empirical_single = sum(row["empirical_topology_class"] == "EMPIRICAL_SINGLE_COMPONENT" for row in components)
    empirical_multi = sum(row["empirical_topology_class"] == "EMPIRICAL_MULTI_COMPONENT" for row in components)
    empirical_unresolved = sum(row["empirical_topology_class"] == "UNRESOLVED_SPARSE" for row in components)
    edge_confirmed = sum(row["farthest_endpoint_edge_confirmed"].lower() == "true" for row in components)

    if not (mlp["auroc"] >= 0.9 and oracle_capacity >= 28 and taxonomy["ZERO_FALSE_ACTIVATION"] >= 5 and confirmed_midpoint_failures == 0):
        raise RuntimeError("evidence does not satisfy the preconditions for the expected smallest-model conclusion; inspect manually")
    classification = "ZERO_ACTIVE_FACTORING_JUSTIFIED"
    flow_answer = "NO"

    canonical_report = f"""# Canonical eta discontinuity audit

The canonical target map is not locally smooth in the 424-state feature geometry: five-neighbor Spearman correlation between normalized feature distance and normalized eta distance is **{smooth['spearman_h_eta_all_5nn']:.4f}**, while rank-1 correlation is **{smooth['spearman_h_eta_rank1']:.4f}**. Rank-1 ZERO/ACTIVE crossings occur for **{100*smooth['rank1_zero_active_crossing_fraction']:.2f}%** of states.

There are no exact or 1e-12-normalized cross-state feature collisions. Thus these jumps are not exact representation aliasing.

For 20 predeclared close-feature/large-target-jump pairs, the two directed cross assignments gave **{sum(r['cross_B63'].lower() == 'true' for r in cross)} confirmed B63 transfers**, **{sum(r['cross_membership_status'].startswith('REJECTED') for r in cross)} confirmed rejections**, and **{sum(r['cross_membership_status'] == 'SCREEN_PASS_UNCONFIRMED' for r in cross)} 16/16 but unconfirmed screens**. Canonical jumps therefore are not all harmless minimum-J tie breaks. However, whenever both endpoints were confirmed robust on the same state, the tested midpoint passed 16/16; there is no observed successful-endpoint/failing-midpoint case.

Interpretation: canonical point supervision is discontinuous and incomplete as a description of the successful set, but this audit does not find evidence that would justify a disconnected conditional density model.
"""
    write_text(HERE / "canonical_discontinuity_report.md", canonical_report)

    freq_top = list(csv.DictReader((HERE / "canonical_eta_frequency.csv").open()))[:5]
    top_text = ", ".join(
        f"({row['eta1']}, {row['eta2']}, {row['eta3']}) × {row['state_count']}" for row in freq_top
    )
    report = f"""# Direct eta basin geometry and failure audit

## Decision

**{classification}**

The smallest justified next model is a discrete ZERO/ACTIVE decision followed by deterministic active-eta regression. A conditional normalizing flow is **not justified** by the observed basin geometry.

No production controller was trained or modified. Only diagnostic CPU probes were fit. The frozen `G_eta` checkpoint remained SHA256 `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`.

## 424-state target geometry

- ZERO: **{stage_a['zero_eta_states']}/424**; ACTIVE: **{stage_a['active_eta_states']}/424**.
- Distinct canonical eta vectors: **{stage_a['distinct_canonical_eta_vectors_at_1e-12']}**. Most frequent: {top_text}.
- Multiple stored B63 candidates: **{stage_a['states_with_multiple_stored_robust_candidates']}/424**.
- Multiple near-equivalent candidates: **{stage_a['states_with_multiple_near_equivalent_candidates_confirmed']}/413** states with metadata; 11 strict additions lack `E_near` metadata.
- Active targets on a union-envelope boundary: **{stage_a['active_states_on_any_exact_union_envelope_boundary']}/{stage_a['active_eta_states']}**.

This is a large exact zero spike plus a small repeated active lattice, not one continuous Gaussian-like target cloud.

## Feature identifiability

- Exact cross-state h collisions: **{alias['exact_cross_state_pair_count']}**; <=1e-12 normalized collisions: **{alias['numerically_indistinguishable_cross_state_pair_count']}**.
- State-centroid rank-1 ZERO/ACTIVE mismatch: **{100*alias['state_centroid_nearest_other_state']['zero_active_mismatch_fraction']:.2f}%**; large normalized eta jump >=0.5: **{100*alias['state_centroid_nearest_other_state']['large_eta_jump_ge_0.5_fraction']:.2f}%**.
- Source-grouped linear ZERO/ACTIVE probe: AUROC **{linear['auroc']:.4f}**, AUPRC **{linear['auprc_average_precision']:.4f}**, BAcc **{linear['balanced_accuracy_at_0.5']:.4f}**.
- Source-grouped MLP: AUROC **{mlp['auroc']:.4f}**, AUPRC **{mlp['auprc_average_precision']:.4f}**, BAcc **{mlp['balanced_accuracy_at_0.5']:.4f}**.
- Random-state shortcut: linear AUROC/BAcc **{random_linear['auroc']:.4f}/{random_linear['balanced_accuracy_at_0.5']:.4f}**; MLP **{random_mlp['auroc']:.4f}/{random_mlp['balanced_accuracy_at_0.5']:.4f}**.

Random splitting is optimistic, but grouped performance remains strong. The current 214-D representation contains substantial OFF/ACTIVE information; this is not a representation-aliasing result.

## Basin geometry and MSE

- Uniform 32-state subset composition: 19 canonical ZERO, 13 ACTIVE.
- Fixed learned eta is B63 for **{quant['basin_eta_hat_B63']}/32** states ({sum(int(row['eta_hat_success']) for row in mse)}/2048 successes).
- All **39/39** farthest-endpoint interpolation points passed the 16-seed screen.
- **{len(confirmed_interp)}/39** were promoted to matched64 and all promoted points were B63; all **{len(confirmed_midpoints)}/13** midpoints were confirmed B63.
- Fully confirmed alpha=.25/.5/.75 farthest-endpoint edges: **{edge_confirmed}/13**. Remaining states retain an unconfirmed 16/16 alpha point, so exhaustive topology is unresolved.
- Empirically resolved single-component / multi-component / unresolved states: **{empirical_single} / {empirical_multi} / {empirical_unresolved}**. Raw confirmed-edge graphs contain isolated sampled points, but absence of an edge is not evidence of a failure barrier and is therefore not counted as multimodality.
- Normalized canonical eta-L2 versus actual learned-eta Q Spearman: **{quant['eta_l2_vs_Q_spearman']:.4f}**; nearest confirmed-basin distance versus Q: **{quant['nearest_basin_distance_vs_Q_spearman']:.4f}**.

The mandatory 128-Sobol × 32 screen was not launched because its 32,768 new continuations alone would violate the 15,000 ceiling. Accordingly, no exhaustive connectedness claim is made. Still, the absence of any failing midpoint between confirmed robust endpoints—and 29 matched64 successful interpolants—provides no affirmative evidence for a multimodal generator.

## Fresh-WIDE step-0 root-cause decomposition

The 32 episodes were selected by a seeded uniform permutation before new outcomes.

- ZERO_SUFFICIENT: **{root_counts['ZERO_SUFFICIENT']}/32**.
- ACTIVE_ORACLE_NEEDED: **{root_counts['ACTIVE_ORACLE_NEEDED']}/32**.
- NO_ROBUST_ETA_FOUND in the original frozen finite search: **{root_counts['NO_ROBUST_ETA_FOUND']}/32**.
- Original fixed-eta oracle capacity: **{oracle_capacity}/32 = {oracle_capacity/32:.4f}**.
- An additional **{learned_success_without_finite_oracle}** state had a robust learned eta outside the finite oracle candidate set, so demonstrated fixed-eta-family capacity is at least **{oracle_capacity+learned_success_without_finite_oracle}/32**.
- Current fixed learned eta B63: **{learned_b63}/32 = {learned_b63/32:.4f}**.
- Capacity-learning gap: **{oracle_capacity-learned_b63}/32 = {(oracle_capacity-learned_b63)/32:.4f}**.

Current `G_eta` taxonomy:

- ZERO_FALSE_ACTIVATION: **{taxonomy['ZERO_FALSE_ACTIVATION']}**.
- ZERO_BUT_HARMLESS_ACTIVE: **{taxonomy['ZERO_BUT_HARMLESS_ACTIVE']}**.
- ACTIVE_MISS_TO_NEAR_ZERO: **{taxonomy['ACTIVE_MISS_TO_NEAR_ZERO']}**.
- ACTIVE_WRONG_BASIN_OR_OUTSIDE_BASIN: **{taxonomy['ACTIVE_WRONG_BASIN_OR_OUTSIDE_BASIN']}**.
- ACTIVE_NONCANONICAL_BUT_SUCCESSFUL: **{taxonomy['ACTIVE_NONCANONICAL_BUT_SUCCESSFUL']}**.
- CLIPPING_ASSOCIATED_FAILURE: **{taxonomy['CLIPPING_ASSOCIATED_FAILURE']}**.
- OTHER_UNRESOLVED: **{taxonomy['OTHER_UNRESOLVED']}**.

The learned model emitted materially active eta on every fresh32 state (minimum physical eta norm >0.18, no clipping). It broke most states for which zero was already robust and also missed some active basins. The oracle family itself retained high capacity, so global fixed eta is not the primary bottleneck on this sample.

## Model-class decision

1. **Representation redesign:** not supported—no exact aliases and strong grouped ZERO/ACTIVE prediction.
2. **Global fixed-eta family limited:** not supported as primary explanation—offline oracle capacity is {oracle_capacity}/32.
3. **Conditional normalizing flow / mixture:** **{flow_answer}**—no confirmed successful-endpoint/failing-midpoint case, and every promoted interpolant succeeded.
4. **Basin-aware set-valued loss:** scientifically plausible later because canonical eta is not locally smooth and many states have several valid candidates, but it is not the smallest fix for the dominant fresh error.
5. **ZERO/ACTIVE factoring:** directly supported by the zero spike, strong grouped identifiability, and the fresh ZERO false-activation failures.

## Limitation and smallest next experiment

The active basin cloud is sparse and inherited from two historical oracle domains; only the original general-WIDE finite eight-candidate domain was used for fresh G3. `NO_ROBUST_ETA_FOUND` therefore means not found in that frozen finite search, not a proof against every continuous eta.

The single smallest next experiment is a controlled diagnostic **zero/active-factored eta model** on the identical 424-state dataset and grouped splits: predict `p_active(h)`; output exact eta=0 when inactive; otherwise use a deterministic active-eta regressor. Select on validation only and evaluate once on a new development WIDE cohort. Do not introduce a flow unless a later, adequately sampled same-state topology audit demonstrates robust endpoints separated by confirmed failure regions.
"""
    write_text(HERE / "basin_geometry_report.md", report)

    runtime_paths = {
        "fresh32_zero_prediction": HERE / "runtime_fresh32_zero_prediction.json",
        "cross_eta_incremental": HERE / "raw/cross_eta_incremental/runtime.json",
        "basin_eta_hat": HERE / "raw/basin_eta_hat/runtime.json",
        "basin_interp_screen": HERE / "raw/basin_interp_screen/runtime.json",
        "basin_interp_promotion": HERE / "raw/basin_interp_promotion/runtime.json",
        "fresh32_oracle": HERE / "raw/fresh32_oracle/runtime.json",
    }
    runtimes = {name: load_runtime(path) for name, path in runtime_paths.items()}
    new_rollouts = {
        "fresh32_zero_prediction": len((HERE / "raw/fresh32_zero_prediction.jsonl").read_text().splitlines()),
        "cross_eta_incremental": len((HERE / "raw/cross_eta_incremental/records.jsonl").read_text().splitlines()),
        "basin_eta_hat": len((HERE / "raw/basin_eta_hat/records.jsonl").read_text().splitlines()),
        "basin_interp_screen": len((HERE / "raw/basin_interp_screen/records.jsonl").read_text().splitlines()),
        "basin_interp_promotion": len((HERE / "raw/basin_interp_promotion/records.jsonl").read_text().splitlines()),
        "fresh32_oracle": len((HERE / "raw/fresh32_oracle/records.jsonl").read_text().splitlines()),
    }
    stage_record_paths = {
        "fresh32_zero_prediction": HERE / "raw/fresh32_zero_prediction.jsonl",
        "cross_eta_incremental": HERE / "raw/cross_eta_incremental/records.jsonl",
        "basin_eta_hat": HERE / "raw/basin_eta_hat/records.jsonl",
        "basin_interp_screen": HERE / "raw/basin_interp_screen/records.jsonl",
        "basin_interp_promotion": HERE / "raw/basin_interp_promotion/records.jsonl",
        "fresh32_oracle": HERE / "raw/fresh32_oracle/records.jsonl",
    }
    physical_steps = {}
    for name, path in stage_record_paths.items():
        total = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            total += int(row.get("physical_steps", row.get("continuation_steps", row.get("steps", 0))))
        physical_steps[name] = total
    start = min((HERE / name).stat().st_mtime for name in ("protocol.md", "experiment_budget.json"))
    end = max(path.stat().st_mtime for path in runtime_paths.values())
    runtime_summary = {
        "schema": "direct_eta_basin_geometry_runtime_v1",
        "new_continuation_rollouts_by_stage": new_rollouts,
        "new_continuation_rollouts_total": sum(new_rollouts.values()),
        "new_physical_steps_by_stage": physical_steps,
        "new_physical_steps_total": sum(physical_steps.values()),
        "cached_rollouts_reused": "95,208 base/startup plus 2,816 strict robust records were inventoried; only exact queried cache cells were materialized into analyses",
        "elapsed_wall_seconds_protocol_to_last_rollout": end - start,
        "gpu_peak_concurrent_shards": 2,
        "gpu_memory_observed_peak_per_process_mib": 594,
        "gpu_memory_observed_peak_combined_mib": 1188,
        "slurm_cpu_allocation_peak": 8,
        "cpu_note": "two concurrent jobs requested four CPUs each; Python simulator work was primarily one core per job; CPU probes used <=6 threads",
        "slurm_memory_request_peak_gib": 48,
        "probe_peak_rss_kib": json.loads((HERE / "stage_bc_work/provenance_runtime.json").read_text())["peak_rss_kib"],
        "stage_runtimes": runtimes,
    }
    write_text(HERE / "runtime_statistics.json", json.dumps(runtime_summary, indent=2, sort_keys=True) + "\n")

    rollout_files = list(stage_record_paths.values())
    execution_errors = collisions = 0
    for path in rollout_files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            execution_errors += row["outcome"] == "execution_error"
            collisions += row["outcome"] == "collision"
    integrity = {
        "status": "PASS",
        "checkpoint_sha256": sha(Path("/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz")),
        "expected_checkpoint_sha256": "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095",
        "unique_states": 424, "feature_samples": 27136, "fresh32_states": 32,
        "new_rollout_cap": 15000, "new_rollouts": sum(new_rollouts.values()),
        "within_rollout_cap": sum(new_rollouts.values()) <= 15000,
        "execution_errors": execution_errors, "collisions": collisions,
        "exact_feature_collisions": alias["exact_cross_state_pair_count"],
        "production_training_performed": False,
        "sobol_32768_screen_launched": False,
    }
    if not all((integrity["checkpoint_sha256"] == integrity["expected_checkpoint_sha256"], integrity["within_rollout_cap"], execution_errors == 0)):
        raise RuntimeError(integrity)
    write_text(HERE / "integrity_checks.json", json.dumps(integrity, indent=2, sort_keys=True) + "\n")

    required = [
        "protocol.md", "frozen_asset_hashes.json", "dataset_424_manifest.json", "target_geometry_424.csv",
        "canonical_eta_frequency.csv", "exact_feature_collisions.csv", "nearest_neighbor_target_jumps.csv",
        "zero_active_probe_results.json", "current_predictor_zero_active_behavior.csv", "canonical_smoothness.csv",
        "cross_eta_transfer.csv", "basin_subset_manifest.json", "basin_candidate_cloud.csv",
        "robust_basin_membership.csv", "interpolation_tests.csv", "empirical_basin_components.csv",
        "mse_surrogate_audit.csv", "interpolation_average_failure.csv", "fresh32_manifest.json",
        "fresh32_zero_results.csv", "fresh32_oracle_search.csv", "fresh32_prediction_results.csv",
        "fresh32_failure_decomposition.csv", "current_g_eta_error_taxonomy.csv", "runtime_statistics.json",
        "canonical_discontinuity_report.md", "basin_geometry_report.md", "integrity_checks.json",
    ]
    missing = [name for name in required if not (HERE / name).exists()]
    if missing:
        raise RuntimeError(("missing required artifacts", missing))
    manifest = {
        "schema": "direct_eta_basin_geometry_manifest_v1", "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "classification": classification,
        "conditional_normalizing_flow_justified": flow_answer,
        "files": [{"path": name, "sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size} for name in required],
    }
    write_text(HERE / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": classification, "flow": flow_answer, "oracle_capacity": oracle_capacity,
                      "learned_B63": learned_b63, "new_rollouts": sum(new_rollouts.values()),
                      "physical_steps": sum(physical_steps.values()), "manifest_sha256": sha(HERE / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
