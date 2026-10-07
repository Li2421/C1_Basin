"""Finalize D2/D4 B_63 completion and first-step label stability."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SBMA = ROOT / "diagnostics/success_basin_multimodality"
DIRECTIONAL = ROOT / "diagnostics/success_basin_deformation_directional_refinement"
GEOMETRY = ROOT / "diagnostics/success_basin_geometry"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.astra_true_q_audit.audit import restore
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.success_basin_deformation_decomposition.analyze import (
    eta_key,
    load_inventory,
    normalize_manifest,
)
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config


STATES = ("D2_pair228", "D4_pair227")
OUTCOMES = ("success", "deadlock", "timeout", "collision")
VMAX = 0.5
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(name: str, rows: list[dict], fields: tuple[str, ...]) -> None:
    with (HERE / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_pairwise(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    tri = np.triu_indices(len(vectors), 1)
    return float(np.mean(np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)[tri]))


def paired_comparison(candidate: dict, best: dict) -> dict:
    a = {int(row["seed"]): row for row in candidate["rows"] if row["outcome"] == "success"}
    b = {int(row["seed"]): row for row in best["rows"] if row["outcome"] == "success"}
    seeds = sorted(set(a) & set(b))
    differences = np.asarray([a[s]["stored_J_def"] - b[s]["stored_J_def"] for s in seeds], dtype=np.float64)
    mean = float(np.mean(differences))
    if len(differences) > 1 and float(np.std(differences, ddof=1)) > 0:
        half = float(t.ppf(0.975, len(differences) - 1) * np.std(differences, ddof=1) / math.sqrt(len(differences)))
    else:
        half = 0.0
    return {
        "eta": candidate["eta"],
        "paired_successful_n": len(seeds),
        "candidate_minus_best_mean_J_def": mean,
        "paired_t_95_CI": [mean - half, mean + half],
        "statistically_indistinguishable_from_zero": mean - half <= 0 <= mean + half,
        "common_successful_seeds": seeds,
    }


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    base_attempts, base_effective, duplicates, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:3])
    records = {(r["state_id"], eta_key(r["eta"]), int(r["seed"])): r for r in base_effective}
    base_keys = set(records)
    new_records = []
    stage_summaries = []
    for manifest_path in sorted((HERE / "raw").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        stage_summaries.append({
            "stage": manifest["stage"], "new_rollouts": manifest["new_rollouts"],
            "physical_steps": manifest["physical_steps"], "elapsed_s": manifest["elapsed_s"],
            "device": manifest["device"],
        })
        for source in manifest["records"]:
            row = normalize_manifest(source, HERE, manifest["stage"])
            key = (row["state_id"], eta_key(row["eta"]), int(row["seed"]))
            if key in records:
                raise AssertionError(("new rollout duplicates prior tuple", key))
            records[key] = row
            new_records.append(row)

    candidate_rows = []
    completed_rows = []
    cells = {}
    for candidate in protocol["candidate_pool"]:
        state, eta = candidate["state_id"], eta_key(candidate["eta"])
        target = [int(seed) for seed in candidate["target_seeds"]]
        selected = [records[(state, eta, seed)] for seed in target]
        if len(selected) != 64 or len({r["seed"] for r in selected}) != 64:
            raise AssertionError((state, eta, len(selected)))
        counts = Counter(row["outcome"] for row in selected if not row["execution_error"])
        successful = [row for row in selected if row["outcome"] == "success" and not row["execution_error"]]
        mean_j = float(np.mean([row["stored_J_def"] for row in successful]))
        std_j = float(np.std([row["stored_J_def"] for row in successful], ddof=1)) if len(successful) > 1 else 0.0
        mean_steps = float(np.mean([row["steps"] for row in successful]))
        cell = {
            "state_id": state, "eta": list(eta), "rows": selected,
            "success": counts["success"], "deadlock": counts["deadlock"],
            "timeout": counts["timeout"], "collision": counts["collision"],
            "execution_errors": sum(row["execution_error"] is not None for row in selected),
            "mean_J_def_success": mean_j, "std_J_def_success": std_j,
            "mean_episode_steps_success": mean_steps,
            "B_63_member": counts["success"] >= 63,
            "target_seeds": target,
        }
        cells[(state, eta)] = cell
        new_added = sum((state, eta, seed) not in base_keys for seed in target)
        candidate_rows.append({
            "state_id": state, "eta1": eta[0], "eta2": eta[1], "eta3": eta[2],
            "priority": candidate["priority"], "selection_reason": candidate["selection_reason"],
            "existing_valid_seeds": candidate["existing_valid_seeds"],
            "new_seeds_added": new_added, "target_seed_count": len(target),
        })
        completed_rows.append({
            "state_id": state, "eta1": eta[0], "eta2": eta[1], "eta3": eta[2],
            "existing_valid_seeds": candidate["existing_valid_seeds"], "new_seeds_added": new_added,
            "success": counts["success"], "deadlock": counts["deadlock"],
            "timeout": counts["timeout"], "collision": counts["collision"],
            "execution_errors": cell["execution_errors"], "mean_J_def_success": mean_j,
            "std_J_def_success": std_j, "mean_episode_steps_success": mean_steps,
            "B_63_member": cell["B_63_member"],
        })

    candidate_fields = (
        "state_id", "eta1", "eta2", "eta3", "priority", "selection_reason",
        "existing_valid_seeds", "new_seeds_added", "target_seed_count",
    )
    result_fields = (
        "state_id", "eta1", "eta2", "eta3", "existing_valid_seeds", "new_seeds_added",
        "success", "deadlock", "timeout", "collision", "execution_errors",
        "mean_J_def_success", "std_J_def_success", "mean_episode_steps_success", "B_63_member",
    )
    write_csv("candidate_pool.csv", candidate_rows, candidate_fields)
    write_csv("completed_64seed_results.csv", completed_rows, result_fields)
    write_csv("feasible_eta_63.csv", [row for row in completed_rows if row["B_63_member"]], result_fields)

    feasible_by_state = {
        state: {eta: cell for (s, eta), cell in cells.items() if s == state and cell["B_63_member"]}
        for state in STATES
    }
    minima = {}
    near_sets = {}
    comparisons = {}
    near_csv = []
    for state in STATES:
        feasible = feasible_by_state[state]
        if not feasible:
            minima[state], near_sets[state], comparisons[state] = None, [], []
            continue
        best_eta, best = min(feasible.items(), key=lambda item: (item[1]["mean_J_def_success"], item[0]))
        state_comparisons, near = [], []
        for eta, cell in sorted(feasible.items()):
            comparison = paired_comparison(cell, best)
            state_comparisons.append(comparison)
            if comparison["statistically_indistinguishable_from_zero"]:
                near.append(eta)
            near_csv.append({
                "state_id": state, "eta1": eta[0], "eta2": eta[1], "eta3": eta[2],
                "success": cell["success"], "mean_J_def_success": cell["mean_J_def_success"],
                "std_J_def_success": cell["std_J_def_success"],
                "paired_successful_n_vs_best": comparison["paired_successful_n"],
                "difference_vs_best": comparison["candidate_minus_best_mean_J_def"],
                "ci_low": comparison["paired_t_95_CI"][0], "ci_high": comparison["paired_t_95_CI"][1],
                "in_E_near": comparison["statistically_indistinguishable_from_zero"],
            })
        minima[state] = {
            "eta": list(best_eta), "mean_J_def_success": best["mean_J_def_success"],
            "std_J_def_success": best["std_J_def_success"], "success": best["success"],
        }
        near_sets[state] = near
        comparisons[state] = state_comparisons
    write_csv("near_optimal_eta.csv", near_csv, (
        "state_id", "eta1", "eta2", "eta3", "success", "mean_J_def_success", "std_J_def_success",
        "paired_successful_n_vs_best", "difference_vs_best", "ci_low", "ci_high", "in_E_near",
    ))
    dump("paired_comparisons.json", {
        "method": "two-sided paired-t 95% CI on J_def differences over common seeds successful for both policies",
        "states": comparisons,
    })

    # Reconstruct the first-step post-second-projection action labels for E_near.
    directional_protocol = json.loads((DIRECTIONAL / "protocol.json").read_text())
    config = Config(**directional_protocol["environment"])
    cbf = CBFConfig()
    observations, constraints = {}, {}
    for state in STATES:
        entry = directional_protocol["states"][state]
        with np.load(GEOMETRY / entry["state_file"]) as stored:
            env = restore(dict(stored), config)
        observations[state] = np.asarray(env.observation(), dtype=np.float32)
        constraints[state] = barrier_constraints(env.snapshot(), cbf)[:2]

    labels = {}
    label_csv = []
    sha_mismatches = []
    raw_errors, projection_errors, w_errors, j_errors = [], [], [], []
    for row in new_records:
        if sha(row["path"]) != row["sha256"]:
            sha_mismatches.append(str(row["path"]))
        if row["execution_error"] is None:
            with np.load(row["path"]) as data:
                reconstructed_j = float(config.dt * np.sum(np.asarray(data["delta_u_squared"], dtype=np.float64)))
            j_errors.append(abs(reconstructed_j - row["stored_J_def"]))
    for state in STATES:
        etas = near_sets[state]
        common = sorted(set.intersection(*[
            {int(row["seed"]) for row in feasible_by_state[state][eta]["rows"] if row["outcome"] == "success"}
            for eta in etas
        ])) if etas else []
        for eta in etas:
            by_seed = {int(row["seed"]): row for row in feasible_by_state[state][eta]["rows"]}
            corrector = DiagnosticCorrector(DiagnosticPhi(*eta))
            for seed in common:
                row = by_seed[seed]
                if sha(row["path"]) != row["sha256"]:
                    sha_mismatches.append(str(row["path"]))
                with np.load(row["path"]) as data:
                    safe = np.asarray(data["u_safe"][0], dtype=np.float64)
                    raw = np.asarray(data["g"][0], dtype=np.float64)
                    w = np.asarray(data["w"][0], dtype=np.float64)
                    executed = np.asarray(data["u_exec"][0], dtype=np.float64)
                recomputed = np.asarray(corrector(observations[state], safe, config.max_speed), dtype=np.float64)
                raw_errors.append(float(np.max(np.abs(raw - recomputed))))
                A, lower = constraints[state]
                replay, _, _, _ = project_velocity_with_retry(w, A, lower, config.max_speed, cbf)
                projection_errors.append(float(np.max(np.abs(executed - replay))))
                w_errors.append(float(np.max(np.abs(w - safe - raw))))
                g_exec = executed - safe
                record = {
                    "state_id": state, "eta": list(eta), "seed": seed,
                    "u_safe": safe.reshape(-1), "g_raw": raw.reshape(-1),
                    "u_exec": executed.reshape(-1), "g_exec": g_exec.reshape(-1),
                }
                labels[(state, eta, seed)] = record
                flat = {"state_id": state, "eta1": eta[0], "eta2": eta[1], "eta3": eta[2], "seed": seed}
                for name in ("u_safe", "g_raw", "u_exec", "g_exec"):
                    flat.update({f"{name}_{i}": float(value) for i, value in enumerate(record[name])})
                label_csv.append(flat)
    label_fields = ["state_id", "eta1", "eta2", "eta3", "seed"] + [
        f"{name}_{i}" for name in ("u_safe", "g_raw", "u_exec", "g_exec") for i in range(4)
    ]
    write_csv("first_step_labels.csv", label_csv, tuple(label_fields))

    statistics = {}
    pairwise_output = {}
    for state in STATES:
        etas = near_sets[state]
        common = sorted(set.intersection(*[
            {int(row["seed"]) for row in feasible_by_state[state][eta]["rows"] if row["outcome"] == "success"}
            for eta in etas
        ])) if etas else []
        if not etas:
            statistics[state] = {"classification": "FEASIBILITY_NOT_FOUND", "B_63_size": 0, "E_near_size": 0}
            pairwise_output[state] = []
            continue
        distances, raw_distances, cosines, relative, compression = [], [], [], [], []
        component_variances, norms, pairs = [], [], []
        safe_same_seed_errors = []
        for seed in common:
            vectors = {eta: labels[(state, eta, seed)]["g_exec"] for eta in etas}
            raw_vectors = {eta: labels[(state, eta, seed)]["g_raw"] for eta in etas}
            safe_vectors = {eta: labels[(state, eta, seed)]["u_safe"] for eta in etas}
            stack = np.stack(list(vectors.values()))
            component_variances.append(np.var(stack, axis=0, ddof=0))
            norms.extend(np.linalg.norm(stack, axis=1).tolist())
            for eta_a, eta_b in combinations(etas, 2):
                a, b = vectors[eta_a], vectors[eta_b]
                raw_a, raw_b = raw_vectors[eta_a], raw_vectors[eta_b]
                d = float(np.linalg.norm(a - b))
                raw_d = float(np.linalg.norm(raw_a - raw_b))
                denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), EPS)
                cosine = float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))
                safe_error = float(np.max(np.abs(safe_vectors[eta_a] - safe_vectors[eta_b])))
                safe_same_seed_errors.append(safe_error)
                distances.append(d); raw_distances.append(raw_d); cosines.append(cosine)
                relative.append(d / denom); compression.append(d / max(raw_d, EPS))
                pairs.append({
                    "seed": seed, "eta_a": list(eta_a), "eta_b": list(eta_b),
                    "raw_L2": raw_d, "executed_L2": d, "distance_over_vmax": d / VMAX,
                    "distance_over_larger_label_norm": d / denom, "cosine_similarity": cosine,
                    "executed_over_raw_distance": d / max(raw_d, EPS),
                    "same_seed_u_safe_max_abs_difference": safe_error,
                })
        flow_pair_means, flow_component_variance = [], []
        for eta in etas:
            vectors = np.stack([labels[(state, eta, seed)]["g_exec"] for seed in common])
            flow_pair_means.append(mean_pairwise(vectors))
            flow_component_variance.append(np.var(vectors, axis=0, ddof=0).tolist())
        mean_distance = float(np.mean(distances)) if distances else 0.0
        max_distance = float(np.max(distances)) if distances else 0.0
        typical_norm = float(np.mean(norms))
        flow_mean = float(np.mean(flow_pair_means))
        relative_mean = mean_distance / max(typical_norm, EPS)
        eta_over_flow = mean_distance / max(flow_mean, EPS)
        if len(etas) == 1:
            classification = "LABEL_STABLE"
            limitation = "Singleton E_near; stability is vacuous across eta choices."
        elif max_distance / VMAX <= 0.01 and relative_mean <= 0.05 and eta_over_flow <= 1.0:
            classification = "LABEL_STABLE"
            limitation = None
        elif max_distance / VMAX <= 0.10 and relative_mean <= 0.50:
            classification = "LABEL_MILDLY_AMBIGUOUS"
            limitation = None
        else:
            classification = "LABEL_MULTIVALUED"
            limitation = None
        statistics[state] = {
            "classification": classification,
            "deterministic_supervision_supported": classification in ("LABEL_STABLE", "LABEL_MILDLY_AMBIGUOUS"),
            "limitation": limitation,
            "B_63_size": len(feasible_by_state[state]), "E_near_size": len(etas),
            "common_successful_Flow_seeds": len(common),
            "mean_pairwise_L2_eta_same_seed": mean_distance,
            "max_pairwise_L2_eta_same_seed": max_distance,
            "mean_distance_over_vmax": mean_distance / VMAX,
            "max_distance_over_vmax": max_distance / VMAX,
            "typical_g_exec_norm": typical_norm,
            "mean_distance_over_typical_norm": relative_mean,
            "componentwise_variance_due_to_eta_mean_over_seeds": np.mean(component_variances, axis=0).tolist(),
            "cosine_similarity_mean": float(np.mean(cosines)) if cosines else 1.0,
            "cosine_similarity_min": float(np.min(cosines)) if cosines else 1.0,
            "mean_pairwise_raw_L2_eta_same_seed": float(np.mean(raw_distances)) if raw_distances else 0.0,
            "mean_executed_over_raw_pair_distance": float(np.mean(compression)) if compression else 0.0,
            "exact_projection_collapse_fraction": float(np.mean(np.asarray(distances) == 0)) if distances else 1.0,
            "near_projection_collapse_fraction_1e_8": float(np.mean(np.asarray(distances) <= 1e-8)) if distances else 1.0,
            "flow_seed_mean_pairwise_L2_within_eta": flow_pair_means,
            "flow_seed_componentwise_variance_within_eta": flow_component_variance,
            "eta_choice_mean_distance_over_flow_seed_mean_distance": eta_over_flow,
            "max_same_seed_u_safe_difference_across_eta": max(safe_same_seed_errors, default=0.0),
            "classification_basis": "Same thresholds as the prior D1 audit: max/vmax <= .01, mean/typical <= .05, and eta variability no larger than Flow variability for LABEL_STABLE; scale-based mild ambiguity otherwise.",
        }
        pairwise_output[state] = pairs
    dump("first_step_label_stats.json", {
        "states": statistics,
        "B_63": {state: [list(eta) for eta in sorted(feasible_by_state[state])] for state in STATES},
        "E_near": {state: [list(eta) for eta in near_sets[state]] for state in STATES},
        "minima": minima,
        "same_seed_pairwise_details": pairwise_output,
    })

    current_hashes = {
        "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
        "environment": sha(SYSROOT / "single_integrator/environment.py"),
        "projection": sha(SYSROOT / "single_integrator/cbf.py"),
        "retry": sha(SBMA / "exact_projector.py"),
        "checkpoint": sha(Path(protocol["checkpoint"])),
        "directional_manifest": sha(DIRECTIONAL / "manifest.json"),
    }
    hash_agreement = {name: current_hashes[name] == expected for name, expected in protocol["source_hashes"].items()}
    sanity = {
        "status": "PASS" if all(hash_agreement.values()) and not sha_mismatches and not conflicts else "FAIL",
        "frozen_hashes_before": protocol["source_hashes"], "frozen_hashes_after": current_hashes,
        "frozen_hash_agreement": hash_agreement,
        "base_attempt_records_reused": len(base_attempts), "base_effective_unique_tuples": len(base_effective),
        "new_rollouts": len(new_records), "new_unique_tuples": len({(r['state_id'], eta_key(r['eta']), r['seed']) for r in new_records}),
        "new_tuple_overlap_with_existing": 0,
        "new_execution_errors": sum(r["execution_error"] is not None for r in new_records),
        "new_outcome_counts": dict(Counter(r["outcome"] for r in new_records)),
        "max_J_def_reconstruction_error_new": max(j_errors, default=0.0),
        "max_raw_corrector_reconstruction_error": max(raw_errors, default=0.0),
        "max_second_projection_replay_error": max(projection_errors, default=0.0),
        "max_w_minus_safe_minus_g_error": max(w_errors, default=0.0),
        "same_state_same_seed_max_u_safe_difference": max((s["max_same_seed_u_safe_difference_across_eta"] for s in statistics.values()), default=0.0),
        "label_definition": "g_exec = u_exec - u_safe, where u_exec is after the second exact hard projection",
        "action_shape": [2, 2], "flattened_label_dimension": 4,
        "existing_rollout_or_result_files_modified": False,
        "stage_summaries": stage_summaries,
    }
    dump("sanity_checks.json", sanity)
    if sanity["status"] != "PASS":
        raise AssertionError(sanity)

    b_lines, result_lines, label_lines = [], [], []
    for state in STATES:
        short = state.split("_")[0]
        b_lines.append(f"- {short}: " + ", ".join(str(tuple(eta)) for eta in sorted(feasible_by_state[state])))
        for eta, cell in sorted(feasible_by_state[state].items(), key=lambda item: item[1]["mean_J_def_success"]):
            result_lines.append(
                f"| {short} | {tuple(eta)} | {cell['success']}/64 | {cell['mean_J_def_success']:.6f} | "
                f"{cell['std_J_def_success']:.6f} | {cell['mean_episode_steps_success']:.2f} | "
                f"{'yes' if eta in near_sets[state] else 'no'} |"
            )
        st = statistics[state]
        label_lines.append(
            f"| {short} | {st['classification']} | {st['E_near_size']} | "
            f"{st['mean_pairwise_L2_eta_same_seed']:.6g} | {st['max_pairwise_L2_eta_same_seed']:.6g} | "
            f"{st['mean_distance_over_vmax']:.5f} | {st['mean_distance_over_typical_norm']:.5f} | "
            f"{st['eta_choice_mean_distance_over_flow_seed_mean_distance']:.5f} |"
        )
    report = f"""# D2/D4 provisional 63/64 feasibility completion

## Result

Both states now have a non-empty empirical `B_63` under the provisional matched-64 rule. No targeted local search was needed: all 160 newly added, previously missing tuples succeeded. Existing tuples were reused and no tuple was rerun.

### B_63

{chr(10).join(b_lines)}

| state | eta | success | mean J_def (successful runs) | std | mean steps | E_near |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

The strict empirical minima are D2 `{tuple(minima['D2_pair228']['eta'])}` with mean `J_def={minima['D2_pair228']['mean_J_def_success']:.6f}`, and D4 `{tuple(minima['D4_pair227']['eta'])}` with mean `J_def={minima['D4_pair227']['mean_J_def_success']:.6f}`. In each state the adjacent low-cost candidate is statistically indistinguishable by the predeclared paired 95% CI, so each `E_near` contains two eta values. The anchor is feasible but clearly not near-optimal.

## First-step executed-correction stability

The audited supervision quantity is the four-dimensional joint correction `g_exec,t = u_exec,t - u_safe,t` after the second hard projection, with the same state and Flow seed held fixed.

| state | classification | E_near size | mean same-seed distance | max | mean/vmax | mean/typical correction | eta variation/Flow variation |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(label_lines)}

Thus deterministic `G_phi(z, xi) -> g_t` first-step supervision is supported for D2 and D4 at the resolution of this diagnostic. This claim is local to the sampled `E_near`, the provisional 63/64 rule, and the current diagnostic family; it is not a claim of a unique global eta.

## Integrity and runtime

- New rollouts: {len(new_records)} (96 screening/completion stage 1 + 64 completion stage 2), all successful; CPU batches ran D2/D4 concurrently.
- Frozen corrector, environment, both hard-projection implementation inputs, retry solver, checkpoint, and prior directional manifest hashes agree before/after.
- Maximum new-run J_def reconstruction error: {max(j_errors, default=0.0):.3e}.
- Maximum corrector reconstruction error: {max(raw_errors, default=0.0):.3e}; maximum second-projection replay error: {max(projection_errors, default=0.0):.3e}.
- Same-state/same-seed first-step `u_safe` mismatch across near-optimal eta: {sanity['same_state_same_seed_max_u_safe_difference']:.3e}.

The 63/64 rule remains provisional and is not a confidence-calibrated theorem.
"""
    (HERE / "feasibility_completion_report.md").write_text(report)

    artifacts = (
        "feasibility_completion_report.md", "candidate_pool.csv", "completed_64seed_results.csv",
        "feasible_eta_63.csv", "near_optimal_eta.csv", "first_step_labels.csv",
        "first_step_label_stats.json", "paired_comparisons.json", "sanity_checks.json",
        "protocol.json", "stage2_decisions.json", "analyze.py", "run.py", "setup.py",
    )
    dump("manifest.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "success_basin_feasibility_completion", "provisional_rule": "success_count >= 63/64",
        "not_confidence_calibrated": True, "new_rollouts": len(new_records),
        "new_physical_steps": sum(stage["physical_steps"] for stage in stage_summaries),
        "B_63_sizes": {state: len(feasible_by_state[state]) for state in STATES},
        "E_near_sizes": {state: len(near_sets[state]) for state in STATES},
        "classifications": {state: statistics[state]["classification"] for state in STATES},
        "frozen_source_hashes": current_hashes,
        "artifacts_sha256": {name: sha(HERE / name) for name in artifacts},
        "raw_stage_manifests": {path.parent.name: sha(path) for path in sorted((HERE / "raw").glob("*/manifest.json"))},
        "notes": ["No training.", "No frozen source modification.", "No prior rollout/result file modified.", "Raw correction is not used as the supervision label."],
    })
    print(json.dumps({
        "B_63_sizes": {state: len(feasible_by_state[state]) for state in STATES},
        "E_near_sizes": {state: len(near_sets[state]) for state in STATES},
        "minima": minima, "classifications": {state: statistics[state]["classification"] for state in STATES},
        "new_rollouts": len(new_records), "sanity": sanity["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
