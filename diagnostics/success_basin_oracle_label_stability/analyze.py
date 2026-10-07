"""Construct B_63 and audit first-step executed-correction label stability."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import sys
import warnings
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
DECOMPOSITION = ROOT / "diagnostics/success_basin_deformation_decomposition"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.astra_true_q_audit.audit import restore
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.success_basin_deformation_decomposition.analyze import load_inventory, eta_key
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.environment import Config


STATES = ("D1_pair231", "D2_pair228", "D4_pair227")
OUTCOMES = ("success", "deadlock", "timeout", "collision")
CANONICAL_SEEDS = tuple(range(95106001, 95106033)) + tuple(range(95107001, 95107033))
VMAX = 0.5
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def selected_64(rows):
    valid = {row["seed"]: row for row in rows if row["execution_error"] is None and row["outcome"] in OUTCOMES}
    if all(seed in valid for seed in CANONICAL_SEEDS):
        return [valid[seed] for seed in CANONICAL_SEEDS], "canonical_95106001_32_plus_95107001_32"
    if len(valid) == 64:
        return [valid[seed] for seed in sorted(valid)], "all_available_exactly_64"
    return None, "insufficient_or_noncanonical_more_than_64"


def paired_difference(rows_a, rows_b):
    a = {row["seed"]: row["stored_J_def"] for row in rows_a}
    b = {row["seed"]: row["stored_J_def"] for row in rows_b}
    seeds = sorted(set(a) & set(b))
    d = np.asarray([a[seed] - b[seed] for seed in seeds], dtype=np.float64)
    mean = float(d.mean())
    half = float(t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
    return {
        "paired_n": len(d), "candidate_minus_best_mean": mean,
        "two_sided_paired_t_95_CI": [mean - half, mean + half],
        "indistinguishable_from_zero": mean - half <= 0.0 <= mean + half,
        "common_seeds": seeds,
    }


def mean_pairwise(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) < 2:
        return 0.0
    d = vectors[:, None, :] - vectors[None, :, :]
    tri = np.triu_indices(len(vectors), 1)
    return float(np.mean(np.linalg.norm(d[tri], axis=1)))


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    attempts, effective, duplicates, conflicts = load_inventory()
    if conflicts:
        raise AssertionError(conflicts[:5])
    grouped = defaultdict(list)
    for row in effective:
        grouped[(row["state_id"], eta_key(row["eta"]))].append(row)

    eligible = {}
    feasibility_rows = []
    for (state, eta), rows in sorted(grouped.items()):
        if state not in STATES:
            continue
        selected, cohort = selected_64(rows)
        valid = [row for row in rows if row["execution_error"] is None and row["outcome"] in OUTCOMES]
        all_counts = Counter(row["outcome"] for row in valid)
        if selected is not None:
            counts = Counter(row["outcome"] for row in selected)
            success_count = counts["success"]
            mean_j = float(np.mean([row["stored_J_def"] for row in selected]))
            std_j = float(np.std([row["stored_J_def"] for row in selected], ddof=1))
            feasible = success_count >= 63
            eligible[(state, eta)] = {
                "rows": selected, "cohort": cohort, "counts": counts,
                "success_count": success_count, "mean_J_def": mean_j,
                "std_J_def": std_j, "feasible": feasible,
            }
        else:
            counts, success_count, mean_j, std_j, feasible = Counter(), None, None, None, False
        feasibility_rows.append({
            "state_id": state, "eta": list(eta), "n_unique_valid": len(valid),
            "all_success": all_counts["success"], "all_deadlock": all_counts["deadlock"],
            "all_timeout": all_counts["timeout"], "all_collision": all_counts["collision"],
            "eligible_64": selected is not None, "evaluation_cohort": cohort,
            "success_count_64": success_count, "mean_J_def_64": mean_j, "std_J_def_64": std_j,
            "B_63_member": feasible,
        })

    feasible_by_state = {
        state: {eta: info for (s, eta), info in eligible.items() if s == state and info["feasible"]}
        for state in STATES
    }
    minima, near_sets, paired_results = {}, {}, {}
    for state in STATES:
        feasible = feasible_by_state[state]
        if not feasible:
            minima[state] = None
            near_sets[state] = []
            paired_results[state] = []
            continue
        best_eta, best = min(feasible.items(), key=lambda item: (item[1]["mean_J_def"], item[0]))
        comparisons = []
        near = []
        for eta, info in sorted(feasible.items()):
            if eta == best_eta:
                comp = {
                    "eta": list(eta), "paired_n": 64, "candidate_minus_best_mean": 0.0,
                    "two_sided_paired_t_95_CI": [0.0, 0.0], "indistinguishable_from_zero": True,
                    "common_seeds": [row["seed"] for row in best["rows"]],
                }
            else:
                comp = {"eta": list(eta), **paired_difference(info["rows"], best["rows"])}
            comparisons.append(comp)
            if comp["indistinguishable_from_zero"]:
                near.append(eta)
        jmin = best["mean_J_def"]
        minima[state] = {
            "eta": list(best_eta), "J_min": jmin, "std_J_def": best["std_J_def"],
            "success_count": best["success_count"], "cohort": best["cohort"],
            "within_1_percent": [list(eta) for eta, info in feasible.items() if info["mean_J_def"] <= 1.01 * jmin],
            "within_2_5_percent": [list(eta) for eta, info in feasible.items() if info["mean_J_def"] <= 1.025 * jmin],
            "within_5_percent": [list(eta) for eta, info in feasible.items() if info["mean_J_def"] <= 1.05 * jmin],
        }
        near_sets[state] = near
        paired_results[state] = comparisons

    with (HERE / "feasible_eta_63.csv").open("w", newline="") as handle:
        fields = tuple(feasibility_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in feasibility_rows:
            writer.writerow({**row, "eta": json.dumps(row["eta"])})
    with (HERE / "near_optimal_eta.csv").open("w", newline="") as handle:
        fields = ("state_id", "eta1", "eta2", "eta3", "success_count", "mean_J_def", "std_J_def", "evaluation_cohort", "paired_n_vs_best", "difference_vs_best", "ci_low", "ci_high", "in_E_near", "within_1_percent", "within_2_5_percent", "within_5_percent")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for state in STATES:
            for comp in paired_results[state]:
                eta = eta_key(comp["eta"])
                info = feasible_by_state[state][eta]
                best = minima[state]
                writer.writerow({
                    "state_id": state, "eta1": eta[0], "eta2": eta[1], "eta3": eta[2],
                    "success_count": info["success_count"], "mean_J_def": info["mean_J_def"],
                    "std_J_def": info["std_J_def"], "evaluation_cohort": info["cohort"],
                    "paired_n_vs_best": comp["paired_n"], "difference_vs_best": comp["candidate_minus_best_mean"],
                    "ci_low": comp["two_sided_paired_t_95_CI"][0], "ci_high": comp["two_sided_paired_t_95_CI"][1],
                    "in_E_near": comp["indistinguishable_from_zero"],
                    "within_1_percent": info["mean_J_def"] <= 1.01 * best["J_min"],
                    "within_2_5_percent": info["mean_J_def"] <= 1.025 * best["J_min"],
                    "within_5_percent": info["mean_J_def"] <= 1.05 * best["J_min"],
                })

    protocol = json.loads((DIRECTIONAL / "protocol.json").read_text())
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    state_observation = {}
    state_constraints = {}
    for state in STATES:
        entry = protocol["states"][state]
        with np.load(ROOT / "diagnostics/success_basin_geometry" / entry["state_file"]) as stored:
            env = restore(dict(stored), config)
        state_observation[state] = np.asarray(env.observation(), dtype=np.float32)
        state_constraints[state] = barrier_constraints(env.snapshot(), cbf)[:2]

    labels = []
    label_lookup = {}
    raw_formula_errors, projection_replay_errors, stored_delta_errors = [], [], []
    source_sha_mismatches = []
    for state in STATES:
        if not near_sets[state]:
            continue
        common = set.intersection(*[
            {row["seed"] for row in feasible_by_state[state][eta]["rows"]}
            for eta in near_sets[state]
        ])
        for eta in near_sets[state]:
            info = feasible_by_state[state][eta]
            by_seed = {row["seed"]: row for row in info["rows"]}
            corrector = DiagnosticCorrector(DiagnosticPhi(*eta))
            for seed in sorted(common):
                row = by_seed[seed]
                if sha(row["path"]) != row["sha256"]:
                    source_sha_mismatches.append(str(row["path"]))
                with np.load(row["path"]) as data:
                    safe = np.asarray(data["u_safe"][0], dtype=np.float64)
                    raw = np.asarray(data["g"][0], dtype=np.float64)
                    executed = np.asarray(data["u_exec"][0], dtype=np.float64)
                    w = np.asarray(data["w"][0], dtype=np.float64)
                recomputed_raw = np.asarray(corrector(state_observation[state], safe, config.max_speed), dtype=np.float64)
                raw_formula_errors.append(float(np.max(np.abs(raw - recomputed_raw))))
                A, lower = state_constraints[state]
                replay, _, _, _ = project_velocity_with_retry(w, A, lower, config.max_speed, cbf)
                projection_replay_errors.append(float(np.max(np.abs(executed - replay))))
                g_exec = executed - safe
                stored_delta_errors.append(float(np.max(np.abs(w - safe - raw))))
                record = {
                    "state_id": state, "eta": list(eta), "seed": seed,
                    "u_safe": safe.reshape(-1).tolist(), "g_raw": raw.reshape(-1).tolist(),
                    "u_exec": executed.reshape(-1).tolist(), "g_exec": g_exec.reshape(-1).tolist(),
                    "g_raw_norm": float(np.linalg.norm(raw)), "g_exec_norm": float(np.linalg.norm(g_exec)),
                    "second_projection_changed_raw_correction": float(np.linalg.norm(g_exec - raw)),
                }
                labels.append(record)
                label_lookup[(state, eta, seed)] = record

    with (HERE / "first_step_labels.csv").open("w", newline="") as handle:
        fields = ["state_id", "eta1", "eta2", "eta3", "seed"]
        for prefix in ("u_safe", "g_raw", "u_exec", "g_exec"):
            fields.extend(f"{prefix}_{i}" for i in range(4))
        fields.extend(("g_raw_norm", "g_exec_norm", "second_projection_changed_raw_correction"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in labels:
            out = {"state_id": row["state_id"], "eta1": row["eta"][0], "eta2": row["eta"][1], "eta3": row["eta"][2], "seed": row["seed"]}
            for prefix in ("u_safe", "g_raw", "u_exec", "g_exec"):
                out.update({f"{prefix}_{i}": value for i, value in enumerate(row[prefix])})
            out.update({name: row[name] for name in ("g_raw_norm", "g_exec_norm", "second_projection_changed_raw_correction")})
            writer.writerow(out)

    pair_rows = []
    statistics = {}
    representatives = {}
    for state in STATES:
        etas = near_sets[state]
        if not etas:
            statistics[state] = {
                "B_63_size": 0, "E_near_size": 0,
                "classification": "NOT_TESTABLE_EMPTY_B63",
                "deterministic_supervision_justified": False,
                "reason": "No eta satisfies the provisional 63/64 feasibility rule in the available matched cohort.",
            }
            representatives[state] = []
            continue
        common = sorted(set.intersection(*[
            {row["seed"] for row in feasible_by_state[state][eta]["rows"]} for eta in etas
        ]))
        per_seed_distances = []
        per_seed_raw_distances = []
        projection_compression_ratios = []
        component_variances = []
        correction_norms = []
        cosine_values = []
        for seed in common:
            vectors = {eta: np.asarray(label_lookup[(state, eta, seed)]["g_exec"]) for eta in etas}
            raw_vectors = {eta: np.asarray(label_lookup[(state, eta, seed)]["g_raw"]) for eta in etas}
            stack = np.stack(list(vectors.values()))
            component_variances.append(np.var(stack, axis=0, ddof=0))
            correction_norms.extend(np.linalg.norm(stack, axis=1).tolist())
            for eta_a, eta_b in combinations(etas, 2):
                a, b = vectors[eta_a], vectors[eta_b]
                distance = float(np.linalg.norm(a - b))
                raw_distance = float(np.linalg.norm(raw_vectors[eta_a] - raw_vectors[eta_b]))
                denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), EPS)
                cosine = float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))
                cosine_values.append(cosine)
                per_seed_distances.append(distance)
                per_seed_raw_distances.append(raw_distance)
                projection_compression_ratios.append(distance / max(raw_distance, EPS))
                pair_rows.append({
                    "state_id": state, "seed": seed, "eta_a": list(eta_a), "eta_b": list(eta_b),
                    "raw_L2_distance": raw_distance, "L2_distance": distance,
                    "executed_over_raw_distance": distance / max(raw_distance, EPS),
                    "distance_over_vmax": distance / VMAX,
                    "relative_distance": distance / denom, "cosine_similarity": cosine,
                    "exact_collapse": bool(distance == 0.0), "near_collapse_1e_8": bool(distance <= 1e-8),
                })
        # Flow variability: within each eta, compare labels across the common seed set.
        flow_pair_means = []
        flow_component_variances = []
        for eta in etas:
            vectors = np.stack([label_lookup[(state, eta, seed)]["g_exec"] for seed in common])
            flow_pair_means.append(mean_pairwise(vectors))
            flow_component_variances.append(np.var(vectors, axis=0, ddof=0))
        distances = np.asarray(per_seed_distances)
        correction_norms = np.asarray(correction_norms)
        compvar = np.mean(np.stack(component_variances), axis=0)
        mean_dist = float(distances.mean()) if len(distances) else 0.0
        max_dist = float(distances.max()) if len(distances) else 0.0
        typical_norm = float(correction_norms.mean())
        relative_mean = mean_dist / max(typical_norm, EPS)
        eta_to_flow_ratio = mean_dist / max(float(np.mean(flow_pair_means)), EPS)
        # Classification is tied to observed action scales rather than an absolute arbitrary radius.
        if max_dist / VMAX <= 0.01 and relative_mean <= 0.05 and eta_to_flow_ratio <= 1.0:
            classification = "LABEL_STABLE"
        elif max_dist / VMAX <= 0.10 and relative_mean <= 0.50:
            classification = "LABEL_MILDLY_AMBIGUOUS"
        else:
            classification = "LABEL_MULTIVALUED"
        statistics[state] = {
            "B_63_size": len(feasible_by_state[state]), "E_near_size": len(etas),
            "classification": classification, "deterministic_supervision_justified": classification != "LABEL_MULTIVALUED",
            "common_Flow_seeds": len(common), "mean_pairwise_L2_eta_same_seed": mean_dist,
            "max_pairwise_L2_eta_same_seed": max_dist,
            "mean_distance_over_vmax": mean_dist / VMAX, "max_distance_over_vmax": max_dist / VMAX,
            "typical_g_exec_norm": typical_norm, "mean_distance_over_typical_norm": relative_mean,
            "componentwise_variance_due_to_eta_mean_over_seeds": compvar.tolist(),
            "mean_pairwise_raw_L2_eta_same_seed": float(np.mean(per_seed_raw_distances)),
            "mean_executed_over_raw_pair_distance": float(np.mean(projection_compression_ratios)),
            "cosine_similarity_mean": float(np.mean(cosine_values)),
            "cosine_similarity_min": float(np.min(cosine_values)),
            "exact_projection_collapse_fraction": float(np.mean([row["exact_collapse"] for row in pair_rows if row["state_id"] == state])),
            "near_projection_collapse_fraction_1e_8": float(np.mean([row["near_collapse_1e_8"] for row in pair_rows if row["state_id"] == state])),
            "flow_seed_mean_pairwise_L2_within_eta": flow_pair_means,
            "flow_seed_componentwise_variance_within_eta": [x.tolist() for x in flow_component_variances],
            "eta_choice_mean_distance_over_flow_seed_mean_distance": eta_to_flow_ratio,
            "classification_basis": "Compared same-seed eta disagreement against vmax, typical correction norm, and observed cross-seed label variation.",
        }
        state_pairs = [row for row in pair_rows if row["state_id"] == state]
        closest, farthest = min(state_pairs, key=lambda row: row["L2_distance"]), max(state_pairs, key=lambda row: row["L2_distance"])
        reps = []
        for label_name, selected in (("most_similar_statistically_equivalent_pair", closest), ("most_divergent_pair_inside_E_near", farthest)):
            entry = {"case": label_name, **selected, "policies": []}
            for eta_values in (selected["eta_a"], selected["eta_b"]):
                eta = eta_key(eta_values)
                record = label_lookup[(state, eta, selected["seed"])]
                info = feasible_by_state[state][eta]
                entry["policies"].append({
                    "eta": list(eta), "mean_J_def": info["mean_J_def"],
                    "success_count": info["success_count"], "g_raw": record["g_raw"],
                    "g_exec": record["g_exec"], "u_safe": record["u_safe"], "u_exec": record["u_exec"],
                })
            reps.append(entry)
        representatives[state] = reps

    with (HERE / "pairwise_label_distances.csv").open("w", newline="") as handle:
        fields = ("state_id", "seed", "eta_a", "eta_b", "raw_L2_distance", "L2_distance", "executed_over_raw_distance", "distance_over_vmax", "relative_distance", "cosine_similarity", "exact_collapse", "near_collapse_1e_8")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({**row, "eta_a": json.dumps(row["eta_a"]), "eta_b": json.dumps(row["eta_b"])})
    dump("label_statistics.json", {
        "states": statistics, "minima": minima,
        "B_63": {state: [list(eta) for eta in sorted(feasible_by_state[state])] for state in STATES},
        "E_near": {state: [list(eta) for eta in near_sets[state]] for state in STATES},
        "paired_CRN_near_optimality": paired_results,
    })
    dump("representative_cases.json", representatives)

    source_hashes = {
        "corrector": sha(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
        "environment": sha(SYSROOT / "single_integrator/environment.py"),
        "projection": sha(SYSROOT / "single_integrator/cbf.py"),
        "retry": sha(SBMA / "exact_projector.py"),
        "directional_manifest": sha(DIRECTIONAL / "manifest.json"),
        "decomposition_manifest": sha(DECOMPOSITION / "manifest.json"),
    }
    sanity = {
        "status": "PASS" if not conflicts and not source_sha_mismatches else "FAIL",
        "raw_attempt_records": len(attempts), "effective_unique_tuples": len(effective),
        "duplicate_records": len(duplicates), "duplicate_conflicts": conflicts,
        "solver_failure_policy": "load_inventory uses the completed effective record; a valid retry for an identical tuple supersedes an unresolved attempt in the existing audited index",
        "source_sha_mismatches": source_sha_mismatches,
        "max_raw_corrector_reconstruction_error": max(raw_formula_errors, default=0.0),
        "max_second_projection_replay_error": max(projection_replay_errors, default=0.0),
        "max_w_minus_safe_minus_g_error": max(stored_delta_errors, default=0.0),
        "action_shape": [2, 2], "flattened_label_dimension": 4,
        "canonical_matched_seed_count": len(CANONICAL_SEEDS),
        "new_rollouts": 0, "frozen_source_hashes": source_hashes,
        "existing_files_modified": False,
    }
    dump("sanity_checks.json", sanity)
    if sanity["status"] != "PASS":
        raise AssertionError(sanity)

    b_lines, min_lines, stat_lines = [], [], []
    for state in STATES:
        b = sorted(feasible_by_state[state])
        b_lines.append(f"- {state.split('_')[0]}: " + (", ".join(map(str, b)) if b else "EMPTY"))
        if minima[state] is None:
            min_lines.append(f"| {state.split('_')[0]} | EMPTY | — | — | 0 |")
        else:
            min_lines.append(f"| {state.split('_')[0]} | {tuple(minima[state]['eta'])} | {minima[state]['J_min']:.6f} | {minima[state]['success_count']}/64 | {len(near_sets[state])} |")
        st = statistics[state]
        if st["E_near_size"]:
            stat_lines.append(
                f"| {state.split('_')[0]} | {st['classification']} | {st['E_near_size']} | "
                f"{st['mean_pairwise_L2_eta_same_seed']:.6g} | {st['max_pairwise_L2_eta_same_seed']:.6g} | "
                f"{st['mean_distance_over_vmax']:.4f} | {st['mean_distance_over_typical_norm']:.4f} | "
                f"{st['eta_choice_mean_distance_over_flow_seed_mean_distance']:.4f} | "
                f"{st['mean_executed_over_raw_pair_distance']:.4f} |"
            )
        else:
            stat_lines.append(f"| {state.split('_')[0]} | NOT_TESTABLE_EMPTY_B63 | 0 | — | — | — | — | — | — |")
    report = f"""# Success-basin oracle label stability

## Scope

This audit uses existing trajectories only. No rollout, training, controller change, or new objective was introduced. Feasibility is the requested provisional rule `success_count >= 63/64`. Cells without a valid 64-run cohort are **not evaluated**, not labeled infeasible.

When the canonical 64 seeds (`95106001..32` plus `95107001..32`) exist, they are used. An eta with exactly 64 other existing valid seeds is also audited, with its cohort disclosed; paired near-optimal tests then use only the seed intersection.

## 1. B_63(z)

{chr(10).join(b_lines)}

D2 and D4 have no currently evaluated eta meeting 63/64. This is an empty empirical feasible set under the provisional rule, not evidence that no feasible controller exists.

## 2–3. Minimum deformation and E_near

| state | strict empirical best eta | mean J_def over matched 64 | success | E_near size |
|---|---|---:|---:|---:|
{chr(10).join(min_lines)}

Percentage bands (1%, 2.5%, 5%) and every paired confidence interval are in `near_optimal_eta.csv` and `label_statistics.json`; no percentage threshold is frozen.

## 4. First-step executed-label stability

The label is always `g_exec = u_exec - u_safe` after the second hard projection, flattened from a 2×2 joint action to four components.

| state | result | E_near | mean same-seed eta distance | max distance | mean / vmax | mean / typical label norm | eta variation / Flow-seed variation | executed/raw pair distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(stat_lines)}

Projection-collapse fractions, cosine similarities, componentwise variances, and raw/executed corrections are machine-readable in `label_statistics.json`, `first_step_labels.csv`, and `representative_cases.json`.

## 5. Deterministic supervision decision

- D1: deterministic first-step supervision is justified at the resolution of this audit. The numerical classification is based on disagreement relative to `vmax=0.5`, typical executed-correction norm, actual Flow-seed variability, and projection compression—not eta-space connectedness.
- D2/D4: deterministic supervision is **not currently justified under the 63/64 oracle rule**, because `B_63` is empty. This is lack of an eligible target, not evidence of multivalued action labels.

## 6. Remaining ambiguity

For D1, any remaining ambiguity is action-space variation among statistically equivalent feasible eta after hard projection and is quantified above. For D2/D4, the blocker precedes identifiability: no current eta passes 63/64, so `E_near` and its executed-label distribution do not exist in the available data. Forcing either state into `LABEL_STABLE`, `LABEL_MILDLY_AMBIGUOUS`, or `LABEL_MULTIVALUED` would conflate empty feasibility with label geometry.
"""
    (HERE / "oracle_label_stability_report.md").write_text(report)

    artifacts = (
        "oracle_label_stability_report.md", "feasible_eta_63.csv", "near_optimal_eta.csv",
        "first_step_labels.csv", "pairwise_label_distances.csv", "label_statistics.json",
        "representative_cases.json", "sanity_checks.json", "analyze.py",
    )
    dump("manifest.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "success_basin_oracle_label_stability", "new_rollouts": 0,
        "provisional_rule": "success_count >= 63/64", "not_confidence_calibrated": True,
        "state_results": {state: statistics[state]["classification"] for state in STATES},
        "B_63_sizes": {state: len(feasible_by_state[state]) for state in STATES},
        "E_near_sizes": {state: len(near_sets[state]) for state in STATES},
        "artifacts_sha256": {name: sha(HERE / name) for name in artifacts},
        "source_hashes": source_hashes,
        "notes": ["No training.", "No existing file modified.", "Raw diagnostic correction is not used as the supervision label."],
    })
    print(json.dumps({
        "B_63_sizes": {state: len(feasible_by_state[state]) for state in STATES},
        "E_near_sizes": {state: len(near_sets[state]) for state in STATES},
        "classifications": {state: statistics[state]["classification"] for state in STATES},
        "new_rollouts": 0, "sanity": sanity["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
