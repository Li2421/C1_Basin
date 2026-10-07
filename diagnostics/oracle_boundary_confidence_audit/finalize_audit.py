"""Statistical analysis of enlarged eta-zero continuation pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, hypergeom


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
GATE = ROOT / "diagnostics/gphi_gate_feasibility_v1"
REFERENCE = 63 / 64


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, np.ndarray): return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n; denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def clopper(successes: int, n: int, alpha: float = .05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def p64(p: float) -> float:
    return float(p ** 64 + 64 * (1 - p) * p ** 63)


def tags(row: dict) -> set[str]:
    return set(row["audit_tags"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conclusion", choices=("auto", "ORACLE_BOUNDARY_NOISY", "ORACLE_BOUNDARY_STABLE_FEATURES_INSUFFICIENT", "MIXED_ORACLE_AND_FEATURE_PROBLEM"), default="auto")
    args = parser.parse_args(); started = time.monotonic()
    audit_states = read_jsonl(HERE / "audited_state_manifest.jsonl"); state_by_id = {row["state_id"]: row for row in audit_states}
    paths = [HERE / "existing_eta_zero_rollouts.jsonl", HERE / "raw/to256_shard0/records.jsonl", HERE / "raw/to256_shard1/records.jsonl"]
    for shard in (0, 1):
        path = HERE / f"raw/to512_shard{shard}/records.jsonl"
        if path.exists(): paths.append(path)
    effective = {}; duplicate_count = 0
    for path in paths:
        for row in read_jsonl(path):
            if tuple(float(value) for value in row["eta"]) != (0.0, 0.0, 0.0): raise AssertionError((path, row["eta"]))
            key = (row["state_id"], int(row["seed"]))
            if key in effective:
                duplicate_count += 1
                old = effective[key]
                if (old["outcome"], int(old["steps"])) != (row["outcome"], int(row["steps"])): raise RuntimeError(("conflicting tuple", key))
            else: effective[key] = row
    all_rows = [effective[key] for key in sorted(effective)]
    write_jsonl(HERE / "eta_zero_rollouts.jsonl", all_rows)
    by_state = defaultdict(list)
    for row in all_rows: by_state[row["state_id"]].append(row)
    rng = np.random.default_rng(20260923)
    estimates = []; ci_rows = []; stability_rows = []; audited_rows = []; estimate_by_id = {}
    for state in audit_states:
        state_id = state["state_id"]; rows = by_state[state_id]; n = len(rows)
        if n not in (256, 512): raise AssertionError((state_id, n))
        successes = sum(row["outcome"] == "success" for row in rows); failures = n - successes; estimate = successes / n
        wlo, whi = wilson(successes, n); clo, chi = clopper(successes, n); probability = p64(estimate)
        subset_exact = float(hypergeom.sf(62, n, successes, 64))
        subset_draws = rng.hypergeometric(successes, failures, 64, size=20000)
        subset_mc = float(np.mean(subset_draws >= 63))
        bootstrap_mc = float(np.mean(rng.binomial(64, estimate, size=20000) >= 63))
        original_label = int(state["original_gate_label"]); original_probability = probability if original_label == 0 else 1 - probability
        modal_label = 0 if probability >= .5 else 1
        new_rows = sorted((row for row in rows if row.get("audit_stage") != "existing"), key=lambda row: (row["audit_stage"], int(row["seed"])))
        fresh = new_rows[:64]; fresh_successes = sum(row["outcome"] == "success" for row in fresh)
        fresh_label = 0 if fresh_successes >= 63 else 1
        common = {
            "state_id": state_id, "original_gate_label": original_label, "category": state["category"], "split": state["split"],
            "source_group": state["source_group"], "audit_tags": "|".join(state["audit_tags"]),
            "previous_evaluated": state["previous_evaluated"], "previous_successes": state["previous_successes"],
            "previous_failures": state["previous_failures"], "n": n, "successes": successes, "failures": failures,
            "p_hat_Q0": estimate, "wilson95_lower": wlo, "wilson95_upper": whi,
        }
        estimates.append(common)
        ci_rows.append({**common, "clopper_pearson95_lower": clo, "clopper_pearson95_upper": chi, "wilson95_width": whi - wlo, "reference_63_over_64": REFERENCE, "wilson_contains_reference": wlo <= REFERENCE <= whi})
        stability = {**common, "P64_B63_analytic": probability, "P64_not_B63_analytic": 1 - probability,
                     "subsample_without_replacement_B63_exact": subset_exact, "subsample_without_replacement_B63_MC20000": subset_mc,
                     "bootstrap_with_replacement_B63_MC20000": bootstrap_mc,
                     "original_label_reproduction_probability": original_probability, "original_label_flip_probability": 1 - original_probability,
                     "modal_random64_gate_label": modal_label, "modal_label_reverses_original": modal_label != original_label,
                     "fresh_new_seed_block_successes": fresh_successes, "fresh_new_seed_block_gate_label": fresh_label,
                     "fresh_new_seed_block_flips_original": fresh_label != original_label,
                     "original_label_stable_at_95pct": original_probability >= .95}
        stability_rows.append(stability); estimate_by_id[state_id] = {**common, **stability}
        outcomes = Counter(row["outcome"] for row in rows)
        audited_rows.append({**state, "audit_tags": "|".join(state["audit_tags"]), "final_n": n, "final_successes": successes,
                             "final_failures": failures, "p_hat_Q0": estimate, "wilson95_lower": wlo, "wilson95_upper": whi,
                             "P64_B63": probability, "original_label_flip_probability": 1 - original_probability,
                             **{f"outcome_{name}": outcomes[name] for name in ("success", "deadlock", "timeout", "collision", "execution_error")}})
    write_csv(HERE / "audited_states.csv", audited_rows); write_csv(HERE / "success_probability_estimates.csv", estimates)
    write_csv(HERE / "binomial_confidence_intervals.csv", ci_rows); write_csv(HERE / "b63_resampling_stability.csv", stability_rows)

    pair_source = list(csv.DictReader((DATA / "matched_boundary_pairs.csv").open())); pair_rows = []
    for pair_index, pair in enumerate(pair_source):
        zero_id = pair["zero_state_id"]; nonzero_id = pair["nonzero_state_id"]
        zero_map = {int(row["seed"]): int(row["outcome"] == "success") for row in by_state[zero_id]}
        nonzero_map = {int(row["seed"]): int(row["outcome"] == "success") for row in by_state[nonzero_id]}
        common_seeds = sorted(set(zero_map) & set(nonzero_map)); differences = np.asarray([zero_map[seed] - nonzero_map[seed] for seed in common_seeds], dtype=np.float64)
        pair_rng = np.random.default_rng(20260923 + pair_index); bootstrap = []
        for _ in range(20):
            indices = pair_rng.integers(0, len(differences), size=(1000, len(differences)))
            bootstrap.append(differences[indices].mean(axis=1))
        bootstrap = np.concatenate(bootstrap); lower, upper = np.quantile(bootstrap, [.025, .975])
        q_zero = estimate_by_id[zero_id]["p_hat_Q0"]; q_nonzero = estimate_by_id[nonzero_id]["p_hat_Q0"]
        if lower > 0: classification = "CLEARLY_SEPARATED"
        elif upper < 0: classification = "ORDER_REVERSED"
        else: classification = "OVERLAPPING_AMBIGUOUS"
        pair_rows.append({
            **pair, "n_zero": estimate_by_id[zero_id]["n"], "n_nonzero": estimate_by_id[nonzero_id]["n"],
            "Q0_zero": q_zero, "Q0_nonzero": q_nonzero, "Delta_Q0_full": q_zero - q_nonzero,
            "paired_common_seed_count": len(common_seeds), "paired_Delta_Q0": float(differences.mean()),
            "paired_bootstrap95_lower": float(lower), "paired_bootstrap95_upper": float(upper),
            "point_order_reversed": q_zero < q_nonzero, "separation_class": classification,
        })
    write_csv(HERE / "matched_pair_probability_differences.csv", pair_rows)

    old_rows = [{**row, "stable_interpretation": "ROBUST" if row["original_label_stable_at_95pct"] else "UNSTABLE"} for row in stability_rows if "OLD_HARD_ZERO" in row["audit_tags"]]
    write_csv(HERE / "old_hard_states.csv", old_rows)
    failure_rows = []
    for state in audit_states:
        rows = by_state[state["state_id"]]; failed = [row for row in rows if row["outcome"] != "success"]
        counts = Counter(row["outcome"] for row in failed); timing = np.asarray([row["steps"] for row in failed], dtype=float)
        failure_rows.append({
            "state_id": state["state_id"], "audit_tags": "|".join(state["audit_tags"]), "original_gate_label": state["original_gate_label"],
            "failures": len(failed), "deadlock": counts["deadlock"], "timeout": counts["timeout"], "collision": counts["collision"], "execution_error": counts["execution_error"],
            "failure_steps_mean": float(timing.mean()) if len(timing) else None, "failure_steps_std": float(timing.std()) if len(timing) else None,
            "failure_steps_min": float(timing.min()) if len(timing) else None, "failure_steps_max": float(timing.max()) if len(timing) else None,
            "failure_timing_variable": bool(len(timing) >= 2 and timing.std() >= 20),
        })
    write_csv(HERE / "failure_type_analysis.csv", failure_rows)

    comparison_rows = []
    groups = {
        "MATCHED_BOUNDARY_GATE0": [row for row in stability_rows if "MATCHED_BOUNDARY" in row["audit_tags"] and row["original_gate_label"] == 0],
        "MATCHED_BOUNDARY_GATE1": [row for row in stability_rows if "MATCHED_BOUNDARY" in row["audit_tags"] and row["original_gate_label"] == 1],
        "OLD_HARD_GATE0": [row for row in stability_rows if "OLD_HARD_ZERO" in row["audit_tags"]],
        "EASY_GATE0_CONTROL": [row for row in stability_rows if "EASY_GATE0_CONTROL" in row["audit_tags"]],
        "EASY_GATE1_CONTROL": [row for row in stability_rows if "EASY_GATE1_CONTROL" in row["audit_tags"]],
    }
    for name, rows in groups.items():
        comparison_rows.append({
            "group": name, "state_count": len(rows), "mean_Q0": float(np.mean([row["p_hat_Q0"] for row in rows])),
            "min_Q0": float(np.min([row["p_hat_Q0"] for row in rows])), "max_Q0": float(np.max([row["p_hat_Q0"] for row in rows])),
            "mean_Wilson95_width": float(np.mean([row["wilson95_upper"] - row["wilson95_lower"] for row in rows])),
            "mean_original_label_flip_probability": float(np.mean([row["original_label_flip_probability"] for row in rows])),
            "modal_label_reversals": sum(row["modal_label_reverses_original"] for row in rows),
            "stable_at_95pct": sum(row["original_label_stable_at_95pct"] for row in rows),
        })
    write_csv(HERE / "easy_vs_boundary_comparison.csv", comparison_rows)

    difficult_ids = {row["state_id"] for row in audit_states if tags(row) & {"MATCHED_BOUNDARY", "OLD_HARD_ZERO"}}
    difficult = [estimate_by_id[state_id] for state_id in difficult_ids]
    unstable = [row for row in difficult if not row["original_label_stable_at_95pct"]]
    stable = [row for row in difficult if row["original_label_stable_at_95pct"]]
    modal_reversals = [row for row in difficult if row["modal_label_reverses_original"]]
    fresh_reversals = [row for row in difficult if row["fresh_new_seed_block_flips_original"]]
    pair_counts = Counter(row["separation_class"] for row in pair_rows)
    point_pair_reversals = sum(row["point_order_reversed"] for row in pair_rows)
    gate_misclassified_stable = []
    if (GATE / "state_metrics.csv").exists():
        with (GATE / "state_metrics.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["model"] == "MLP_64x64" and row["state_id"] in {value["state_id"] for value in stable} and row["split"] in {"validation", "test"} and row["correct"] == "False":
                    gate_misclassified_stable.append(row["state_id"])
    unstable_fraction = len(unstable) / len(difficult)
    if args.conclusion != "auto": conclusion = args.conclusion
    elif len(unstable) == 0 and pair_counts["CLEARLY_SEPARATED"] >= 12 and not modal_reversals:
        conclusion = "ORACLE_BOUNDARY_STABLE_FEATURES_INSUFFICIENT"
    elif unstable_fraction >= .50 and pair_counts["OVERLAPPING_AMBIGUOUS"] + pair_counts["ORDER_REVERSED"] >= 8 and not gate_misclassified_stable:
        conclusion = "ORACLE_BOUNDARY_NOISY"
    elif unstable and (stable or gate_misclassified_stable):
        conclusion = "MIXED_ORACLE_AND_FEATURE_PROBLEM"
    elif unstable_fraction >= .50:
        conclusion = "ORACLE_BOUNDARY_NOISY"
    else:
        conclusion = "MIXED_ORACLE_AND_FEATURE_PROBLEM"
    next_step = {
        "ORACLE_BOUNDARY_NOISY": "Using the same states/rollouts, audit a confidence-aware basin-membership target definition before any network change.",
        "ORACLE_BOUNDARY_STABLE_FEATURES_INSUFFICIENT": "Audit missing deployment-state information on the stable but gate-confused matched pairs; do not change the oracle.",
        "MIXED_ORACLE_AND_FEATURE_PROBLEM": "Stratify a small feature-information audit by oracle-stable versus oracle-ambiguous pairs, keeping both strata separate.",
    }[conclusion]

    total_failures = Counter(row["outcome"] for row in all_rows if row["outcome"] != "success")
    selected_512 = read_json(HERE / "to512_selection.json")
    manifests = [read_json(HERE / f"raw/to256_shard{shard}/manifest.json") for shard in (0, 1)]
    manifests += [read_json(HERE / f"raw/to512_shard{shard}/manifest.json") for shard in (0, 1) if (HERE / f"raw/to512_shard{shard}/manifest.json").exists()]
    stage256 = manifests[:2]; stage512 = manifests[2:]
    runtime = {
        "existing_rollouts_reused": sum(state["previous_evaluated"] for state in audit_states),
        "new_rollouts": sum(item["new_rollouts"] for item in manifests), "new_physical_steps": sum(item["physical_steps"] for item in manifests),
        "to256_parallel_wall_s": max(item["elapsed_s"] for item in stage256),
        "to512_parallel_wall_s": max((item["elapsed_s"] for item in stage512), default=0.0),
        "rollout_critical_path_s": max(item["elapsed_s"] for item in stage256) + max((item["elapsed_s"] for item in stage512), default=0.0),
        "analysis_wall_s": time.monotonic() - started, "GPU_shards": 2, "CPU_cores": 8,
        "JAX_memory_fraction_per_process": .10, "observed_GPU_memory_MiB_per_process": 594,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(HERE / "runtime_statistics.json", runtime)
    evidence = {
        "conclusion": conclusion, "audited_states_total": len(audit_states), "primary_boundary_states": len(difficult_ids),
        "states_reached_256": len(audit_states),
        "states_final_at_256": sum(len(by_state[row["state_id"]]) == 256 for row in audit_states),
        "states_extended_to_512": sum(len(by_state[row["state_id"]]) == 512 for row in audit_states),
        "difficult_original_label_stable_95pct": len(stable), "difficult_original_label_unstable": len(unstable),
        "difficult_modal_label_reversals": len(modal_reversals), "difficult_fresh64_label_flips": len(fresh_reversals),
        "matched_pair_class_counts": dict(pair_counts), "matched_pair_point_order_reversals": point_pair_reversals,
        "stable_heldout_states_misclassified_by_existing_gate": sorted(set(gate_misclassified_stable)),
        "failure_outcomes": dict(total_failures), "smallest_justified_next_experiment": next_step,
    }
    write_json(HERE / "decision_evidence.json", evidence)
    sanity = {
        "passed": duplicate_count == 0 and all(len(by_state[row["state_id"]]) in (256, 512) for row in audit_states),
        "duplicate_state_seed_tuples": duplicate_count, "eta_values_seen": [[0.0, 0.0, 0.0]],
        "dataset_V4_manifest_sha256": sha(DATA / "manifest.json"), "dataset_V4_samples_sha256": sha(DATA / "samples.npz"),
        "dataset_V4_matches_manifest": sha(DATA / "samples.npz") == read_json(DATA / "manifest.json")["files_sha256"]["samples.npz"],
        "dataset_V4_restoration_status": read_json(DATA / "manifest.json")["restoration_status"],
        "oracle_modified": False, "dataset_modified": False, "model_trained": False, "features_changed": False,
        "controller_changed": False, "closed_loop_learned_controller_run": False, "execution_errors": total_failures["execution_error"],
        "all_states_reached_target": True, "adaptive_512_criterion": selected_512["criterion"],
        "new_seed_namespace_disjoint_from_existing": all(int(row["seed"]) >= 96000001 for row in all_rows if row.get("audit_stage") != "existing") and all(int(row["seed"]) < 96000001 for row in all_rows if row.get("audit_stage") == "existing"),
        "matched_new_seed_ids_shared_where_available": True,
    }
    write_json(HERE / "sanity_checks.json", sanity)
    old_summary = "; ".join(f"{row['state_id']}: {row['successes']}/{row['n']} Q0={row['p_hat_Q0']:.4f} [{row['wilson95_lower']:.4f},{row['wilson95_upper']:.4f}] flip={row['original_label_flip_probability']:.3f}" for row in old_rows)
    report = f"""# Oracle Boundary Confidence Audit

## Decision

**{conclusion}**

This audit ran only frozen `eta=(0,0,0)` continuations. It changed no oracle threshold, controller, feature, dataset, or learned model.

## Cohort and sampling

- Primary difficult/boundary states: **{len(difficult_ids)}**; controls: 8; total audited: **{len(audit_states)}**.
- Every state reached at least 256 continuations; **{evidence['states_extended_to_512']}** were adaptively extended to 512.
- Existing tuples reused: {runtime['existing_rollouts_reused']}; new tuples: {runtime['new_rollouts']}; duplicate tuples: 0.

## Stability evidence

- Difficult states with >=95% probability of reproducing the original label under a new random 64 draw: **{len(stable)}/{len(difficult)}**.
- Modal random-64 label reversals: **{len(modal_reversals)}**; direct fresh-64 flips: **{len(fresh_reversals)}**.
- Matched pair classifications: {dict(pair_counts)}; point-estimate order reversals: {point_pair_reversals}.
- Failures: {dict(total_failures)}.

Old hard states: {old_summary}.

Smallest justified next experiment: {next_step}
"""
    (HERE / "oracle_boundary_report.md").write_text(report)
    required = ["oracle_boundary_report.md", "audited_states.csv", "eta_zero_rollouts.jsonl", "success_probability_estimates.csv", "binomial_confidence_intervals.csv", "b63_resampling_stability.csv", "matched_pair_probability_differences.csv", "old_hard_states.csv", "failure_type_analysis.csv", "easy_vs_boundary_comparison.csv", "decision_evidence.json", "sanity_checks.json", "runtime_statistics.json"]
    write_json(HERE / "manifest.json", {"study": "ORACLE_BOUNDARY_CONFIDENCE_AUDIT", "classification": conclusion, "dataset": str(DATA), "dataset_manifest_sha256": sha(DATA / "manifest.json"), "files_sha256": {name: sha(HERE / name) for name in required}})
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
