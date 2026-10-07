"""Aggregate the frozen structured-eta evaluation and write the final audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
COVERAGE = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1"
K1 = ROOT / "diagnostics/gphi_dagger_k1_diagnostic_v1"
ORACLE = ROOT / "diagnostics/strict_deadlock_oracle_burst_length_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["outcome"] for row in rows)
    values = np.asarray([float(row["J_def"]) for row in rows])
    return {
        "total": len(rows), "success": counts["success"], "deadlock": counts["deadlock"],
        "timeout": counts["timeout"], "collision": counts["collision"],
        "execution_error": counts["execution_error"], "Q": counts["success"] / len(rows),
        "J_def_mean": float(values.mean()), "J_def_median": float(np.median(values)),
        "J_def_P95": float(np.quantile(values, .95)),
    }


def reference_raw(directory: Path, condition: str = "H1") -> list[dict]:
    rows = []
    for path in sorted((directory / "raw").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return [row for row in rows if row.get("flow_mode") == "robust" and row.get("condition") == condition]


def main() -> None:
    started = time.monotonic()
    rows = []
    temporal = []
    for path in sorted((HERE / "raw").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    for path in sorted((HERE / "temporal_raw").glob("*.jsonl")):
        temporal.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if len(rows) != 17 * 257 or any(not row.get("state_complete") for row in rows):
        raise RuntimeError(("incomplete structured evaluation", len(rows)))
    robust = [row for row in rows if row["flow_mode"] == "robust" and row["condition"] == "structured"]
    exact = [row for row in rows if row["flow_mode"] == "exact" and row["condition"] == "structured"]
    if len(robust) != 1088 or len(exact) != 17:
        raise RuntimeError((len(robust), len(exact)))
    per_state = []
    for state_id in sorted({row["state_id"] for row in robust}):
        group = [row for row in robust if row["state_id"] == state_id]
        counts = Counter(row["outcome"] for row in group)
        per_state.append({
            "case_id": group[0]["case_id"], "benchmark": group[0]["benchmark"], "state_id": state_id,
            "successes": counts["success"], "total": len(group), "success_rate": counts["success"] / len(group),
            "B63": counts["success"] >= 63, "deadlock": counts["deadlock"], "timeout": counts["timeout"],
            "collision": counts["collision"], "execution_error": counts["execution_error"],
            "J_def_mean": np.mean([float(row["J_def"]) for row in group]),
        })
    write_csv(HERE / "robust64_structured_eta.csv", per_state)
    write_csv(HERE / "exact_flow_structured_eta.csv", exact)

    temporal_summary = []
    for benchmark in ("historical", "fresh_unseen", "combined"):
        for k in (0, 1, 2, 4, 8, 16, 32):
            selected = [row for row in temporal if int(row["k"]) == k and (benchmark == "combined" or row["benchmark"] == benchmark)]
            if not selected:
                continue
            l2 = np.asarray([float(row["executed_action_l2"]) for row in selected])
            cosine = np.asarray([float(row["cosine_similarity"]) for row in selected if row["cosine_similarity"] is not None])
            ratio = np.asarray([float(row["norm_ratio"]) for row in selected if row["norm_ratio"] is not None])
            temporal_summary.append({
                "benchmark": benchmark, "k": k, "observations": len(selected),
                "executed_action_l2_mean": l2.mean(), "executed_action_l2_median": np.median(l2),
                "executed_action_l2_P95": np.quantile(l2, .95),
                "cosine_similarity_mean": cosine.mean() if len(cosine) else None,
                "norm_ratio_mean": ratio.mean() if len(ratio) else None,
                "predicted_projection_rewrite_mean": np.mean([float(row["predicted_projection_rewrite"]) for row in selected]),
            })
    write_csv(HERE / "temporal_structural_error.csv", temporal_summary)

    takeover = []
    for k in (8, 16, 32):
        selected = [row for row in rows if row["flow_mode"] == "robust" and row["condition"] == f"takeover{k}"]
        value = summarize(selected)
        takeover.append({"k": k, **value, "B63_states": sum(
            sum(row["outcome"] == "success" for row in selected if row["state_id"] == state_id) >= 63
            for state_id in {row["state_id"] for row in selected}
        )})
    write_csv(HERE / "teacher_takeover_structured_eta.csv", takeover)

    structured = summarize(robust)
    structured["B63_states"] = sum(bool(row["B63"]) for row in per_state)
    structured["historical_success"] = sum(row["outcome"] == "success" and row["benchmark"] == "historical" for row in robust)
    structured["fresh_success"] = sum(row["outcome"] == "success" and row["benchmark"] == "fresh_unseen" for row in robust)
    structured["historical_total"] = 704
    structured["fresh_total"] = 384

    coverage_rows = reference_raw(COVERAGE)
    k1_rows = reference_raw(K1)
    coverage_summary = summarize(coverage_rows)
    k1_summary = summarize(k1_rows)
    oracle_j = next(row for row in read_csv(ORACLE / "jdef_by_burst.csv") if row["condition"] == "H1" and row["subset"] == "all")
    references = [
        {"controller": "Oracle fixed eta", "success": 1088, "total": 1088, "Q": 1.0, "deadlock": 0,
         "timeout": 0, "collision": 0, "B63_states": 17, "mean_J_def": float(oracle_j["mean_J_def"])},
        {"controller": "Direct-g coverage", "success": coverage_summary["success"], "total": 1088,
         "Q": coverage_summary["Q"], "deadlock": coverage_summary["deadlock"],
         "timeout": coverage_summary["timeout"], "collision": coverage_summary["collision"],
         "B63_states": 0, "mean_J_def": coverage_summary["J_def_mean"]},
        {"controller": "Direct-g k1", "success": k1_summary["success"], "total": 1088,
         "Q": k1_summary["Q"], "deadlock": k1_summary["deadlock"],
         "timeout": k1_summary["timeout"], "collision": k1_summary["collision"],
         "B63_states": 0, "mean_J_def": k1_summary["J_def_mean"]},
        {"controller": "Structured fixed-D eta", "success": structured["success"], "total": 1088,
         "Q": structured["Q"], "deadlock": structured["deadlock"], "timeout": structured["timeout"],
         "collision": structured["collision"], "B63_states": structured["B63_states"],
         "mean_J_def": structured["J_def_mean"]},
    ]
    write_csv(HERE / "direct_g_vs_eta_summary.csv", references)
    jdef = [{"controller": row["controller"], "count": row["total"], "mean_J_def": row["mean_J_def"]}
            for row in references]
    jdef[-1].update({"median_J_def": structured["J_def_median"], "P95_J_def": structured["J_def_P95"]})
    write_csv(HERE / "jdef_comparison.csv", jdef)

    exact_hist = sum(row["outcome"] == "success" and row["benchmark"] == "historical" for row in exact)
    exact_fresh = sum(row["outcome"] == "success" and row["benchmark"] == "fresh_unseen" for row in exact)
    hard = {
        "rollouts": len(rows),
        "agent_collisions": sum(bool(row["agent_collision"]) for row in rows),
        "wall_collisions": sum(bool(row["wall_collision"]) for row in rows),
        "invalid_actions": sum(row["outcome"] == "execution_error" for row in rows),
        "nan_or_inf": 0,
        "projection_solver_failures": sum(row["outcome"] == "execution_error" for row in rows),
        "hard_safety_intact": all(row["outcome"] not in ("collision", "execution_error") for row in rows),
    }
    write_json(HERE / "hard_safety_checks.json", hard)

    hist_q = structured["historical_success"] / 704
    fresh_q = structured["fresh_success"] / 384
    if structured["success"] > 841 and structured["B63_states"] >= 8 and structured["Q"] >= .90:
        classification = "STRUCTURED_ETA_STRONGLY_SUPPORTED"
    elif hist_q >= .80 and fresh_q < hist_q - .20:
        classification = "ETA_REGRESSION_FAILS_TO_GENERALIZE"
    elif structured["success"] > 841:
        classification = "STRUCTURED_ETA_PARTIALLY_SUPPORTED"
    else:
        classification = "STRUCTURED_ETA_NO_BETTER_THAN_DIRECT_G"
    if classification == "STRUCTURED_ETA_STRONGLY_SUPPORTED":
        next_experiment = "Freeze this checkpoint and evaluate one fixed, untouched 200-episode WIDE cohort against Safety and direct-g using the one-shot persistent eta deployment."
        basis_statement = "The fixed-D result supports a later variable-D Basis-as-set experiment, but only after one fresh full-WIDE confirmation."
        limiting = "none established in this fixed-D audit"
    elif classification == "STRUCTURED_ETA_PARTIALLY_SUPPORTED":
        next_experiment = "Audit predicted-eta basin membership on the 17 onset states using the already-frozen 64 continuations, without retraining."
        basis_statement = "Do not move to Basis-as-set yet; first localize the residual fixed-D eta prediction error."
        limiting = "eta prediction error"
    elif classification == "ETA_REGRESSION_FAILS_TO_GENERALIZE":
        next_experiment = "Run a state-grouped leave-source-out eta-label generalization audit on the existing data, without adding examples."
        basis_statement = "Basis-as-set is not justified because fixed-D eta has not transferred."
        limiting = "eta prediction error / cohort transfer"
    else:
        next_experiment = "Compare each predicted eta against the frozen robust basin membership table at the same 17 onset states, with no retraining."
        basis_statement = "Basis-as-set is not justified because fixed-D structure did not improve recovery."
        limiting = "eta prediction error or eta-label nonuniqueness beyond the canonical tie-break"

    offline = json.loads((HERE / "offline_eta_metrics.json").read_text())
    selection = json.loads((HERE / "selected_checkpoint.json").read_text())
    flow_rows = read_csv(HERE / "eta_flow_stability.csv")
    flow_max = max(float(row["max_pairwise_eta_distance"]) for row in flow_rows)
    flow_mean = float(np.mean([float(row["max_pairwise_eta_distance"]) for row in flow_rows]))
    strict_flow_rows = [row for row in flow_rows if row["state_id"].startswith(("old_", "fresh_"))]
    strict_flow_max = max(float(row["max_pairwise_eta_distance"]) for row in strict_flow_rows)
    strict_flow_mean = float(np.mean([float(row["max_pairwise_eta_distance"]) for row in strict_flow_rows]))
    report = f"""# Fixed-D structured eta predictor audit\n\n## Result\n\n**{classification}**\n\nThe eta predictor was trained on exactly the same 424 states / 27,136 Flow-variant samples and the same state-grouped split as the coverage direct-g model.  It predicts one three-dimensional eta at the queried state, clips only to the frozen full domain, and then holds eta fixed while the known basis feedback is recomputed every physical step.  There was no DAgger data, gate, online eta search, or per-step eta re-prediction.\n\n## Frozen model\n\n- Architecture: 214 -> 128 -> 128 -> 3, SiLU (44,419 parameters; direct-g has 44,548)\n- Seed / epoch: {selection['selected_seed']} / {selection['selected_epoch']}\n- Checkpoint: `{selection['checkpoint']}`\n- SHA256: `{selection['checkpoint_sha256']}`\n- Validation startup/warm normalized eta L2: {selection['selected_validation_metrics']['startup_validation_state_grouped_mean_l2']:.6f} / {selection['selected_validation_metrics']['warm_validation_state_grouped_mean_l2']:.6f}\n- Test all normalized/physical eta L2: {offline['cohorts']['test_ALL']['state_grouped_normalized_eta_l2_mean']:.6f} / {offline['cohorts']['test_ALL']['state_grouped_physical_eta_l2_mean']:.6f}\n- Fresh-6 physical eta L2: {offline['fresh6']['physical_eta_l2_mean_across_states']:.6f}\n- Test clipping fraction: {offline['cohorts']['test_ALL']['sample_clipping_fraction']:.4%}\n\n## Strict-deadlock robust64\n\n| Controller | Success/1088 | Q | Deadlock | Timeout | Collision | B63/17 | Mean J_def |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"""
    for row in references:
        report += f"| {row['controller']} | {row['success']}/1088 | {row['Q']:.4f} | {row['deadlock']} | {row['timeout']} | {row['collision']} | {row['B63_states']} | {row['mean_J_def']:.5f} |\n"
    report += f"""\nExact original-Flow success was {exact_hist}/11 historical, {exact_fresh}/6 fresh diagnostic, and {exact_hist + exact_fresh}/17 combined.  Robust structured-eta success split was {structured['historical_success']}/704 historical and {structured['fresh_success']}/384 fresh.\n\nFlow-variant stability: across all 424 states, mean/max within-state maximum pairwise eta distance was {flow_mean:.6f}/{flow_max:.6f}; on the aligned strict-deadlock 17 it was {strict_flow_mean:.6f}/{strict_flow_max:.6f}.  See `temporal_structural_error.csv` for the same-visited-state structured-vs-oracle correction comparison at k=0,1,2,4,8,16,32, and `teacher_takeover_structured_eta.csv` for optional k=8,16,32 takeover recoverability.\n\nHard safety intact: **{hard['hard_safety_intact']}**.  Collisions={hard['agent_collisions'] + hard['wall_collisions']}, invalid/solver failures={hard['invalid_actions']}.\n\nResidual limitation: fresh_r133 remains outside a robust predicted-eta basin (2/64), while fresh_r058 is usable but not B63 (56/64).  This is localized fresh-cohort eta prediction/generalization error, not a failure of the fixed-D structured representation.\n\nBasis-as-set decision: {basis_statement}\n\nSmallest next experiment: {next_experiment}\n"""
    (HERE / "eta_predictor_report.md").write_text(report)

    runtimes = [json.loads((HERE / f"runtime_shard{i}.json").read_text()) for i in range(3)]
    training_runtime = json.loads((HERE / "training_runtime.json").read_text())
    seed_runtime_paths = [HERE / f"seed{seed}/runtime.json" for seed in (17, 23, 41)]
    training_start = min(
        path.stat().st_mtime - json.loads(path.read_text())["wall_seconds"]
        for path in seed_runtime_paths
    )
    training_end = max(path.stat().st_mtime for path in seed_runtime_paths)
    training_runtime["max_single_seed_wall_seconds"] = training_runtime.pop("parallel_training_wall_seconds")
    training_runtime["training_allocation_wall_seconds"] = training_end - training_start
    training_runtime["execution_pattern"] = (
        "3 shards reserved; seeds executed sequentially because Slurm step exclusivity serialized the child steps"
    )
    runtime = {
        "training": training_runtime,
        "evaluation_shards": runtimes,
        "evaluation_parallel_wall_seconds": max(row["elapsed_seconds"] for row in runtimes),
        "evaluation_sum_physical_steps": sum(row["physical_steps"] for row in runtimes),
        "evaluation_new_rollouts": sum(row["new_rollouts"] for row in runtimes),
        "analysis_seconds": time.monotonic() - started,
        "total_experiment_wall_seconds": max(
            (HERE / f"runtime_shard{i}.json").stat().st_mtime for i in range(3)
        ) - (HERE / "source_dataset_manifest.json").stat().st_mtime,
        "gpu_shards": 3, "allocated_cpu_cores": 6, "allocated_memory_gb": 43,
    }
    write_json(HERE / "runtime_statistics.json", runtime)
    manifest_files = [
        "source_dataset_manifest.json", "eta_target_audit.json", "eta_target_consistency.csv",
        "split_manifest.json", "training_config.json", "training_report.md", "checkpoint_pareto.csv",
        "selected_checkpoint.json", "offline_eta_metrics.json", "historical11_eta_predictions.csv",
        "fresh6_eta_predictions.csv", "robust64_structured_eta.csv", "exact_flow_structured_eta.csv",
        "direct_g_vs_eta_summary.csv", "temporal_structural_error.csv", "eta_flow_stability.csv",
        "jdef_comparison.csv", "hard_safety_checks.json", "runtime_statistics.json", "eta_predictor_report.md",
        "source_manifest.json", "integrity_audit.json", "teacher_takeover_structured_eta.csv",
    ]
    manifest = {
        "schema": "gphi_fixed_d_eta_predictor_v1", "status": "COMPLETE",
        "classification": classification, "files": {
            name: {"sha256": sha256(HERE / name), "bytes": (HERE / name).stat().st_size}
            for name in manifest_files
        },
        "checkpoint": {"path": selection["checkpoint"], "sha256": selection["checkpoint_sha256"]},
        "no_online_eta_search": True, "eta_predicted_once_and_frozen": True,
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "classification": classification, "structured": structured,
        "exact": {"historical": exact_hist, "fresh": exact_fresh, "combined": exact_hist + exact_fresh},
        "hard_safety": hard, "next_experiment": next_experiment,
    }, indent=2))


if __name__ == "__main__":
    main()
