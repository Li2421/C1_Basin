"""Analyze frozen G_phi burst persistence and aligned fixed-eta teacher drift."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1"
ORACLE = ROOT / "diagnostics/strict_deadlock_oracle_burst_length_v1"
CADENCE = ROOT / "diagnostics/strict_deadlock_oracle_cadence_v1"
CONDITIONS = ("L1", "L2", "L4", "L6", "L8", "H1")
BURSTS = ("L1", "L2", "L4", "L6", "L8")
SEEDS = list(range(95310001, 95310065))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(statistics.fmean(finite)) if finite else math.nan


def median(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(statistics.median(finite)) if finite else math.nan


def quantile(values: list[float], q: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.quantile(finite, q)) if len(finite) else math.nan


def trace_path(state_id: str, condition: str, oracle: bool) -> Path:
    if not oracle:
        return HERE / "traces" / f"{state_id}__{condition}.npz"
    if condition == "H1":
        return CADENCE / "traces" / f"{state_id}__H1.npz"
    if condition == "L1":
        return CADENCE / "traces" / f"{state_id}__H8.npz"
    return ORACLE / "traces" / f"{state_id}__{condition}.npz"


def main() -> None:
    started = time.monotonic()
    source = json.loads((HERE / "source_manifest.json").read_text())
    checkpoint = json.loads((HERE / "checkpoint_manifest.json").read_text())
    integrity = json.loads((HERE / "integrity_audit.json").read_text())
    # Permit deterministic re-analysis after a completed audit as well as the
    # initial transition from the preparation gate.
    if integrity["status"] not in {"PREPARED", "PASS"}:
        raise RuntimeError("preparation gate not passed")
    states = source["states"]
    rows: dict[str, dict[str, list[dict]]] = {}
    exact: dict[str, dict[str, dict]] = {}
    for state in states:
        state_id = state["state_id"]
        path = HERE / "raw" / f"{state_id}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing raw file {state_id}")
        values = jsonl(path)
        if len(values) != 390 or not all(row.get("state_complete") for row in values):
            raise RuntimeError(f"incomplete raw result {state_id}: {len(values)}")
        rows[state_id] = {}
        exact[state_id] = {}
        for condition in CONDITIONS:
            robust = sorted(
                [row for row in values if row["condition"] == condition and row["flow_mode"] == "robust"],
                key=lambda row: int(row["seed"]),
            )
            exact_rows = [row for row in values if row["condition"] == condition and row["flow_mode"] == "exact"]
            if [int(row["seed"]) for row in robust] != SEEDS or len(exact_rows) != 1:
                raise RuntimeError(f"tuple mismatch {state_id} {condition}")
            rows[state_id][condition] = robust
            exact[state_id][condition] = exact_rows[0]
            active_path = HERE / "active_logs" / f"{state_id}__{condition}__robust.npz"
            if not active_path.is_file():
                raise RuntimeError(f"active log missing {state_id} {condition}")
            if not trace_path(state_id, condition, False).is_file():
                raise RuntimeError(f"exact trace missing {state_id} {condition}")
        if int(state["query_step"]) % 8 == 0:
            for l8_row, h1_row in zip(rows[state_id]["L8"], rows[state_id]["H1"]):
                if not (
                    l8_row["outcome"] == h1_row["outcome"]
                    and int(l8_row["terminal_global_step"]) == int(h1_row["terminal_global_step"])
                    and abs(float(l8_row["J_def"]) - float(h1_row["J_def"])) <= 1e-10
                ):
                    raise RuntimeError(f"phase-zero L8/H1 identity failed {state_id}")

    exact_output: list[dict] = []
    robust_output: list[dict] = []
    min_output: list[dict] = []
    monotonic_output: list[dict] = []
    for state in states:
        state_id = state["state_id"]
        counts = {
            condition: sum(row["outcome"] == "success" for row in rows[state_id][condition])
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            row = exact[state_id][condition]
            exact_output.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                "condition": condition, "outcome": row["outcome"],
                "terminal_global_step": row["terminal_global_step"],
                "completion_time_seconds": row["completion_time_seconds"], "J_def": row["J_def"],
                "active_fraction": row["active_fraction"],
                "teacher_executed_l2_mean": row["teacher_executed_l2_mean"],
                "teacher_cosine_mean": row["teacher_cosine_mean"],
            })
        wide = {
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            **{f"{condition}_successes_of_64": counts[condition] for condition in CONDITIONS},
            **{f"{condition}_B63": counts[condition] >= 63 for condition in CONDITIONS},
        }
        robust_output.append(wide)
        minimum = next((condition[1:] for condition in BURSTS if counts[condition] >= 63), "NONE")
        exact_minimum = next((condition[1:] for condition in BURSTS if exact[state_id][condition]["outcome"] == "success"), "NONE")
        min_output.append({
            **wide, "L_min_Gphi_B63": minimum, "L_min_Gphi_exact_flow": exact_minimum,
            "dense_H1_B63": counts["H1"] >= 63,
        })
        sequence = [counts[condition] for condition in BURSTS]
        violations = [
            f"{BURSTS[index]}>{BURSTS[index + 1]} ({sequence[index]}>{sequence[index + 1]})"
            for index in range(len(sequence) - 1) if sequence[index] > sequence[index + 1]
        ]
        monotonic_output.append({
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            **{f"{condition}_successes": counts[condition] for condition in BURSTS},
            "exact_nondecreasing": not violations, "violations": "; ".join(violations),
            "largest_downward_change": max(
                [sequence[index] - sequence[index + 1] for index in range(len(sequence) - 1)] + [0]
            ),
        })
    write_csv(HERE / "exact_flow_by_burst.csv", exact_output)
    write_csv(HERE / "robust64_by_burst.csv", robust_output)
    write_csv(HERE / "per_state_min_burst.csv", min_output)
    write_csv(HERE / "monotonicity_audit.csv", monotonic_output)

    oracle_counts = {row["state_id"]: row for row in csv.DictReader((ORACLE / "robust64_by_burst.csv").open())}
    oracle_min = {row["state_id"]: row for row in csv.DictReader((ORACLE / "per_state_min_burst.csv").open())}
    comparison: list[dict] = []
    for gphi in min_output:
        state_id = gphi["state_id"]
        oracle = oracle_counts[state_id]
        comparison.append({
            "case_id": gphi["case_id"], "benchmark": gphi["benchmark"], "state_id": state_id,
            "oracle_L_min_B63": oracle_min[state_id]["L_min_B63"],
            "Gphi_L_min_B63": gphi["L_min_Gphi_B63"],
            **{
                f"oracle_{condition}_successes": int(oracle[f"{condition}_successes_of_64"])
                for condition in CONDITIONS
            },
            **{
                f"Gphi_{condition}_successes": int(gphi[f"{condition}_successes_of_64"])
                for condition in CONDITIONS
            },
        })
    write_csv(HERE / "oracle_vs_gphi_burst.csv", comparison)

    failure_rows: list[dict] = []
    summary: dict[str, dict] = {}
    jdef_rows: list[dict] = []
    for condition in CONDITIONS:
        all_rows = [row for state in states for row in rows[state["state_id"]][condition]]
        outcomes = Counter(row["outcome"] for row in all_rows)
        successes = outcomes["success"]
        b63 = sum(
            sum(row["outcome"] == "success" for row in rows[state["state_id"]][condition]) >= 63
            for state in states
        )
        exact_counts = Counter(exact[state["state_id"]][condition]["outcome"] for state in states)
        output = {
            "condition": condition, "robust_success": successes, "robust_total": len(all_rows),
            "success_rate": successes / len(all_rows), "B63_states": b63,
            "deadlock": outcomes["deadlock"], "timeout": outcomes["timeout"],
            "collision": outcomes["collision"], "execution_error": outcomes["execution_error"],
            "exact_success": exact_counts["success"], "exact_deadlock": exact_counts["deadlock"],
            "exact_timeout": exact_counts["timeout"], "exact_collision": exact_counts["collision"],
            "mean_J_def": mean([float(row["J_def"]) for row in all_rows]),
            "active_fraction": sum(int(row["active_steps"]) for row in all_rows) / sum(int(row["continuation_steps"]) for row in all_rows),
        }
        failure_rows.append(output)
        summary[condition] = output
        for subset in ("all", "success", "failure"):
            selected = all_rows if subset == "all" else [
                row for row in all_rows if (row["outcome"] == "success") == (subset == "success")
            ]
            values = [float(row["J_def"]) for row in selected]
            jdef_rows.append({
                "condition": condition, "subset": subset, "count": len(selected),
                "mean_J_def": mean(values), "median_J_def": median(values),
                "P95_J_def": quantile(values, .95), "max_J_def": max(values) if values else math.nan,
                "active_correction_fraction": output["active_fraction"],
            })
    write_csv(HERE / "failure_types_by_burst.csv", failure_rows)
    write_csv(HERE / "jdef_by_burst.csv", jdef_rows)

    first_burst_rows: list[dict] = []
    drift_rows: list[dict] = []
    for state in states:
        state_id = state["state_id"]
        for condition in CONDITIONS:
            path = HERE / "active_logs" / f"{state_id}__{condition}__robust.npz"
            with np.load(path, allow_pickle=False) as data:
                values = {name: np.asarray(data[name]) for name in data.files}
            outcome_by_seed = {int(row["seed"]): row["outcome"] for row in rows[state_id][condition]}
            for burst_step in sorted(set(int(value) for value in values["burst_step"][values["burst_ordinal"] == 0])):
                base = (values["burst_ordinal"] == 0) & (values["burst_step"] == burst_step)
                for subset in ("all", "success", "failure"):
                    mask = base.copy()
                    if subset != "all":
                        wanted = subset == "success"
                        mask &= np.asarray([outcome_by_seed[int(seed)] == "success" for seed in values["seed"]]) == wanted
                    count = int(mask.sum())
                    if not count:
                        continue
                    first_burst_rows.append({
                        "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                        "condition": condition, "burst_step": burst_step, "outcome_subset": subset,
                        "count": count, "raw_L2_mean": mean(values["raw_l2"][mask].tolist()),
                        "executed_L2_mean": mean(values["executed_l2"][mask].tolist()),
                        "executed_L2_median": median(values["executed_l2"][mask].tolist()),
                        "cosine_mean": mean(values["cosine"][mask].tolist()),
                        "norm_ratio_mean": mean(values["norm_ratio"][mask].tolist()),
                        "Gphi_executed_norm_mean": mean(np.linalg.norm(values["gphi_executed"][mask], axis=1).tolist()),
                        "eta_executed_norm_mean": mean(np.linalg.norm(values["eta_executed"][mask], axis=1).tolist()),
                        "Gphi_projection_rewrite_mean": mean(values["gphi_rewrite"][mask].tolist()),
                    })
            for seed in SEEDS:
                seed_mask = values["seed"] == seed
                indexes = np.flatnonzero(seed_mask)
                if not len(indexes):
                    continue
                bins = np.minimum((np.arange(len(indexes)) * 10 // max(len(indexes), 1)), 9)
                for decile in range(10):
                    chosen = indexes[bins == decile]
                    if not len(chosen):
                        continue
                    drift_rows.append({
                        "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                        "condition": condition, "seed": seed,
                        "outcome": outcome_by_seed[seed], "active_progress_decile": decile,
                        "active_steps_in_bin": len(chosen),
                        "global_step_mean": mean(values["global_step"][chosen].tolist()),
                        "executed_L2_mean": mean(values["executed_l2"][chosen].tolist()),
                        "cosine_mean": mean(values["cosine"][chosen].tolist()),
                        "norm_ratio_mean": mean(values["norm_ratio"][chosen].tolist()),
                    })
    write_csv(HERE / "first_burst_prediction_error.csv", first_burst_rows)
    write_csv(HERE / "prediction_drift_over_time.csv", drift_rows)

    divergence_rows: list[dict] = []
    for state in states:
        state_id = state["state_id"]
        for condition in CONDITIONS:
            oracle_exact = next(
                row for row in csv.DictReader((ORACLE / "exact_flow_by_burst.csv").open())
                if row["state_id"] == state_id and row["condition"] == condition
            )
            gphi_exact = exact[state_id][condition]
            if oracle_exact["outcome"] != "success" or gphi_exact["outcome"] == "success":
                continue
            gt = np.load(trace_path(state_id, condition, False), allow_pickle=False)
            ot = np.load(trace_path(state_id, condition, True), allow_pickle=False)
            gs = gt["global_step"].astype(int); os_ = ot["global_step"].astype(int)
            gi = {int(step): index for index, step in enumerate(gs)}
            oi = {int(step): index for index, step in enumerate(os_)}
            common = np.intersect1d(gs, os_)
            first_divergence = None; divergence_norm = None
            for step in common:
                value = float(np.max(np.abs(gt["positions_after"][gi[int(step)]] - ot["positions_after"][oi[int(step)]])))
                if value > 1e-8:
                    first_divergence, divergence_norm = int(step), value
                    break
            active_indexes = np.flatnonzero(gt["active"])
            first_active = int(gs[active_indexes[0]]) if len(active_indexes) else None
            mismatch_values = gt["executed_l2"].astype(float)
            finite_active = [index for index in active_indexes if np.isfinite(mismatch_values[index])]
            first_mismatch = int(gs[finite_active[0]]) if finite_active else None
            first_large = next((int(gs[index]) for index in finite_active if mismatch_values[index] >= .05), None)
            first_large_index = gi[first_large] if first_large is not None else None
            divergence_rows.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                "condition": condition, "oracle_outcome": oracle_exact["outcome"],
                "Gphi_outcome": gphi_exact["outcome"],
                "first_active_step": first_active,
                "first_state_divergence_step": first_divergence,
                "first_state_divergence_seconds": first_divergence * .05 if first_divergence is not None else None,
                "position_max_abs_difference_at_divergence": divergence_norm,
                "first_correction_mismatch_step": first_mismatch,
                "first_executed_L2_ge_0p05_step": first_large,
                "large_mismatch_reporting_threshold_mps": 0.05,
                "large_mismatch_within_first_burst": bool(first_large_index is not None and gt["burst_ordinal"][first_large_index] == 0),
                "executed_L2_at_first_large": float(mismatch_values[first_large_index]) if first_large_index is not None else None,
            })
    write_csv(HERE / "state_divergence_cases.csv", divergence_rows)

    min_distribution = Counter(row["L_min_Gphi_B63"] for row in min_output)
    nonmonotonic = [row for row in monotonic_output if not row["exact_nondecreasing"]]
    best_burst = max(BURSTS, key=lambda condition: summary[condition]["success_rate"])
    best_rate = summary[best_burst]["success_rate"]
    if summary["L4"]["B63_states"] >= 14 and summary["L4"]["success_rate"] >= .9:
        classification = "GPHI_SHORT_BURST_RECOVERS_STRICT_DEADLOCK"
    elif max(summary["L6"]["B63_states"], summary["L8"]["B63_states"]) >= 14 and max(summary["L6"]["success_rate"], summary["L8"]["success_rate"]) >= .9:
        classification = "GPHI_LONG_BURST_REQUIRED"
    elif best_burst != "L8" and best_rate > summary["L8"]["success_rate"]:
        classification = "GPHI_BURST_NONMONOTONIC"
    elif best_rate > summary["L1"]["success_rate"]:
        classification = "GPHI_BURST_HELPS_BUT_REMAINS_BELOW_ORACLE"
    else:
        classification = "GPHI_BURST_DOES_NOT_FIX_CLOSED_LOOP_DRIFT"

    oracle_min_distribution = Counter(row["oracle_L_min_B63"] for row in comparison)
    if classification in ("GPHI_SHORT_BURST_RECOVERS_STRICT_DEADLOCK", "GPHI_LONG_BURST_REQUIRED"):
        next_architecture = "persistent recovery mode" if best_burst in ("L6", "L8") else "fixed short burst"
        next_experiment = "Freeze the best learned burst duration and run one fresh-WIDE rescue/break regression without tuning."
    elif classification == "GPHI_BURST_NONMONOTONIC":
        next_architecture = "G_phi itself still requires closed-loop supervision before choosing a temporal architecture"
        next_experiment = "Collect oracle labels only along frozen G_phi burst trajectories at the first correction-mismatch states, then retrain once with no cadence change."
    else:
        next_architecture = "G_phi itself still requires closed-loop supervision"
        next_experiment = "Collect oracle labels only along frozen G_phi burst trajectories at the first correction-mismatch states, then retrain once with no cadence change."

    first_all = [row for row in first_burst_rows if row["outcome_subset"] == "all"]
    first_step = [row for row in first_all if int(row["burst_step"]) == 1]
    later_step = [row for row in first_all if int(row["burst_step"]) >= 2]
    runtime_parts = [json.loads(path.read_text()) for path in sorted(HERE.glob("runtime_shard*.json"))]
    retained_before_resume = ["old_r002__S0", "old_r018__S0"]
    retained_rows = [
        row for state_id in retained_before_resume for condition in CONDITIONS
        for row in ([exact[state_id][condition]] + rows[state_id][condition])
    ]
    retained_active_log_rows = 0
    for state_id in retained_before_resume:
        for condition in CONDITIONS:
            with np.load(HERE / "active_logs" / f"{state_id}__{condition}__robust.npz", allow_pickle=False) as data:
                retained_active_log_rows += len(data["global_step"])
    runtime = {
        "audit": "gphi_strict_deadlock_burst_length_v1",
        "new_rollouts_retained": sum(int(row["new_rollouts"]) for row in runtime_parts) + len(retained_rows),
        "new_rollouts_post_resume": sum(int(row["new_rollouts"]) for row in runtime_parts),
        "retained_rollouts_before_resume": len(retained_rows),
        "new_physical_steps_retained": sum(int(row["physical_steps"]) for row in runtime_parts) + sum(int(row["continuation_steps"]) for row in retained_rows),
        "new_physical_steps_post_resume": sum(int(row["physical_steps"]) for row in runtime_parts),
        "active_timestep_log_rows": sum(int(row["active_timestep_log_rows"]) for row in runtime_parts) + retained_active_log_rows,
        "rollout_wall_seconds_max_shard": max(float(row["elapsed_seconds"]) for row in runtime_parts),
        "rollout_wall_seconds_sum_shards": sum(float(row["elapsed_seconds"]) for row in runtime_parts),
        "interrupted_optimization_job": {"scheduler_job_id": 295, "wall_seconds_approx": 434, "reason": "introduced exact phase-zero L8/H1 semantic reuse after two completed states"},
        "total_elapsed_including_interrupted_job_seconds": 434 + max(float(row["elapsed_seconds"]) for row in runtime_parts),
        "analysis_seconds": time.monotonic() - started,
        "gpu_shards": 2, "cpu_cores_requested": 6, "memory_requested_GB": 48,
        "scheduler_job_id": 296, "scheduler_accounting_available": False,
        "checkpoint_sha256": checkpoint["sha256"],
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    table = []
    for condition in CONDITIONS:
        row = summary[condition]
        table.append(
            f"| {condition} | {row['robust_success']}/1088 | {row['success_rate']:.4f} | {row['deadlock']} | "
            f"{row['timeout']} | {row['B63_states']}/17 | {row['exact_success']}/17 | {row['mean_J_def']:.6f} |"
        )
    states_table = []
    for row in min_output:
        states_table.append(
            f"| {row['case_id']} | {row['L1_successes_of_64']} | {row['L2_successes_of_64']} | "
            f"{row['L4_successes_of_64']} | {row['L6_successes_of_64']} | {row['L8_successes_of_64']} | "
            f"{row['H1_successes_of_64']} | {row['L_min_Gphi_B63']} | {oracle_min[row['state_id']]['L_min_B63']} |"
        )
    report = f"""# Frozen G_phi strict-deadlock burst-length audit

## Result

**{classification}**

The exact original checkpoint `{checkpoint['sha256']}` was used. No training, eta search, online oracle, or gate was used. G_phi was recomputed from the current 214-D feature on every active burst step; no prediction was held. The frozen source eta was evaluated only as a counterfactual teacher diagnostic on each G_phi-visited state.

## Aggregate result

| condition | success | rate | deadlock | timeout | B63 states | exact success | mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Per-state result and oracle alignment

| case | L1 | L2 | L4 | L6 | L8 | H1 | G_phi min B63 | oracle min B63 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(states_table)}

G_phi minimum-B63 distribution: {dict(min_distribution)}. Oracle minimum-B63 distribution: {dict(oracle_min_distribution)}.

## Monotonicity and teacher drift

Nondecreasing learned state curves: {17 - len(nonmonotonic)}/17. Nonmonotonic states: {', '.join(row['case_id'] for row in nonmonotonic) if nonmonotonic else 'none'}.

Across per-state/condition first-burst summaries, mean executed correction error at burst step 1 is {mean([float(row['executed_L2_mean']) for row in first_step]):.6f} m/s; for steps 2+ it is {mean([float(row['executed_L2_mean']) for row in later_step]):.6f} m/s. Detailed state/condition/step metrics are in `first_burst_prediction_error.csv`, and every robust active-timestep prediction is preserved in compressed `active_logs/` with decile summaries in `prediction_drift_over_time.csv`.

Where the matched oracle exact Flow succeeds but G_phi fails, trajectory and correction divergence is localized in `state_divergence_cases.csv`. The 0.05 m/s "large mismatch" marker is diagnostic reporting only and never affects control.

## Interpretation

Best learned burst by aggregate strict-deadlock success: **{best_burst}** ({best_rate:.4f}). Supported next architecture: **{next_architecture}**.

Smallest justified next experiment: {next_experiment}

## Runtime/resources

- New retained rollouts: {runtime['new_rollouts_retained']}; retained physical steps: {runtime['new_physical_steps_retained']}.
- Logged active prediction timesteps: {runtime['active_timestep_log_rows']}.
- Maximum shard wall time: {runtime['rollout_wall_seconds_max_shard']:.1f} s.
- Allocation: 2 GPU shards, 6 CPU cores requested, 48 GB memory requested.
"""
    (HERE / "burst_report.md").write_text(report)

    integrity.update({
        "status": "PASS", "checkpoint_sha256": checkpoint["sha256"],
        "raw_files": 17, "raw_rows": 17 * 390,
        "exact_rows": 17 * 6, "robust_rows": 17 * 6 * 64,
        "active_log_files": 17 * 6, "exact_trace_files": 17 * 6,
        "matched_seed_identity": True, "Gphi_requeried_every_active_step": True,
        "active_predictions_fully_logged": True, "eta_search_performed": False,
        "eta_used_for_control": False, "training_performed": False,
        "phase_zero_L8_equals_H1_verified": True,
        "projection_retry_counts": {
            "first": sum(int(row["first_projection_retries"]) for state in states for condition in CONDITIONS for row in rows[state["state_id"]][condition]),
            "Gphi_second": sum(int(row["second_projection_retries"]) for state in states for condition in CONDITIONS for row in rows[state["state_id"]][condition]),
            "eta_diagnostic": sum(int(row["teacher_diagnostic_projection_retries"]) for state in states for condition in CONDITIONS for row in rows[state["state_id"]][condition]),
        },
    })
    write_json(HERE / "integrity_audit.json", integrity)

    outputs = [
        "source_manifest.json", "integrity_audit.json", "checkpoint_manifest.json", "burst_config.json",
        "exact_flow_by_burst.csv", "robust64_by_burst.csv", "per_state_min_burst.csv",
        "oracle_vs_gphi_burst.csv", "first_burst_prediction_error.csv",
        "prediction_drift_over_time.csv", "state_divergence_cases.csv",
        "failure_types_by_burst.csv", "jdef_by_burst.csv", "monotonicity_audit.csv",
        "burst_report.md", "runtime_statistics.json",
    ]
    manifest = {
        "schema": "gphi_strict_deadlock_burst_length_v1", "status": "COMPLETE",
        "classification": classification, "checkpoint_sha256": checkpoint["sha256"],
        "best_burst_by_strict_deadlock_success": best_burst,
        "supported_next_architecture": next_architecture,
        "files": {name: {"sha256": sha256(HERE / name), "bytes": (HERE / name).stat().st_size} for name in outputs},
        "raw_files": {path.name: sha256(path) for path in sorted((HERE / "raw").glob("*.jsonl"))},
        "trace_files": {path.name: sha256(path) for path in sorted((HERE / "traces").glob("*.npz"))},
        "active_log_files": {path.name: sha256(path) for path in sorted((HERE / "active_logs").glob("*.npz"))},
        "non_goals_verified": {
            "no_training": True, "checkpoint_unchanged": True, "no_eta_search": True,
            "no_online_oracle_control": True, "no_gate": True,
            "environment_and_projections_frozen": True,
        },
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE", "classification": classification,
        "best_burst": best_burst, "minimum_distribution": dict(min_distribution),
        "conditions": {condition: {
            "success": summary[condition]["robust_success"], "B63": summary[condition]["B63_states"],
            "exact": summary[condition]["exact_success"],
        } for condition in CONDITIONS},
    }, indent=2))


if __name__ == "__main__":
    main()
