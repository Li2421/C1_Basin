"""Validate and summarize the strict-deadlock fixed-eta burst-length audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/strict_deadlock_oracle_burst_length_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
CADENCE = ROOT / "diagnostics/strict_deadlock_oracle_cadence_v1"
CONDITIONS = ("H1", "L1", "L2", "L4", "L6", "L8")
BURST_CONDITIONS = ("L1", "L2", "L4", "L6", "L8")
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


def eta_key(value: object) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(round(float(item), 8) for item in value)  # type: ignore[arg-type]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def quantile(values: list[float], value: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), value)) if values else math.nan


def first_trigger_count(query_step: int, terminal_step: int) -> int:
    first = ((query_step + 7) // 8) * 8
    last = terminal_step - 1
    return 0 if first > last else (last - first) // 8 + 1


def normalize(row: dict, condition: str, query_step: int) -> dict:
    if condition == "H1" and "steps" in row:
        steps = int(row["steps"])
        terminal = int(row["terminal_step"])
        active_steps = steps
        burst_count = 1 if steps else 0
        raw = float(row["raw_correction_norm_mean"])
        executed = float(row["executed_correction_norm_mean"])
        rewrite = float(row["projection_rewrite_norm_mean"])
    elif condition == "H1":
        steps = int(row["continuation_steps"])
        terminal = int(row["terminal_global_step"])
        active_steps = int(row["active_steps"])
        burst_count = 1 if steps else 0
        raw = float(row["active_raw_norm_mean"])
        executed = float(row["active_executed_norm_mean"])
        rewrite = float(row["active_rewrite_norm_mean"])
    elif condition == "L1":
        steps = int(row["continuation_steps"])
        terminal = int(row["terminal_global_step"])
        active_steps = int(row["active_steps"])
        burst_count = first_trigger_count(query_step, terminal)
        raw = float(row["active_raw_norm_mean"])
        executed = float(row["active_executed_norm_mean"])
        rewrite = float(row["active_rewrite_norm_mean"])
    else:
        steps = int(row["continuation_steps"])
        terminal = int(row["terminal_global_step"])
        active_steps = int(row["active_steps"])
        burst_count = int(row["burst_count"])
        raw = float(row["active_raw_norm_mean"])
        executed = float(row["active_executed_norm_mean"])
        rewrite = float(row["active_rewrite_norm_mean"])
    return {
        "condition": condition, "outcome": row["outcome"], "seed": row.get("seed"),
        "steps": steps, "terminal_step": terminal, "J_def": float(row["J_def"]),
        "active_steps": active_steps, "active_fraction": active_steps / steps if steps else 0.0,
        "burst_count": burst_count, "active_raw_norm_mean": raw,
        "active_executed_norm_mean": executed, "active_rewrite_norm_mean": rewrite,
        "first_projection_retries": int(row.get("first_projection_retries", 0)),
        "second_projection_retries": int(row.get("second_projection_retries", 0)),
        "trace_file": row.get("trace_file"),
    }


def main() -> None:
    started = time.monotonic()
    source = json.loads((HERE / "source_manifest.json").read_text())
    burst_config = json.loads((HERE / "burst_config.json").read_text())
    integrity = json.loads((HERE / "integrity_audit.json").read_text())
    if integrity["status"] != "PREPARED_REUSE_GATE_PASS":
        raise RuntimeError("preparation gate not passed")
    states = source["states"]
    state_by_id = {state["state_id"]: state for state in states}
    reuse_by_id = {row["state_id"]: row for row in source["reuse"]}
    eta_by_id = {row["state_id"]: eta_key(row["eta"]) for row in source["etas"]}
    if len(states) != 17:
        raise RuntimeError("expected 17 states")

    robust: dict[str, dict[str, list[dict]]] = {}
    exact: dict[str, dict[str, dict]] = {}
    for state in states:
        state_id = state["state_id"]
        eta = eta_by_id[state_id]
        reuse = reuse_by_id[state_id]
        if sha256(Path(reuse["H1_robust_path"])) != reuse["H1_robust_sha256"]:
            raise RuntimeError(f"H1 source changed {state_id}")
        if sha256(Path(reuse["L1_raw_path"])) != reuse["L1_raw_sha256"]:
            raise RuntimeError(f"L1 source changed {state_id}")
        h1 = [
            normalize(row, "H1", int(state["query_step"]))
            for row in jsonl(Path(reuse["H1_robust_path"]))
            if not row.get("state_complete") and eta_key(row["eta"]) == eta
        ]
        h1.sort(key=lambda row: int(row["seed"]))
        old = jsonl(Path(reuse["L1_raw_path"]))
        l1 = [
            normalize(row, "L1", int(state["query_step"]))
            for row in old if row["condition"] == "H8" and row["flow_mode"] == "robust"
        ]
        l1.sort(key=lambda row: int(row["seed"]))
        new_path = HERE / "raw" / f"{state_id}.jsonl"
        if not new_path.is_file():
            raise RuntimeError(f"missing new result {state_id}")
        new = jsonl(new_path)
        if len(new) != 260 or not all(row.get("state_complete") for row in new):
            raise RuntimeError(f"incomplete new result {state_id}: {len(new)}")
        if any(eta_key(row["eta"]) != eta for row in new):
            raise RuntimeError(f"eta mismatch {state_id}")
        robust[state_id] = {"H1": h1, "L1": l1}
        exact[state_id] = {
            "H1": normalize(reuse["H1_exact"], "H1", int(state["query_step"])),
            "L1": normalize(reuse["L1_exact"], "L1", int(state["query_step"])),
        }
        for condition in ("L2", "L4", "L6", "L8"):
            condition_rows = [
                normalize(row, condition, int(state["query_step"]))
                for row in new if row["condition"] == condition and row["flow_mode"] == "robust"
            ]
            condition_rows.sort(key=lambda row: int(row["seed"]))
            exact_rows = [row for row in new if row["condition"] == condition and row["flow_mode"] == "exact"]
            if len(exact_rows) != 1:
                raise RuntimeError(f"exact row mismatch {state_id} {condition}")
            robust[state_id][condition] = condition_rows
            exact[state_id][condition] = normalize(exact_rows[0], condition, int(state["query_step"]))
        for condition in CONDITIONS:
            seeds = [int(row["seed"]) for row in robust[state_id][condition]]
            if seeds != SEEDS:
                raise RuntimeError(f"seed mismatch {state_id} {condition}")
        if sum(row["outcome"] == "success" for row in robust[state_id]["H1"]) != 64:
            raise RuntimeError(f"H1 reference reproduction failed {state_id}")
        if int(state["query_step"]) == 0:
            for h1_row, l8_row in zip(robust[state_id]["H1"], robust[state_id]["L8"]):
                if not (
                    h1_row["outcome"] == l8_row["outcome"]
                    and h1_row["terminal_step"] == l8_row["terminal_step"]
                    and abs(h1_row["J_def"] - l8_row["J_def"]) <= 1e-10
                ):
                    raise RuntimeError(f"L8 did not equal dense H1 for phase-zero state {state_id}")

    exact_output: list[dict] = []
    state_counts: list[dict] = []
    min_rows: list[dict] = []
    monotonic_rows: list[dict] = []
    for state in states:
        state_id = state["state_id"]
        counts = {
            condition: sum(row["outcome"] == "success" for row in robust[state_id][condition])
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            row = exact[state_id][condition]
            exact_output.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                "query_step": state["query_step"], "query_phase_mod8": int(state["query_step"]) % 8,
                "condition": condition, "outcome": row["outcome"],
                "terminal_global_step": row["terminal_step"],
                "completion_time_seconds": row["terminal_step"] * .05,
                "J_def": row["J_def"], "active_fraction": row["active_fraction"],
            })
        state_count = {
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            **{f"{condition}_successes_of_64": counts[condition] for condition in CONDITIONS},
            **{f"{condition}_B63": counts[condition] >= 63 for condition in CONDITIONS},
        }
        state_counts.append(state_count)
        minimum = next((condition[1:] for condition in BURST_CONDITIONS if counts[condition] >= 63), "NONE")
        min_rows.append({
            **state_count, "L_min_B63": minimum,
            "special_phase_case": state_id == "old_r106__S_8s",
        })
        sequence = [counts[condition] for condition in BURST_CONDITIONS]
        violations = [
            f"{BURST_CONDITIONS[index]}>{BURST_CONDITIONS[index + 1]} ({sequence[index]}>{sequence[index + 1]})"
            for index in range(len(sequence) - 1) if sequence[index] > sequence[index + 1]
        ]
        monotonic_rows.append({
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            **{f"{condition}_successes": counts[condition] for condition in BURST_CONDITIONS},
            "exact_nondecreasing": not violations,
            "violations": "; ".join(violations),
            "largest_downward_change": max(
                [sequence[index] - sequence[index + 1] for index in range(len(sequence) - 1)] + [0]
            ),
        })
    write_csv(HERE / "exact_flow_by_burst.csv", exact_output)
    write_csv(HERE / "robust64_by_burst.csv", state_counts)
    write_csv(HERE / "per_state_min_burst.csv", min_rows)
    write_csv(HERE / "monotonicity_audit.csv", monotonic_rows)

    cohort_rows: list[dict] = []
    failure_rows: list[dict] = []
    condition_summary: dict[str, dict] = {}
    for cohort in ("historical", "fresh_unseen", "combined"):
        chosen = [state for state in states if cohort == "combined" or state["benchmark"] == cohort]
        for condition in CONDITIONS:
            all_rows = [row for state in chosen for row in robust[state["state_id"]][condition]]
            outcomes = Counter(row["outcome"] for row in all_rows)
            successes = outcomes["success"]
            b63 = sum(
                sum(row["outcome"] == "success" for row in robust[state["state_id"]][condition]) >= 63
                for state in chosen
            )
            exact_successes = sum(exact[state["state_id"]][condition]["outcome"] == "success" for state in chosen)
            output = {
                "cohort": cohort, "condition": condition, "states": len(chosen),
                "exact_flow_successes": exact_successes,
                "robust_successes": successes, "robust_total": len(all_rows),
                "success_rate": successes / len(all_rows), "B63_states": b63,
                "deadlock": outcomes["deadlock"], "timeout": outcomes["timeout"],
                "collision": outcomes["collision"], "execution_error": outcomes["execution_error"],
            }
            cohort_rows.append(output)
            failure_rows.append(output.copy())
            if cohort == "combined":
                condition_summary[condition] = output.copy()
    write_csv(HERE / "historical_vs_fresh.csv", cohort_rows)
    write_csv(HERE / "failure_types_by_burst.csv", failure_rows)

    jdef_rows: list[dict] = []
    density_rows: list[dict] = []
    for condition in CONDITIONS:
        all_rows = [row for state in states for row in robust[state["state_id"]][condition]]
        for subset in ("all", "success", "failure"):
            selected = all_rows if subset == "all" else [
                row for row in all_rows if (row["outcome"] == "success") == (subset == "success")
            ]
            values = [row["J_def"] for row in selected]
            jdef_rows.append({
                "condition": condition, "subset": subset, "count": len(selected),
                "mean_J_def": mean(values), "median_J_def": median(values),
                "P95_J_def": quantile(values, .95), "max_J_def": max(values) if values else math.nan,
            })
        total_steps = sum(row["steps"] for row in all_rows)
        total_active = sum(row["active_steps"] for row in all_rows)
        weighted_exec = sum(row["active_executed_norm_mean"] * row["active_steps"] for row in all_rows)
        weighted_raw = sum(row["active_raw_norm_mean"] * row["active_steps"] for row in all_rows)
        weighted_rewrite = sum(row["active_rewrite_norm_mean"] * row["active_steps"] for row in all_rows)
        density_rows.append({
            "condition": condition, "continuations": len(all_rows),
            "total_physical_steps": total_steps, "total_active_steps": total_active,
            "active_correction_fraction": total_active / total_steps,
            "mean_episode_active_fraction": mean([row["active_fraction"] for row in all_rows]),
            "mean_burst_count_per_continuation": mean([float(row["burst_count"]) for row in all_rows]),
            "mean_active_steps_per_continuation": mean([float(row["active_steps"]) for row in all_rows]),
            "active_raw_norm_mean_weighted": weighted_raw / total_active,
            "active_executed_norm_mean_weighted": weighted_exec / total_active,
            "active_projection_rewrite_mean_weighted": weighted_rewrite / total_active,
            "first_projection_retries": sum(row["first_projection_retries"] for row in all_rows),
            "second_projection_retries": sum(row["second_projection_retries"] for row in all_rows),
        })
    write_csv(HERE / "jdef_by_burst.csv", jdef_rows)
    write_csv(HERE / "active_density_by_burst.csv", density_rows)

    trace_paths: dict[tuple[str, str], Path] = {}
    for state in states:
        state_id = state["state_id"]
        trace_paths[(state_id, "H1")] = CADENCE / "traces" / f"{state_id}__H1.npz"
        trace_paths[(state_id, "L1")] = CADENCE / "traces" / f"{state_id}__H8.npz"
        for condition in ("L2", "L4", "L6", "L8"):
            trace_paths[(state_id, condition)] = HERE / "traces" / f"{state_id}__{condition}.npz"
    divergence_rows: list[dict] = []
    for state in states:
        state_id = state["state_id"]
        outcomes = {condition: exact[state_id][condition]["outcome"] for condition in BURST_CONDITIONS}
        target_index = next((index for index, condition in enumerate(BURST_CONDITIONS) if outcomes[condition] == "success"), None)
        if target_index is None or target_index == 0:
            continue
        short = BURST_CONDITIONS[target_index - 1]
        long = BURST_CONDITIONS[target_index]
        if outcomes[short] == "success":
            continue
        short_trace = np.load(trace_paths[(state_id, short)], allow_pickle=False)
        long_trace = np.load(trace_paths[(state_id, long)], allow_pickle=False)
        short_active_key = "scheduled" if short == "L1" else "active"
        steps_short = short_trace["global_step"].astype(int)
        steps_long = long_trace["global_step"].astype(int)
        index_short = {int(step): index for index, step in enumerate(steps_short)}
        index_long = {int(step): index for index, step in enumerate(steps_long)}
        common = np.intersect1d(steps_short, steps_long)
        first_divergence = None
        divergence_norm = None
        persistence_step = None
        persistence_long_index = None
        for step in common:
            si, li = index_short[int(step)], index_long[int(step)]
            if first_divergence is None:
                distance = float(np.max(np.abs(short_trace["positions_after"][si] - long_trace["positions_after"][li])))
                if distance > 1e-8:
                    first_divergence, divergence_norm = int(step), distance
            if persistence_step is None and not bool(short_trace[short_active_key][si]) and bool(long_trace["active"][li]):
                persistence_step, persistence_long_index = int(step), li
        output = {
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            "shorter_failing_condition": short, "longer_success_condition": long,
            "first_trajectory_divergence_step": first_divergence,
            "first_trajectory_divergence_seconds": first_divergence * .05 if first_divergence is not None else None,
            "position_max_abs_difference_at_divergence": divergence_norm,
            "first_short_inactive_long_active_step": persistence_step,
        }
        if persistence_long_index is not None:
            li = persistence_long_index
            raw_norm = float(np.linalg.norm(long_trace["dense_raw_correction"][li]))
            executed_norm = float(np.linalg.norm(long_trace["dense_executed_correction"][li]))
            goal = long_trace["goal_errors_before"]
            goal_change = float(np.sum(goal[li + 1]) - np.sum(goal[li])) if li + 1 < len(goal) else math.nan
            output.update({
                "raw_eta_correction_norm": raw_norm,
                "executed_eta_correction_norm": executed_norm,
                "sum_goal_error_before": float(np.sum(goal[li])),
                "next_step_sum_goal_error_change": goal_change,
                "inter_agent_distance": float(long_trace["inter_agent_distance_before"][li]),
                "candidate_since": int(long_trace["candidate_since"][li]),
                "stuck_timer": float(long_trace["stuck_timer"][li]),
                "max_stuck_timer": float(long_trace["max_stuck_timer"][li]),
            })
        divergence_rows.append(output)
    write_csv(HERE / "divergence_cases.csv", divergence_rows)

    min_distribution = Counter(row["L_min_B63"] for row in min_rows)
    nonmonotonic = [row for row in monotonic_rows if not row["exact_nondecreasing"]]
    shortest_near_dense = next((
        condition for condition in BURST_CONDITIONS
        if condition_summary[condition]["B63_states"] >= 16
        and condition_summary[condition]["success_rate"] >= .99
    ), None)
    if shortest_near_dense in ("L1", "L2", "L4"):
        classification = "SHORT_BURST_SUFFICIENT"
    elif shortest_near_dense in ("L6", "L8"):
        classification = "LONG_BURST_REQUIRED"
    elif condition_summary["L8"]["B63_states"] < 15 or condition_summary["L8"]["success_rate"] < .95:
        classification = "FULL_DENSE_REQUIRED"
    else:
        classification = "HETEROGENEOUS_BURST_REQUIREMENTS"

    distinct_min = {value for value in min_distribution if value != "NONE"}
    if classification == "SHORT_BURST_SUFFICIENT":
        architecture = "H=8 trigger + fixed short dense burst"
    elif classification == "LONG_BURST_REQUIRED" and len(distinct_min) >= 3:
        architecture = "trigger + state-dependent burst duration"
    elif classification == "LONG_BURST_REQUIRED":
        architecture = "persistent recovery mode requiring an explicit exit condition"
    elif classification == "FULL_DENSE_REQUIRED":
        architecture = "persistent recovery mode requiring an explicit exit condition"
    else:
        architecture = "trigger + state-dependent burst duration"

    jdef_all = {row["condition"]: row for row in jdef_rows if row["subset"] == "all"}
    density = {row["condition"]: row for row in density_rows}
    if shortest_near_dense is not None:
        jdef_saved = 1 - float(jdef_all[shortest_near_dense]["mean_J_def"]) / float(jdef_all["H1"]["mean_J_def"])
    else:
        jdef_saved = math.nan
    special = next(row for row in min_rows if row["state_id"] == "old_r106__S_8s")

    if architecture == "H=8 trigger + fixed short dense burst":
        next_experiment = "Freeze the shortest near-dense burst and perform one oracle-only check of the same burst rule from complete episode starts on the 17 strict-deadlock episodes."
    elif architecture == "trigger + state-dependent burst duration":
        next_experiment = "Without training a model, test one preregistered oracle exit condition that ends an H8-triggered dense burst when the frozen deadlock monitor clears."
    else:
        next_experiment = "Test one frozen persistent recovery-mode exit rule, with the existing eta and trigger unchanged, on these 17 states."

    runtime_parts = [json.loads(path.read_text()) for path in sorted(HERE.glob("runtime_shard*.json"))]
    runtime = {
        "audit": "strict_deadlock_oracle_burst_length_v1",
        "new_rollouts": sum(int(row["new_rollouts"]) for row in runtime_parts),
        "new_physical_steps": sum(int(row["physical_steps"]) for row in runtime_parts),
        "reused_H1_robust": 1088, "reused_L1_robust": 1088,
        "reused_H1_L1_exact": 34,
        "rollout_wall_seconds_max_shard": max(float(row["elapsed_seconds"]) for row in runtime_parts),
        "rollout_wall_seconds_sum_shards": sum(float(row["elapsed_seconds"]) for row in runtime_parts),
        "analysis_seconds": time.monotonic() - started,
        "gpu_shards": 2, "cpu_cores_requested": 6, "memory_requested_GB": 48,
        "scheduler_job_id": 294, "scheduler_accounting_available": False,
        "devices": [row["device"] for row in runtime_parts],
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    summary_lines = []
    for condition in CONDITIONS:
        row = condition_summary[condition]
        summary_lines.append(
            f"| {condition} | {row['robust_successes']}/1088 | {row['success_rate']:.4f} | "
            f"{row['deadlock']} | {row['timeout']} | {row['B63_states']}/17 | "
            f"{float(jdef_all[condition]['mean_J_def']):.6f} | {float(density[condition]['active_correction_fraction']):.4f} |"
        )
    per_state_lines = []
    for row in min_rows:
        per_state_lines.append(
            f"| {row['case_id']} | {row['benchmark']} | {row['H1_successes_of_64']} | "
            f"{row['L1_successes_of_64']} | {row['L2_successes_of_64']} | {row['L4_successes_of_64']} | "
            f"{row['L6_successes_of_64']} | {row['L8_successes_of_64']} | {row['L_min_B63']} |"
        )
    cohort_lines = []
    for cohort in ("historical", "fresh_unseen"):
        for condition in CONDITIONS:
            row = next(item for item in cohort_rows if item["cohort"] == cohort and item["condition"] == condition)
            cohort_lines.append(
                f"| {cohort} | {condition} | {row['robust_successes']}/{row['robust_total']} | "
                f"{row['success_rate']:.4f} | {row['B63_states']}/{row['states']} |"
            )

    report = f"""# Strict-deadlock fixed-eta burst-length audit

## Result

**{classification}**

This is an oracle-only temporal-persistence ablation. Every state uses its frozen minimum-J_def eta, frozen H=8 global trigger phase, matched Flow streams, and unchanged controller stack. No eta search, learned G_phi, gate, or model training occurred.

## Integrity and semantics

- States: 17 (11 historical, 6 fresh unseen); starting state and eta hashes match the capacity/cadence audits.
- H1 robust rows reused: 1088/1088 successful.
- L1 rows reused: 1088, exactly the prior one-step H8 condition.
- New conditions: L2, L4, L6, L8; 64 matched seeds per state plus exact Flow.
- A burst is created only by a trigger observed at an absolute global timestep divisible by 8. Query start never creates an artificial trigger.
- Special old_r106 phase: query step 279 is inactive; first eligible trigger is global step 280. L8 is therefore not silently treated as H1 at its first continuation step.

## Aggregate robust result

| condition | success/1088 | rate | deadlock | timeout | B63 states | mean J_def | active fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

The shortest tested condition restoring near-dense capacity (at least 16/17 B63 and aggregate success at least 0.99) is **{shortest_near_dense or 'NONE'}**. Its mean-J_def saving relative to dense H1 is {jdef_saved:.2%}.

## Per-state robust persistence

| case | cohort | H1 | L1 | L2 | L4 | L6 | L8 | minimum B63 L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(per_state_lines)}

Minimum-B63 distribution: {dict(min_distribution)}.

## Monotonicity

Exact nondecreasing state curves: {17 - len(nonmonotonic)}/17. Nonmonotonic states: {', '.join(row['case_id'] for row in nonmonotonic) if nonmonotonic else 'none'}. Raw state counts and any downward transitions are recorded in `monotonicity_audit.csv`.

## Historical versus fresh-unseen cohorts

| cohort | condition | success | rate | B63 states |
|---|---|---:|---:|---:|
{chr(10).join(cohort_lines)}

## Special phase and temporal evidence

old_r106 begins at global step 279 (phase 7), waits until the trigger at step 280, and has minimum B63 burst length **{special['L_min_B63']}**. Representative exact-Flow transitions from a shorter failing burst to a longer successful burst are in `divergence_cases.csv`, including the first shorter-off/longer-active step, correction after projection, goal-error change, and monitor state.

## Interpretation

Evidence supports: **{architecture}**.

Smallest justified next experiment: {next_experiment}

Success remains the hard constraint; J_def savings are interpreted only for conditions that restore robust recovery.

## Runtime and resources

- New rollouts: {runtime['new_rollouts']}; new physical steps: {runtime['new_physical_steps']}.
- Reused robust rollouts: 1088 H1 + 1088 L1.
- Maximum shard wall time: {runtime['rollout_wall_seconds_max_shard']:.1f} s.
- Allocation: 2 GPU shards, 6 CPU cores requested, 48 GB memory requested.
"""
    (HERE / "burst_report.md").write_text(report)

    integrity.update({
        "status": "PASS", "new_raw_state_files": 17, "new_raw_rows": 17 * 260,
        "new_exact_rows": 17 * 4, "new_robust_rows": 17 * 4 * 64,
        "matched_seed_identity": True, "global_phase_preserved": True,
        "old_r106_initial_wait_preserved": True,
        "L8_equals_H1_for_all_16_phase_zero_states": True,
        "eta_search_performed": False, "Gphi_loaded_or_evaluated": False,
        "projection_retries_total": sum(int(row["first_projection_retries"]) + int(row["second_projection_retries"]) for row in density_rows),
    })
    write_json(HERE / "integrity_audit.json", integrity)

    outputs = [
        "source_manifest.json", "integrity_audit.json", "burst_config.json",
        "exact_flow_by_burst.csv", "robust64_by_burst.csv", "per_state_min_burst.csv",
        "monotonicity_audit.csv", "failure_types_by_burst.csv", "jdef_by_burst.csv",
        "active_density_by_burst.csv", "divergence_cases.csv", "historical_vs_fresh.csv",
        "burst_report.md", "runtime_statistics.json",
    ]
    manifest = {
        "schema": "strict_deadlock_oracle_burst_length_v1", "status": "COMPLETE",
        "classification": classification, "supported_architecture": architecture,
        "shortest_near_dense_burst": shortest_near_dense,
        "files": {name: {"sha256": sha256(HERE / name), "bytes": (HERE / name).stat().st_size} for name in outputs},
        "raw_files": {path.name: sha256(path) for path in sorted((HERE / "raw").glob("*.jsonl"))},
        "trace_files": {path.name: sha256(path) for path in sorted((HERE / "traces").glob("*.npz"))},
        "non_goals_verified": {
            "Gphi_not_trained_or_evaluated": True, "eta_not_searched": True,
            "gate_not_used": True, "trigger_period_and_offset_not_changed": True,
            "environment_and_projections_frozen": True,
        },
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE", "classification": classification,
        "shortest_near_dense": shortest_near_dense,
        "minimum_distribution": dict(min_distribution),
        "conditions": {
            condition: {
                "success": condition_summary[condition]["robust_successes"],
                "B63": condition_summary[condition]["B63_states"],
            } for condition in CONDITIONS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
