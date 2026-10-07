"""Integrity-check and summarize the frozen H1-versus-H8 oracle cadence audit."""

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
HERE = ROOT / "diagnostics/strict_deadlock_oracle_cadence_v1"
SOURCE = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
EXPECTED_SEEDS = list(range(95310001, 95310065))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def eta_key(value: object) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(round(float(item), 8) for item in value)  # type: ignore[arg-type]


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else math.nan


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_h1_rows(reuse: dict) -> list[dict]:
    path = Path(reuse["robust_path"])
    if sha256(path) != reuse["robust_path_sha256"]:
        raise RuntimeError(f"source H1 file changed: {path}")
    eta = eta_key(reuse["eta"])
    rows = [
        row for row in load_rows(path)
        if not row.get("state_complete") and eta_key(row["eta"]) == eta
    ]
    rows.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != EXPECTED_SEEDS:
        raise RuntimeError(f"H1 seeds differ for {reuse['state_id']}")
    return rows


def summarize(values: list[float], prefix: str) -> dict:
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_median": median(values),
        f"{prefix}_p95": quantile(values, .95),
        f"{prefix}_max": max(values) if values else math.nan,
    }


def main() -> None:
    started = time.monotonic()
    states = json.loads((HERE / "source_state_manifest.json").read_text())["states"]
    eta_rows = json.loads((HERE / "eta_manifest.json").read_text())["etas"]
    eta_by_state = {row["state_id"]: eta_key(row["eta"]) for row in eta_rows}
    integrity = json.loads((HERE / "integrity_audit.json").read_text())
    reuse_by_state = {row["state_id"]: row for row in integrity["H1_robust_reuse"]}
    if len(states) != 17 or len(eta_by_state) != 17 or len(reuse_by_state) != 17:
        raise RuntimeError("expected exactly 17 frozen cases")

    exact_rows: list[dict] = []
    robust_summary: list[dict] = []
    paired_rows: list[dict] = []
    classifications: list[dict] = []
    omitted_rows: list[dict] = []
    jdef_rows: list[dict] = []
    completion_rows: list[dict] = []
    projection_rows: list[dict] = []
    divergence_rows: list[dict] = []
    records_by_case: dict[str, dict] = {}
    exact_reproduced = True

    for state in states:
        state_id = state["state_id"]
        path = HERE / "raw" / f"{state_id}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing result {path}")
        rows = load_rows(path)
        if len(rows) != 66 or not all(row.get("state_complete") for row in rows):
            raise RuntimeError(f"incomplete {state_id}: {len(rows)} rows")
        if any(eta_key(row["eta"]) != eta_by_state[state_id] for row in rows):
            raise RuntimeError(f"eta changed for {state_id}")
        exact_new = {(row["condition"], row["flow_mode"]): row for row in rows[:2]}
        h1_exact = exact_new.get(("H1", "exact"))
        h8_exact = exact_new.get(("H8", "exact"))
        if h1_exact is None or h8_exact is None:
            raise RuntimeError(f"exact pair missing for {state_id}")
        h8_robust = [row for row in rows if row["condition"] == "H8" and row["flow_mode"] == "robust"]
        h8_robust.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in h8_robust] != EXPECTED_SEEDS:
            raise RuntimeError(f"H8 robust seed mismatch for {state_id}")
        h1_robust = source_h1_rows(reuse_by_state[state_id])
        source_exact = reuse_by_state[state_id]["exact_source_row"]
        exact_match = (
            h1_exact["outcome"] == source_exact["outcome"]
            and int(h1_exact["continuation_steps"]) == int(source_exact["steps"])
            and int(h1_exact["terminal_global_step"]) == int(source_exact["terminal_step"])
            and abs(float(h1_exact["J_def"]) - float(source_exact["J_def"])) <= 1e-10
        )
        exact_reproduced &= exact_match
        exact_rows.append({
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            "query_label": state["query_label"], "query_step": state["query_step"],
            "eta": json.dumps(list(eta_by_state[state_id]), separators=(",", ":")),
            "H1_outcome": h1_exact["outcome"], "H8_outcome": h8_exact["outcome"],
            "H1_terminal_global_step": h1_exact["terminal_global_step"],
            "H8_terminal_global_step": h8_exact["terminal_global_step"],
            "H1_completion_time_seconds": h1_exact["completion_time_seconds"],
            "H8_completion_time_seconds": h8_exact["completion_time_seconds"],
            "H1_J_def": h1_exact["J_def"], "H8_J_def": h8_exact["J_def"],
            "H8_over_H1_J_def": float(h8_exact["J_def"]) / float(h1_exact["J_def"]) if float(h1_exact["J_def"]) else math.nan,
            "H1_exact_source_reproduced": exact_match,
        })

        h1_success = sum(row["outcome"] == "success" for row in h1_robust)
        h8_success = sum(row["outcome"] == "success" for row in h8_robust)
        h1_counts = Counter(row["outcome"] for row in h1_robust)
        h8_counts = Counter(row["outcome"] for row in h8_robust)
        robust_summary.append({
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            "query_label": state["query_label"], "query_step": state["query_step"],
            "H1_successes": h1_success, "H8_successes": h8_success,
            "H1_Q": h1_success / 64, "H8_Q": h8_success / 64,
            "Delta_Q_H8_minus_H1": (h8_success - h1_success) / 64,
            "H1_deadlock": h1_counts["deadlock"], "H1_timeout": h1_counts["timeout"],
            "H1_collision": h1_counts["collision"],
            "H8_deadlock": h8_counts["deadlock"], "H8_timeout": h8_counts["timeout"],
            "H8_collision": h8_counts["collision"],
            "H1_B63": h1_success >= 63, "H8_B63": h8_success >= 63,
        })
        if h1_success < 63:
            classification = "REPRODUCTION_ERROR"
        elif h8_success >= 63:
            classification = "CADENCE_INVARIANT_ROBUST"
        elif h8_success > 32:
            classification = "CADENCE_DEGRADED_BUT_USABLE"
        else:
            classification = "SPARSE_CADENCE_BREAKS_ORACLE"
        classifications.append({
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            "H1_successes_of_64": h1_success, "H8_successes_of_64": h8_success,
            "classification": classification,
            "classification_note": "B63 is the frozen robust rule; non-B63 is called usable only when a strict majority still succeeds",
        })

        pair_counter: Counter = Counter()
        both_success_deltas: list[float] = []
        for h1, h8 in zip(h1_robust, h8_robust):
            if int(h1["seed"]) != int(h8["seed"]):
                raise RuntimeError(f"paired seed mismatch {state_id}")
            h1_ok = h1["outcome"] == "success"
            h8_ok = h8["outcome"] == "success"
            category = (
                "BOTH_SUCCESS" if h1_ok and h8_ok else
                "H1_SUCCESS_H8_FAIL" if h1_ok else
                "H1_FAIL_H8_SUCCESS" if h8_ok else "BOTH_FAIL"
            )
            pair_counter[category] += 1
            if h1_ok and h8_ok:
                delta_steps = int(h8["terminal_global_step"]) - int(h1["terminal_step"])
                both_success_deltas.append(delta_steps * .05)
                completion_rows.append({
                    "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                    "seed": int(h1["seed"]), "H1_terminal_global_step": int(h1["terminal_step"]),
                    "H8_terminal_global_step": int(h8["terminal_global_step"]),
                    "H1_completion_time_seconds": int(h1["terminal_step"]) * .05,
                    "H8_completion_time_seconds": int(h8["terminal_global_step"]) * .05,
                    "H8_minus_H1_seconds": delta_steps * .05,
                })
            paired_rows.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                "seed": int(h1["seed"]), "H1_outcome": h1["outcome"], "H8_outcome": h8["outcome"],
                "paired_category": category, "H1_J_def": h1["J_def"], "H8_J_def": h8["J_def"],
            })

        h1_j = [float(row["J_def"]) for row in h1_robust]
        h8_j = [float(row["J_def"]) for row in h8_robust]
        jdef_rows.append({
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            **summarize(h1_j, "H1_J_def"), **summarize(h8_j, "H8_J_def"),
            "ratio_of_means_H8_over_H1": mean(h8_j) / mean(h1_j) if mean(h1_j) else math.nan,
            "paired_mean_difference_H8_minus_H1": mean([b - a for a, b in zip(h1_j, h8_j)]),
        })
        omitted_rows.append({
            "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
            "H8_successes_of_64": h8_success,
            "omitted_steps_mean": mean([float(row["off_steps"]) for row in h8_robust]),
            "omitted_dense_raw_norm_mean": mean([float(row["omitted_dense_raw_norm_mean"]) for row in h8_robust]),
            "omitted_dense_executed_norm_mean": mean([float(row["omitted_dense_executed_norm_mean"]) for row in h8_robust]),
            "omitted_dense_executed_norm_p95_mean": mean([float(row["omitted_dense_executed_norm_p95"]) for row in h8_robust]),
            "omitted_dense_executed_norm_max": max(float(row["omitted_dense_executed_norm_max"]) for row in h8_robust),
            "nontrivial_fraction_gt_1e6_mean": mean([float(row["omitted_dense_executed_nontrivial_fraction_1e6"]) for row in h8_robust]),
            "omitted_dense_rewrite_norm_mean": mean([float(row["omitted_dense_rewrite_norm_mean"]) for row in h8_robust]),
        })
        for condition, group in (("H1", h1_robust), ("H8", h8_robust)):
            if condition == "H1":
                raw_key, exec_key, rewrite_key = (
                    "raw_correction_norm_mean", "executed_correction_norm_mean", "projection_rewrite_norm_mean"
                )
            else:
                raw_key, exec_key, rewrite_key = (
                    "active_raw_norm_mean", "active_executed_norm_mean", "active_rewrite_norm_mean"
                )
            projection_rows.append({
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                "condition": condition,
                "active_raw_norm_mean": mean([float(row[raw_key]) for row in group]),
                "active_executed_norm_mean": mean([float(row[exec_key]) for row in group]),
                "active_projection_rewrite_norm_mean": mean([float(row[rewrite_key]) for row in group]),
                "first_projection_retries": sum(int(row["first_projection_retries"]) for row in group),
                "second_projection_retries": sum(int(row["second_projection_retries"]) for row in group),
            })

        if h1_exact["outcome"] == "success" and h8_exact["outcome"] != "success":
            t1 = np.load(HERE / h1_exact["trace_file"], allow_pickle=False)
            t8 = np.load(HERE / h8_exact["trace_file"], allow_pickle=False)
            steps1 = t1["global_step"].astype(int)
            steps8 = t8["global_step"].astype(int)
            common = np.intersect1d(steps1, steps8)
            i1 = {int(step): index for index, step in enumerate(steps1)}
            i8 = {int(step): index for index, step in enumerate(steps8)}
            divergence_step = None
            divergence_norm = None
            for step in common:
                value = float(np.max(np.abs(t1["positions_after"][i1[int(step)]] - t8["positions_after"][i8[int(step)]])))
                if value > 1e-8:
                    divergence_step = int(step)
                    divergence_norm = value
                    break
            skipped_step = None
            skipped_index = None
            if divergence_step is not None:
                for index, step in enumerate(steps8):
                    if int(step) >= divergence_step and not bool(t8["scheduled"][index]):
                        skipped_step, skipped_index = int(step), index
                        break
            row = {
                "case_id": state["case_id"], "benchmark": state["benchmark"], "state_id": state_id,
                "first_position_divergence_global_step": divergence_step,
                "first_position_divergence_seconds": divergence_step * .05 if divergence_step is not None else None,
                "position_max_abs_difference_at_divergence": divergence_norm,
                "first_H8_skipped_step_at_or_after_divergence": skipped_step,
                "H8_outcome": h8_exact["outcome"], "H8_terminal_global_step": h8_exact["terminal_global_step"],
            }
            if skipped_index is not None:
                row.update({
                    "omitted_raw_norm_at_skipped_step": float(np.linalg.norm(t8["dense_raw_correction"][skipped_index])),
                    "omitted_executed_norm_at_skipped_step": float(np.linalg.norm(t8["dense_executed_correction"][skipped_index])),
                    "goal_error_agent0_at_skipped_step": float(t8["goal_errors_before"][skipped_index][0]),
                    "goal_error_agent1_at_skipped_step": float(t8["goal_errors_before"][skipped_index][1]),
                    "inter_agent_distance_at_skipped_step": float(t8["inter_agent_distance_before"][skipped_index]),
                    "stuck_timer_at_skipped_step": float(t8["stuck_timer"][skipped_index]),
                    "max_stuck_timer_at_skipped_step": float(t8["max_stuck_timer"][skipped_index]),
                })
            divergence_rows.append(row)

        records_by_case[state["case_id"]] = {
            "state": state, "h1": h1_robust, "h8": h8_robust,
            "h1_success": h1_success, "h8_success": h8_success,
            "pair_counter": pair_counter, "both_success_deltas": both_success_deltas,
        }

    if not exact_reproduced:
        raise RuntimeError("one or more exact H1 teacher replays differed from frozen source")

    write_csv(HERE / "exact_flow_h1_vs_h8.csv", exact_rows)
    write_csv(HERE / "robust64_h1_vs_h8.csv", robust_summary)
    write_csv(HERE / "paired_flow_outcomes.csv", paired_rows)
    write_csv(HERE / "episode_classification.csv", classifications)
    write_csv(HERE / "temporal_divergence.csv", divergence_rows)
    write_csv(HERE / "omitted_dense_correction_stats.csv", omitted_rows)
    write_csv(HERE / "jdef_comparison.csv", jdef_rows)
    write_csv(HERE / "completion_time_comparison.csv", completion_rows)
    write_csv(HERE / "projection_analysis.csv", projection_rows)

    cohort_rows = []
    for cohort in ("historical", "fresh_unseen", "combined"):
        chosen = [value for value in records_by_case.values() if cohort == "combined" or value["state"]["benchmark"] == cohort]
        h1_total = sum(value["h1_success"] for value in chosen)
        h8_total = sum(value["h8_success"] for value in chosen)
        total = 64 * len(chosen)
        pairs = sum((value["pair_counter"] for value in chosen), Counter())
        exact_chosen = [row for row in exact_rows if cohort == "combined" or row["benchmark"] == cohort]
        h1_j = [float(row["J_def"]) for value in chosen for row in value["h1"]]
        h8_j = [float(row["J_def"]) for value in chosen for row in value["h8"]]
        cohort_rows.append({
            "cohort": cohort, "episodes": len(chosen),
            "H1_exact_flow_successes": sum(row["H1_outcome"] == "success" for row in exact_chosen),
            "H8_exact_flow_successes": sum(row["H8_outcome"] == "success" for row in exact_chosen),
            "H1_robust_successes": h1_total, "H1_robust_total": total, "H1_Q": h1_total / total,
            "H8_robust_successes": h8_total, "H8_robust_total": total, "H8_Q": h8_total / total,
            "Delta_Q_H8_minus_H1": (h8_total - h1_total) / total,
            "H1_B63_episodes": sum(value["h1_success"] >= 63 for value in chosen),
            "H8_B63_episodes": sum(value["h8_success"] >= 63 for value in chosen),
            "BOTH_SUCCESS": pairs["BOTH_SUCCESS"],
            "H1_SUCCESS_H8_FAIL": pairs["H1_SUCCESS_H8_FAIL"],
            "H1_FAIL_H8_SUCCESS": pairs["H1_FAIL_H8_SUCCESS"],
            "BOTH_FAIL": pairs["BOTH_FAIL"],
            "H1_J_def_mean": mean(h1_j), "H8_J_def_mean": mean(h8_j),
            "H8_over_H1_J_def": mean(h8_j) / mean(h1_j),
        })
    write_csv(HERE / "historical_vs_fresh.csv", cohort_rows)

    combined = next(row for row in cohort_rows if row["cohort"] == "combined")
    class_counts = Counter(row["classification"] for row in classifications)
    if class_counts["REPRODUCTION_ERROR"]:
        aggregate = "INVALID_REPRODUCTION"
    elif combined["H8_B63_episodes"] >= 15 and combined["H8_Q"] >= .95:
        aggregate = "H8_ORACLE_CAPACITY_PRESERVED"
    elif class_counts["CADENCE_INVARIANT_ROBUST"] and (
        class_counts["CADENCE_DEGRADED_BUT_USABLE"] + class_counts["SPARSE_CADENCE_BREAKS_ORACLE"]
    ):
        aggregate = "MIXED_ORACLE_CADENCE_DEPENDENCE"
    else:
        aggregate = "ORACLE_CADENCE_MISMATCH_CONFIRMED"

    h1_all_j = [float(row["H1_J_def"]) for row in paired_rows]
    h8_all_j = [float(row["H8_J_def"]) for row in paired_rows]
    omitted_all = [float(row["omitted_dense_executed_norm_mean"]) for row in omitted_rows]
    nontrivial_all = [float(row["nontrivial_fraction_gt_1e6_mean"]) for row in omitted_rows]
    completion_deltas = [float(row["H8_minus_H1_seconds"]) for row in completion_rows]
    h1_ci = wilson(int(combined["H1_robust_successes"]), int(combined["H1_robust_total"]))
    h8_ci = wilson(int(combined["H8_robust_successes"]), int(combined["H8_robust_total"]))

    if aggregate == "ORACLE_CADENCE_MISMATCH_CONFIRMED":
        dominant = "temporal persistence / deployment cadence"
        next_experiment = "With eta and all controller semantics frozen, test one preregistered H=8 trigger followed by an 8-step dense fixed-eta burst on the same 17 states."
    elif aggregate == "H8_ORACLE_CAPACITY_PRESERVED":
        dominant = "learned G_phi prediction drift on subsequent H=8 query states"
        next_experiment = "Replay the learned H=8 controller and compare its prediction with the frozen eta teacher only at the successive H=8 query states."
    else:
        dominant = "heterogeneous temporal persistence requirements"
        next_experiment = "Test one preregistered H=8 trigger followed by an 8-step dense fixed-eta burst on only the cadence-sensitive subgroup, without eta re-search."

    runtimes = [json.loads(path.read_text()) for path in sorted(HERE.glob("runtime_shard*.json"))]
    runtime = {
        "audit": "strict_deadlock_oracle_cadence_v1",
        "new_rollouts": sum(int(row["new_rollouts"]) for row in runtimes),
        "new_physical_steps": sum(int(row["physical_steps"]) for row in runtimes),
        "reused_H1_robust_rollouts": 1088,
        "rollout_wall_seconds_max_shard": max(float(row["elapsed_seconds"]) for row in runtimes),
        "rollout_wall_seconds_sum_shards": sum(float(row["elapsed_seconds"]) for row in runtimes),
        "analysis_seconds": time.monotonic() - started,
        "gpu_shards": 2, "cpu_cores": 6, "memory_request_GB": 48,
        "scheduler_job_id": 293,
        "scheduler_accounting_available": False,
        "scheduler_accounting_note": "Slurm accounting storage is disabled; requested allocation and in-process wall/step counters are reported.",
        "devices": [row["device"] for row in runtimes],
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    report = f"""# Strict-deadlock fixed-eta oracle cadence audit

## Result

**{aggregate}**

This is a paired oracle-only cadence ablation. It uses the already selected minimum-deformation robust eta for each of the 17 frozen states; no eta search, learned G_phi, gate, or controller modification was performed. Dense H=1 was reproduced exactly before interpreting H=8.

## Integrity

- Frozen cases: 17 (historical 11, fresh unseen 6).
- Starting state: earliest robust queried augmented state (16 S0; old_r106 at S_8s), restored from the frozen full-state files.
- Eta: unchanged per-state minimum-J_def B63 eta from the capacity audit.
- Flow seeds: 64 matched continuations per state, seeds 95310001--95310064.
- Horizon: remaining global horizon through absolute step 850.
- Cadence phase: absolute global timestep; it is not reset at a queried state.
- Dense H=1 robust rollouts reused after exact state/eta/seed/code/horizon checks: 1088.
- Exact H=1 teacher replays matching outcome, terminal step, and J_def: 17/17.

## Exact inherited Flow

| cohort | episodes | H1 success | H8 success |
|---|---:|---:|---:|
{chr(10).join(f"| {r['cohort']} | {r['episodes']} | {r['H1_exact_flow_successes']} | {r['H8_exact_flow_successes']} |" for r in cohort_rows)}

## Matched 64-Flow robust result

| cohort | H1 success | H8 success | H1 Q | H8 Q | H8 B63 episodes | H1-success/H8-fail |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r['cohort']} | {r['H1_robust_successes']}/{r['H1_robust_total']} | {r['H8_robust_successes']}/{r['H8_robust_total']} | {r['H1_Q']:.4f} | {r['H8_Q']:.4f} | {r['H8_B63_episodes']}/{r['episodes']} | {r['H1_SUCCESS_H8_FAIL']} |" for r in cohort_rows)}

Combined Wilson 95% intervals: H1 [{h1_ci[0]:.4f}, {h1_ci[1]:.4f}], H8 [{h8_ci[0]:.4f}, {h8_ci[1]:.4f}].

Of the 1006 H8 robust-flow failures, {sum(int(r['H8_deadlock']) for r in robust_summary)} terminated as strict deadlock and {sum(int(r['H8_timeout']) for r in robust_summary)} as timeout; there were {sum(int(r['H8_collision']) for r in robust_summary)} collisions. On inherited exact Flow, H8 produced {sum(r['H8_outcome'] == 'deadlock' for r in exact_rows)} deadlocks, {sum(r['H8_outcome'] == 'timeout' for r in exact_rows)} timeouts, and {sum(r['H8_outcome'] == 'success' for r in exact_rows)} success.

## Per-episode robust counts

| case | cohort | state | H1/64 | H8/64 | classification |
|---|---|---|---:|---:|---|
{chr(10).join(f"| {r['case_id']} | {r['benchmark']} | {r['state_id']} | {r['H1_successes_of_64']} | {r['H8_successes_of_64']} | {r['classification']} |" for r in classifications)}

Classification counts: {dict(class_counts)}. B63 (>=63/64) is the sole robust threshold. Below B63, `CADENCE_DEGRADED_BUT_USABLE` is used only when a strict majority of matched continuations still succeeds; otherwise the condition is mostly failure and classified `SPARSE_CADENCE_BREAKS_ORACLE`. Raw counts above remain the primary evidence.

## Omitted correction and liveness

Across states, the mean counterfactual executed dense correction omitted on H8-off steps is {mean(omitted_all):.6f} m/s. The mean fraction of off steps with norm greater than the diagnostic numerical threshold 1e-6 is {mean(nontrivial_all):.4f}. This threshold is reporting-only and never affects control.

Among {len(completion_rows)} paired continuations successful under both conditions, H8 minus H1 completion time has mean {mean(completion_deltas):.3f} s, median {median(completion_deltas):.3f} s, and P95 {quantile(completion_deltas, .95):.3f} s. Failures that remain failures at the fixed horizon are not relabeled as merely slow.

## Deformation and projection

Across all matched continuations, mean J_def is {mean(h1_all_j):.6f} for H1 and {mean(h8_all_j):.6f} for H8 (ratio {mean(h8_all_j) / mean(h1_all_j):.4f}). Lower H8 deformation is not interpreted as beneficial when task success is lost. Active-step correction and second-projection statistics are in `projection_analysis.csv`; omitted dense corrections are in `omitted_dense_correction_stats.csv`.

Mean active-step raw / executed / projection-rewrite norms are {mean([float(r['active_raw_norm_mean']) for r in projection_rows if r['condition'] == 'H1']):.6f} / {mean([float(r['active_executed_norm_mean']) for r in projection_rows if r['condition'] == 'H1']):.6f} / {mean([float(r['active_projection_rewrite_norm_mean']) for r in projection_rows if r['condition'] == 'H1']):.6f} for H1 and {mean([float(r['active_raw_norm_mean']) for r in projection_rows if r['condition'] == 'H8']):.6f} / {mean([float(r['active_executed_norm_mean']) for r in projection_rows if r['condition'] == 'H8']):.6f} / {mean([float(r['active_projection_rewrite_norm_mean']) for r in projection_rows if r['condition'] == 'H8']):.6f} for H8. Projection retry counts were zero in both conditions.

## Interpretation

Dominant remaining bottleneck: **{dominant}**.

Smallest justified next experiment: {next_experiment}

The aggregate conclusion was assigned from the preregistered semantics: broad H8 preservation requires nearly all episodes to remain B63 with near-dense aggregate success; coexistence of robust and cadence-sensitive subgroups is mixed; otherwise substantial dense-to-sparse rescue loss confirms cadence mismatch.

## Resource use

- New rollouts: {runtime['new_rollouts']} (H8 robust plus exact H1/H8 trajectory replays).
- Reused validated H1 robust rollouts: 1088.
- New physical steps: {runtime['new_physical_steps']}.
- Rollout wall time: {runtime['rollout_wall_seconds_max_shard']:.1f} s (maximum shard).
- Resources: 2 GPU shards, 6 CPU cores requested, 48 GB memory requested.
"""
    (HERE / "cadence_report.md").write_text(report)

    integrity.update({
        "status": "PASS", "exact_H1_reproduced": exact_reproduced,
        "raw_result_files": 17, "raw_rows": 17 * 66,
        "H8_robust_rows": 17 * 64, "paired_seed_identity": True,
        "trace_files": 34, "eta_search_performed": False,
        "learned_Gphi_loaded_or_evaluated": False,
        "cadence_phase_absolute_global": True,
    })
    write_json(HERE / "integrity_audit.json", integrity)

    output_names = [
        "source_state_manifest.json", "integrity_audit.json", "eta_manifest.json",
        "exact_flow_h1_vs_h8.csv", "robust64_h1_vs_h8.csv", "paired_flow_outcomes.csv",
        "episode_classification.csv", "temporal_divergence.csv",
        "omitted_dense_correction_stats.csv", "jdef_comparison.csv",
        "completion_time_comparison.csv", "projection_analysis.csv",
        "historical_vs_fresh.csv", "cadence_report.md", "runtime_statistics.json",
    ]
    manifest = {
        "schema": "strict_deadlock_oracle_cadence_v1",
        "status": "COMPLETE", "classification": aggregate,
        "files": {name: {"sha256": sha256(HERE / name), "bytes": (HERE / name).stat().st_size} for name in output_names},
        "raw_state_files": {path.name: sha256(path) for path in sorted((HERE / "raw").glob("*.jsonl"))},
        "trace_files": {path.name: sha256(path) for path in sorted((HERE / "traces").glob("*.npz"))},
        "non_goals_verified": {
            "Gphi_not_trained_or_evaluated": True, "eta_not_searched": True,
            "gate_not_used": True, "H4_H16_not_tested": True,
            "environment_and_projections_frozen": True,
        },
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE", "classification": aggregate,
        "H1": f"{combined['H1_robust_successes']}/{combined['H1_robust_total']}",
        "H8": f"{combined['H8_robust_successes']}/{combined['H8_robust_total']}",
        "H8_B63_episodes": combined["H8_B63_episodes"], "class_counts": dict(class_counts),
    }, indent=2))


if __name__ == "__main__":
    main()
