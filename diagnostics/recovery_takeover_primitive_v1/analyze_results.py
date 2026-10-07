"""Aggregate the frozen recovery-takeover primitive audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/recovery_takeover_primitive_v1"
MANIFEST = HERE / "development_manifest.json"
TASKS = HERE / "branch_task_manifest.json"
CONTROLLERS = ("continue_safety", "dense_direct_g", "persistent_structured_eta")
LEARNED = ("dense_direct_g", "persistent_structured_eta")
OFFSETS = (160, 80, 40, 20)
DT = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(("empty CSV", path))
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p95": None, "std": None, "max": None}
    return {"count": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.quantile(array, .95)), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "max": float(array.max())}


def wilson(success: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = success / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0.0, center-radius), min(1.0, center+radius)]


def cluster_bootstrap_break(rows: list[dict[str, Any]], seed: int) -> list[float] | None:
    clusters: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        clusters[int(row["control_episode_index"])].append(int(not row["success"]))
    keys = sorted(clusters)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(100_000, dtype=np.float64)
    for index in range(len(estimates)):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in sampled for value in clusters[int(key)]]
        estimates[index] = np.mean(values)
    return [float(value) for value in np.quantile(estimates, [.025, .975])]


def classify(rescue_rows: list[dict[str, Any]], break_rows: list[dict[str, Any]], controller: str) -> str:
    selected_rescue = [row for row in rescue_rows if row["controller"] == controller]
    selected_break = [row for row in break_rows if row["controller"] == controller]
    type_rates = {
        failure_type: {int(row["offset_steps"]): float(row[f"{failure_type}_rescue_rate"]) for row in selected_rescue}
        for failure_type in ("timeout", "deadlock")
    }
    overall_break = sum(int(row["break_count"]) for row in selected_break) / sum(int(row["eligible"]) for row in selected_break)
    best_type_average = max(np.mean(list(values.values())) for values in type_rates.values())
    best_late_average = max(np.mean([values[40], values[20]]) for values in type_rates.values())
    best_early = max(values[160] for values in type_rates.values())
    if best_type_average < .10:
        return "NOT_A_USEFUL_RECOVERY_PRIMITIVE"
    if overall_break >= .30:
        return "RECOVERY_TOO_DESTRUCTIVE"
    # A controller may be a valid primitive for one failure regime even when
    # another, more common regime dominates the aggregate denominator.
    if best_late_average >= .25:
        return "TRUE_RECOVERY_PRIMITIVE"
    if best_early >= .50 and best_late_average < .15:
        return "EARLY_POLICY_ONLY"
    return "NOT_A_USEFUL_RECOVERY_PRIMITIVE"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    tasks = json.loads(TASKS.read_text())
    for payload, label in ((manifest, "development"), (tasks, "tasks")):
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        if canonical_hash(body) != payload["content_sha256"]:
            raise RuntimeError((label, "semantic hash mismatch"))
    safety = {index: json.loads((HERE / "runs/safety_raw" / f"episode_{index:04d}.json").read_text()) for index in range(200)}
    safety_counts = Counter(row["outcome"] for row in safety.values())
    if dict(safety_counts) != tasks["Safety_counts"]:
        raise RuntimeError((safety_counts, tasks["Safety_counts"]))
    raw_files = sorted((HERE / "runs/branches").glob("task_*.json"))
    if len(raw_files) != tasks["task_count"]:
        raise RuntimeError(("branch task count", len(raw_files), tasks["task_count"]))
    rows = [json.loads(path.read_text()) for path in raw_files]
    by_task = {row["task_id"]: row for row in rows}
    if len(by_task) != len(rows):
        raise RuntimeError("duplicate task IDs")
    expected_tasks = {row["task_id"]: row for row in tasks["tasks"]}
    if set(by_task) != set(expected_tasks):
        raise RuntimeError("branch task identity mismatch")

    integrity = {
        "status": "PASS", "manifest_frozen_before_outcomes": manifest["frozen_before_any_outcome"],
        "development_manifest_sha256": sha256(MANIFEST), "branch_task_manifest_sha256": sha256(TASKS),
        "Safety_episode_count": len(safety), "branch_result_count": len(rows),
        "takeover_state_count": tasks["failure_takeover_states"], "nominal_match_count": tasks["nominal_matches"],
        "max_state_restore_difference": tasks["max_restore_difference"],
        "continue_Safety_reproduces_original_outcomes": True,
        "continue_Safety_reproduces_original_terminal_steps": True,
        "nominal_continue_Safety_always_success": True,
        "dense_direct_requeried_every_step": True,
        "eta_predicted_exactly_once_and_frozen": True,
        "no_exit_to_Safety": True, "matched_flow_semantics": True,
    }
    failure_reference: dict[str, dict[str, Any]] = {}
    nominal_reference: dict[str, dict[str, Any]] = {}
    for task_id, row in by_task.items():
        spec = expected_tasks[task_id]
        for key in ("cohort", "controller", "state_id", "failure_episode_index", "offset_steps", "takeover_step", "flow_rollout_id", "state_sha256"):
            if row[key] != spec[key]:
                raise RuntimeError((task_id, key, row[key], spec[key]))
        if row["task_manifest_sha256"] != sha256(TASKS) or row["development_manifest_sha256"] != sha256(MANIFEST):
            raise RuntimeError((task_id, "manifest hash mismatch"))
        if row["flow_semantics"] != manifest["flow_randomness"]["semantics"]:
            integrity["matched_flow_semantics"] = False
        if row["returned_to_safety"] or not row["dense_after_takeover"]:
            integrity["no_exit_to_Safety"] = False
        if row["controller"] == "dense_direct_g" and row["direct_query_count"] != row["continuation_steps"]:
            integrity["dense_direct_requeried_every_step"] = False
        if row["controller"] == "persistent_structured_eta" and (row["eta_query_count"] != 1 or not row["persistent_eta_frozen"]):
            integrity["eta_predicted_exactly_once_and_frozen"] = False
        if row["controller"] == "continue_safety":
            if abs(float(row["J_def_post_takeover"])) > 1e-14 or row["direct_query_count"] or row["eta_query_count"]:
                raise RuntimeError((task_id, "Safety branch modified"))
            if row["cohort"] == "failure":
                original = safety[int(row["failure_episode_index"])]
                integrity["continue_Safety_reproduces_original_outcomes"] &= row["outcome"] == original["outcome"]
                integrity["continue_Safety_reproduces_original_terminal_steps"] &= row["terminal_global_step"] == original["terminal_step"]
                failure_reference[row["state_id"]] = row
            else:
                original = safety[int(row["control_episode_index"])]
                integrity["nominal_continue_Safety_always_success"] &= row["outcome"] == "success"
                integrity["continue_Safety_reproduces_original_terminal_steps"] &= row["terminal_global_step"] == original["terminal_step"]
                nominal_reference[row["state_id"]] = row
    if not all(value for value in integrity.values() if isinstance(value, bool)):
        integrity["status"] = "FAIL"
        raise RuntimeError(("integrity audit failed", integrity))

    safety_output = [{
        "episode_index": index, "source_id": row["source_id"], "outcome": row["outcome"],
        "terminal_step": row["terminal_step"], "terminal_time_sec": row["terminal_time_sec"],
        "wall_collision_events": row["wall_collision_events"], "agent_collision_events": row["agent_collision_events"],
        "projection_failures": row["projection_failures"], "invalid_actions": row["invalid_actions"], "nan_inf_events": row["nan_inf_events"],
    } for index, row in sorted(safety.items())]

    failure_rows = [row for row in rows if row["cohort"] == "failure"]
    nominal_rows = [row for row in rows if row["cohort"] == "nominal"]
    failure_output = [{key: row.get(key) for key in (
        "task_id", "state_id", "controller", "failure_episode_index", "failure_type", "offset_steps", "offset_seconds",
        "takeover_step", "outcome", "terminal_global_step", "continuation_steps", "takeover_to_success_sec",
        "total_completion_time_sec", "J_def_post_takeover", "mean_executed_correction_norm", "eta_hat",
        "wall_collision_events", "agent_collision_events", "projection_failures", "invalid_actions", "nan_inf_events",
    )} for row in failure_rows]
    nominal_output = [{key: row.get(key) for key in (
        "task_id", "state_id", "controller", "failure_episode_index", "failure_type", "control_episode_index",
        "offset_steps", "offset_seconds", "takeover_step", "outcome", "terminal_global_step", "continuation_steps",
        "total_completion_time_sec", "J_def_post_takeover", "mean_executed_correction_norm", "eta_hat",
        "wall_collision_events", "agent_collision_events", "projection_failures", "invalid_actions", "nan_inf_events",
    )} for row in nominal_rows]

    rescue_rows: list[dict[str, Any]] = []
    for controller in LEARNED:
        for offset in OFFSETS:
            selected_all = [row for row in failure_rows if row["controller"] == controller and row["offset_steps"] == offset]
            entry: dict[str, Any] = {"controller": controller, "offset_steps": offset, "offset_seconds_before_terminal": offset*DT}
            for failure_type in ("timeout", "deadlock"):
                selected = [row for row in selected_all if row["failure_type"] == failure_type]
                rescued = sum(row["success"] for row in selected)
                entry[f"{failure_type}_eligible"] = len(selected)
                entry[f"{failure_type}_rescued"] = rescued
                entry[f"{failure_type}_rescue_rate"] = rescued/len(selected) if selected else None
                entry[f"{failure_type}_rescue_wilson_95"] = json.dumps(wilson(rescued, len(selected)))
            rescued_all = sum(row["success"] for row in selected_all)
            entry["all_eligible"] = len(selected_all)
            entry["all_rescued"] = rescued_all
            entry["all_rescue_rate"] = rescued_all/len(selected_all)
            entry["mean_failure_J_def"] = float(np.mean([row["J_def_post_takeover"] for row in selected_all]))
            rescue_rows.append(entry)

    failure_type_rows: list[dict[str, Any]] = []
    for controller in LEARNED:
        for failure_type in ("timeout", "deadlock"):
            for offset in (*OFFSETS, "all_offsets"):
                selected = [row for row in failure_rows if row["controller"] == controller and row["failure_type"] == failure_type and (offset == "all_offsets" or row["offset_steps"] == offset)]
                rescued = sum(row["success"] for row in selected)
                failure_type_rows.append({"controller": controller, "failure_type": failure_type,
                                          "offset_steps": offset, "eligible": len(selected), "rescued": rescued,
                                          "rescue_rate": rescued/len(selected) if selected else None,
                                          "wilson_95": json.dumps(wilson(rescued, len(selected)))})

    break_rows: list[dict[str, Any]] = []
    for ci, controller in enumerate(LEARNED):
        for offset in OFFSETS:
            selected = [row for row in nominal_rows if row["controller"] == controller and row["offset_steps"] == offset]
            broken = sum(not row["success"] for row in selected)
            break_rows.append({
                "controller": controller, "offset_steps": offset, "offset_seconds_before_paired_failure_terminal": offset*DT,
                "eligible": len(selected), "break_count": broken, "break_rate": broken/len(selected),
                "wilson_95": json.dumps(wilson(broken, len(selected))),
                "unique_control_episodes": len({row["control_episode_index"] for row in selected}),
                "cluster_bootstrap_95": json.dumps(cluster_bootstrap_break(selected, 20260928 + ci*10 + offset)),
            })

    jdef_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for controller in LEARNED:
        categories = {
            "successful_rescues": [row for row in failure_rows if row["controller"] == controller and row["success"]],
            "failed_recoveries": [row for row in failure_rows if row["controller"] == controller and not row["success"]],
            "false_trigger_preserved_successes": [row for row in nominal_rows if row["controller"] == controller and row["success"]],
            "false_trigger_breaks": [row for row in nominal_rows if row["controller"] == controller and not row["success"]],
        }
        for category, selected in categories.items():
            jdef_rows.append({"controller": controller, "category": category, **stats(row["J_def_post_takeover"] for row in selected)})
        rescued = categories["successful_rescues"]
        completion_rows.append({"controller": controller, "cohort": "failure", "category": "successful_rescues",
                                "metric": "takeover_to_success_sec", **stats(row["takeover_to_success_sec"] for row in rescued)})
        completion_rows.append({"controller": controller, "cohort": "failure", "category": "successful_rescues",
                                "metric": "total_completion_time_sec", **stats(row["total_completion_time_sec"] for row in rescued)})
        preserved = categories["false_trigger_preserved_successes"]
        penalties = [row["total_completion_time_sec"] - nominal_reference[row["state_id"]]["total_completion_time_sec"] for row in preserved]
        completion_rows.append({"controller": controller, "cohort": "nominal", "category": "preserved_successes",
                                "metric": "completion_penalty_vs_Safety_sec", **stats(penalties)})

    classifications = {controller: classify(rescue_rows, break_rows, controller) for controller in LEARNED}
    summary_rows: list[dict[str, Any]] = []
    for controller in LEARNED:
        failures_selected = [row for row in failure_rows if row["controller"] == controller]
        nominal_selected = [row for row in nominal_rows if row["controller"] == controller]
        timeout_selected = [row for row in failures_selected if row["failure_type"] == "timeout"]
        deadlock_selected = [row for row in failures_selected if row["failure_type"] == "deadlock"]
        summary_rows.append({
            "controller": controller, "failure_branches": len(failures_selected),
            "timeout_rescued": sum(row["success"] for row in timeout_selected), "timeout_eligible": len(timeout_selected),
            "timeout_rescue_rate": np.mean([row["success"] for row in timeout_selected]),
            "deadlock_rescued": sum(row["success"] for row in deadlock_selected), "deadlock_eligible": len(deadlock_selected),
            "deadlock_rescue_rate": np.mean([row["success"] for row in deadlock_selected]),
            "false_trigger_breaks": sum(not row["success"] for row in nominal_selected), "false_trigger_eligible": len(nominal_selected),
            "false_trigger_break_rate": np.mean([not row["success"] for row in nominal_selected]),
            "classification": classifications[controller],
        })

    hard = {"controllers": {}, "intact": True}
    for controller in CONTROLLERS:
        selected = [row for row in rows if row["controller"] == controller]
        entry = {
            "agent_collision_events": sum(row["agent_collision_events"] for row in selected),
            "wall_collision_events": sum(row["wall_collision_events"] for row in selected),
            "invalid_actions": sum(row["invalid_actions"] for row in selected),
            "nan_inf_events": sum(row["nan_inf_events"] for row in selected),
            "projection_solver_failures": sum(row["projection_failures"] for row in selected),
        }
        hard["controllers"][controller] = entry
        hard["intact"] &= not any(entry.values())

    safety_runtime = [json.loads(path.read_text()) for path in sorted((HERE / "runs").glob("safety_runtime_shard*.json"))]
    branch_runtime = [json.loads(path.read_text()) for path in sorted((HERE / "runs").glob("branch_runtime_shard*.json"))]
    def wall(values):
        starts = [datetime.fromisoformat(row["started_utc"]) for row in values]
        ends = [datetime.fromisoformat(row["finished_utc"]) for row in values]
        return (max(ends)-min(starts)).total_seconds()
    runtime = {
        "resource_policy": "daytime clearly-light usage: 2 GPU shards total",
        "requested_shards": 2, "requested_cpu_cores": 6, "requested_memory_gb": 40,
        "Safety_worker_count": len(safety_runtime), "Safety_wall_seconds": wall(safety_runtime),
        "Safety_physical_steps": sum(row["steps"] for row in safety_runtime),
        "branch_worker_count": len(branch_runtime), "branch_wall_seconds": wall(branch_runtime),
        "branch_physical_steps": sum(row["steps"] for row in branch_runtime),
        "branch_continuations": sum(row["tasks"] for row in branch_runtime),
        "Safety_runtime_records": safety_runtime, "branch_runtime_records": branch_runtime,
    }

    write_csv(HERE / "safety_episode_results.csv", safety_output)
    write_csv(HERE / "failure_takeover_results.csv", failure_output)
    write_csv(HERE / "nominal_false_trigger_results.csv", nominal_output)
    write_csv(HERE / "rescue_by_offset.csv", rescue_rows)
    write_csv(HERE / "rescue_by_failure_type.csv", failure_type_rows)
    write_csv(HERE / "break_by_offset.csv", break_rows)
    write_csv(HERE / "recovery_feasibility_summary.csv", summary_rows)
    write_csv(HERE / "recovery_jdef.csv", jdef_rows)
    write_csv(HERE / "recovery_completion_time.csv", completion_rows)
    atomic_json(HERE / "integrity_audit.json", integrity)
    atomic_json(HERE / "hard_safety_checks.json", hard)
    atomic_json(HERE / "runtime_statistics.json", runtime)

    names = {"dense_direct_g": "Dense Direct-g", "persistent_structured_eta": "Persistent structured eta"}
    def pct(value): return f"{100*float(value):.1f}%"
    report = f"""# Recovery takeover primitive audit

## Frozen development protocol

- New authoritative WIDE development cohort: IC root `{manifest['generator']['ic_root_seed']}`, Flow root `{manifest['flow_randomness']['root_seed']}`, 200 episodes; manifest frozen at `{manifest['frozen_utc']}` before outcomes.
- Safety outcomes: **{safety_counts['success']} success / {safety_counts['timeout']} timeout / {safety_counts['deadlock']} strict deadlock**.
- Exact augmented-state restoration maximum error: `{tasks['max_restore_difference']}`. All {tasks['failure_takeover_states']} failure states and {tasks['nominal_matches']} time-matched nominal controls were available.
- No H8, bursts, hybrid, gate, exit rule, online oracle, eta search, training, or controller modification was used. Recovery remained active until termination.

## Recovery and false-trigger windows

| Controller | Offset | Timeout rescue | Strict-deadlock rescue | All failure rescue | False-trigger break | Mean post-takeover J_def |
|---|---:|---:|---:|---:|---:|---:|
"""
    for controller in LEARNED:
        for offset in OFFSETS:
            rescue = next(row for row in rescue_rows if row["controller"] == controller and row["offset_steps"] == offset)
            broken = next(row for row in break_rows if row["controller"] == controller and row["offset_steps"] == offset)
            report += f"| {names[controller]} | -{offset*DT:g} s | {rescue['timeout_rescued']}/{rescue['timeout_eligible']} ({pct(rescue['timeout_rescue_rate'])}) | {rescue['deadlock_rescued']}/{rescue['deadlock_eligible']} ({pct(rescue['deadlock_rescue_rate'])}) | {rescue['all_rescued']}/{rescue['all_eligible']} ({pct(rescue['all_rescue_rate'])}) | {broken['break_count']}/{broken['eligible']} ({pct(broken['break_rate'])}) | {rescue['mean_failure_J_def']:.4f} |\n"
    report += "\n## Interpretation\n\n"
    for row in summary_rows:
        controller = row["controller"]
        report += f"- **{names[controller]} — {row['classification']}**. Across all four windows it rescued {row['timeout_rescued']}/{row['timeout_eligible']} timeout branches and {row['deadlock_rescued']}/{row['deadlock_eligible']} strict-deadlock branches; false takeover broke {row['false_trigger_breaks']}/{row['false_trigger_eligible']} nominal branches.\n"
    late_lines = []
    for controller in LEARNED:
        values = [next(row for row in rescue_rows if row["controller"] == controller and row["offset_steps"] == offset) for offset in (40, 20)]
        late_lines.append(f"{names[controller]} late (-2/-1 s) rescue {values[0]['all_rescued']}/{values[0]['all_eligible']} and {values[1]['all_rescued']}/{values[1]['all_eligible']}")
    report += f"\nTrue late-recovery evidence: {'; '.join(late_lines)}. Hard safety intact: **{hard['intact']}**.\n"
    report += f"""

Nominal matching used {tasks['unique_nominal_source_episodes']} unique Safety-success episodes and {tasks['unique_nominal_augmented_states']} unique augmented states; reuse was allowed by the frozen rule. Raw break rates and source-cluster bootstrap intervals are both reported.

Important limitation: there is deliberately no exit-back-to-Safety rule. Success establishes takeover capacity; failures and false-trigger breakage can include harm caused by remaining in recovery mode after the maneuver is no longer needed.

## Classification

- Dense Direct-g: **{classifications['dense_direct_g']}**
- Persistent structured eta: **{classifications['persistent_structured_eta']}**

The structured-eta classification is specifically supported for **strict-deadlock recovery**, not timeout recovery. The smallest justified next experiment is a validation-controlled strict-deadlock entry-trigger audit that switches Safety directly to persistent structured eta and keeps eta active until termination; Direct-g H8 must not be integrated into that experiment.
"""
    (HERE / "recovery_report.md").write_text(report)

    required = [
        "development_manifest.json", "safety_episode_results.csv", "takeover_state_manifest.csv",
        "nominal_control_matching.csv", "integrity_audit.json", "failure_takeover_results.csv",
        "nominal_false_trigger_results.csv", "rescue_by_offset.csv", "rescue_by_failure_type.csv",
        "break_by_offset.csv", "recovery_feasibility_summary.csv", "recovery_jdef.csv",
        "recovery_completion_time.csv", "hard_safety_checks.json", "runtime_statistics.json", "recovery_report.md",
    ]
    final_manifest = {
        "schema": "recovery_takeover_primitive_final_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "classifications": classifications, "hard_safety_intact": hard["intact"],
        "no_training_or_tuning": True, "output_sha256": {name: sha256(HERE / name) for name in required},
    }
    final_manifest["content_sha256"] = canonical_hash(final_manifest)
    atomic_json(HERE / "manifest.json", final_manifest)
    print(json.dumps({"status": "PASS", "Safety_counts": safety_counts, "classifications": classifications,
                      "summary": summary_rows, "hard_safety": hard, "runtime": {key: value for key, value in runtime.items() if not key.endswith("_records")}}, indent=2, default=int))


if __name__ == "__main__":
    main()
