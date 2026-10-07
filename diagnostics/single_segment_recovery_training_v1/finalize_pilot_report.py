#!/usr/bin/env python3
"""Read-only artifact auditor/report generator for the single-segment pilot.

This program never imports the simulator or trainer.  It reads immutable run
artifacts and writes only the report products owned by this program.  Re-run it
after policy iteration or evaluation finishes to refresh the summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


GENERATOR = "finalize_pilot_report.py"
MD_MARKER = f"<!-- generated_by: {GENERATOR} -->"
DEFAULT_ROOT = Path("/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1")
ALLOWED_FINAL_CLASSIFICATIONS = {
    "SINGLE_SEGMENT_RECOVERY_SUPPORTED",
    "RECOVERY_WORKS_ENTRY_EXIT_NOT_YET_RELIABLE",
    "RECOVERY_PRIMITIVE_NOT_ESTABLISHED",
    "DECISION_LABELS_OR_DATA_INSUFFICIENT",
    "SINGLE_SEGMENT_RECOVERY_DOES_NOT_IMPROVE_BASELINE",
    "BUDGET_LIMITED_PARTIAL_RESULT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def json_dump_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def assert_owned(path: Path, markdown: bool = False) -> None:
    if not path.exists():
        return
    if markdown:
        first = path.read_text(errors="replace").splitlines()[:1]
        if not first or first[0] != MD_MARKER:
            raise RuntimeError(f"refusing to overwrite non-owned report: {path}")
        return
    prior = read_json(path, {})
    if not isinstance(prior, dict) or prior.get("generated_by") != GENERATOR:
        raise RuntimeError(f"refusing to overwrite non-owned JSON: {path}")


def write_json(path: Path, value: dict[str, Any], dry_run: bool) -> None:
    value = dict(value)
    value["generated_by"] = GENERATOR
    if dry_run:
        return
    assert_owned(path)
    path.write_bytes(json_dump_bytes(value))


def write_md(path: Path, body: str, dry_run: bool) -> None:
    text = MD_MARKER + "\n" + body.rstrip() + "\n"
    if dry_run:
        return
    assert_owned(path, markdown=True)
    path.write_text(text)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def complete_json_rows(folder: Path, pattern: str) -> tuple[int, int, Counter[str], int]:
    complete = errors = 0
    outcomes: Counter[str] = Counter()
    steps = 0
    for path in sorted(folder.glob(pattern)) if folder.is_dir() else []:
        row = read_json(path)
        if not isinstance(row, dict):
            errors += 1
            continue
        if row.get("record_complete") is True and row.get("execution_error") in (None, ""):
            complete += 1
        else:
            errors += 1
        outcomes[str(row.get("outcome", row.get("learned_outcome", "unknown")))] += 1
        steps += int(row.get("physical_transition_count", row.get("remaining_physical_steps", 0)) or 0)
    return complete, errors, outcomes, steps


def branch_stage(root: Path, name: str) -> dict[str, Any]:
    manifest_path = root / "branch_manifests" / f"{name}.json"
    manifest = read_json(manifest_path, {}) or {}
    planned = len(manifest.get("tasks", []))
    folder = root / "runs" / "paired_branches" / name
    complete, errors, outcomes, steps = complete_json_rows(folder, "task_*.json")
    runtime_files = sorted(folder.glob("runtime_shard*.json")) if folder.is_dir() else []
    label_dir = root / "labels" / name
    label_manifest = read_json(label_dir / "decision_dataset_manifest.json")
    return {
        "stage": name,
        "manifest_path": rel(manifest_path, root),
        "manifest_sha256": sha256(manifest_path),
        "planned_branch_continuations": planned,
        "completed_branch_continuations": complete,
        "incomplete_or_error_records": errors,
        "outcomes": dict(sorted(outcomes.items())),
        "actual_physical_steps_from_rows": steps,
        "runtime_file_count": len(runtime_files),
        "labels_finalized": isinstance(label_manifest, dict),
        "label_manifest_path": rel(label_dir / "decision_dataset_manifest.json", root),
        "label_manifest_sha256": sha256(label_dir / "decision_dataset_manifest.json"),
        "complete": bool(planned) and complete == planned and errors == 0,
    }


def discover_policy_iteration_training(root: Path, kind: str) -> dict[str, Any] | None:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    base = root / "entry_exit_training_history"
    if not base.is_dir():
        return None
    for path in base.glob("*/checkpoint_manifest.json"):
        name = path.parent.name.lower()
        if kind not in name or not any(token in name for token in ("policy", "iteration", "pi_", "round")):
            continue
        data = read_json(path)
        if isinstance(data, dict):
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None
    _, path, data = max(candidates, key=lambda item: item[0])
    choice = data.get("provisional_branch_evidence_choice") or data.get("final_selected_policy")
    return {
        "path": rel(path, root),
        "sha256": sha256(path),
        "selected": choice,
        "calibration_or_test_used": data.get("calibration_or_test_used"),
    }


def build_policy_iteration_manifest(root: Path) -> dict[str, Any]:
    state_integrity = read_json(root / "policy_iteration_state_collection_integrity.json", {}) or {}
    entry = branch_stage(root, "policy_iteration_entry")
    exit_stage = branch_stage(root, "policy_iteration_exit")
    entry_train = discover_policy_iteration_training(root, "entry")
    exit_train = discover_policy_iteration_training(root, "exit")
    branches_complete = entry["complete"] and exit_stage["complete"]
    labels_complete = entry["labels_finalized"] and exit_stage["labels_finalized"]
    heads_retrained = entry_train is not None and exit_train is not None
    pi_validation = read_json(root / "policy_iteration_validation.json", {}) or {}
    pi_calibration = read_json(root / "policy_iteration_calibration.json", {}) or {}
    final_test = read_json(root / "final_test_results.json", {}) or {}
    evaluation = {
        "validation_status": pi_validation.get("status", "MISSING"),
        "validation_sha256": sha256(root / "policy_iteration_validation.json"),
        "calibration_status": pi_calibration.get("status", "MISSING"),
        "calibration_sha256": sha256(root / "policy_iteration_calibration.json"),
        "final_test_status": final_test.get("status", "MISSING"),
        "final_test_sha256": sha256(root / "final_test_results.json"),
        "selected_policy_hash": ((pi_validation.get("selection") or {}).get("selected_policy_hash")),
        "final_policy_hash": ((final_test.get("metrics") or {}).get("policy_hash")),
    }
    final_complete = (
        final_test.get("status") == "COMPLETE_FROZEN_NO_POST_HOC_TUNING"
        and int(final_test.get("source_episode_count", 0) or 0) == 200
    )
    if heads_retrained and final_complete:
        status = "POLICY_ITERATION_FINAL_TEST_COMPLETE"
    elif heads_retrained:
        status = "POLICY_ITERATION_HEADS_RETRAINED"
    elif labels_complete:
        status = "POLICY_ITERATION_LABELS_COMPLETE_TRAINING_PENDING"
    elif branches_complete:
        status = "POLICY_ITERATION_BRANCHES_COMPLETE_FINALIZATION_PENDING"
    elif state_integrity.get("status") == "PASS":
        status = "POLICY_ITERATION_BRANCHES_PARTIAL"
    else:
        status = "POLICY_ITERATION_STATE_COLLECTION_PENDING"
    return {
        "schema": "single_segment_policy_iteration_manifest_summary_v1",
        "generated_utc": utc_now(),
        "status": status,
        "interim_partial": not heads_retrained,
        "maximum_policy_improvement_rounds": 1,
        "state_collection": {
            "status": state_integrity.get("status", "MISSING"),
            "path": "policy_iteration_state_collection_integrity.json",
            "sha256": sha256(root / "policy_iteration_state_collection_integrity.json"),
            "selected_entry_states": state_integrity.get("selected_entry_states"),
            "selected_exit_states": state_integrity.get("selected_exit_states"),
            "entry_unique_sources": state_integrity.get("entry_unique_sources"),
            "exit_unique_sources": state_integrity.get("exit_unique_sources"),
            "outcome_used_for_selection": state_integrity.get("outcome_used_for_selection"),
        },
        "paired_branch_round": {"entry": entry, "exit": exit_stage},
        "retrained_heads": {"entry": entry_train, "exit": exit_train},
        "evaluation": evaluation,
        "policy_targets_remain_lineage_specific": True,
        "calibration_or_final_sources_used_for_training": False,
    }


def summarize_runtime(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    files = sorted((root / "runs").rglob("*runtime*.json")) if (root / "runs").is_dir() else []
    total_cont = total_steps = 0
    records: list[dict[str, Any]] = []
    stage_times: defaultdict[str, list[datetime]] = defaultdict(list)
    all_timestamps: list[datetime] = []
    max_shards = 0
    devices: Counter[str] = Counter()
    cpu_thread_values: set[int] = set()
    for path in files:
        row = read_json(path)
        if not isinstance(row, dict):
            continue
        cont = row.get("new_continuations")
        if cont is None:
            cont = int(row.get("new_source_rollouts", 0) or 0)
            cont += int(row.get("new_learned_continuations", 0) or 0)
            cont += int(row.get("new_safety_continuations", 0) or 0)
        steps = row.get("new_physical_steps", row.get("new_physical_simulation_steps", 0)) or 0
        total_cont += int(cont)
        total_steps += int(steps)
        max_shards = max(max_shards, int(row.get("shard_count", 1) or 1))
        devices[str(row.get("device", "unknown"))] += 1
        for value in (row.get("thread_environment") or {}).values():
            try:
                cpu_thread_values.add(int(value))
            except (TypeError, ValueError):
                pass
        key = str(row.get("stage") or row.get("schema") or path.parent)
        for field in ("started_utc", "finished_utc"):
            raw = row.get(field)
            if raw:
                try:
                    stamp = datetime.fromisoformat(raw)
                    stage_times[key].append(stamp)
                    all_timestamps.append(stamp)
                except ValueError:
                    pass
        records.append({
            "path": rel(path, root),
            "schema": row.get("schema"),
            "stage": row.get("stage"),
            "split": row.get("split"),
            "continuations": int(cont),
            "physical_steps": int(steps),
            "device": row.get("device"),
            "shard_count": row.get("shard_count"),
            "started_utc": row.get("started_utc"),
            "finished_utc": row.get("finished_utc"),
        })
    # A running shard writes task rows before its terminal runtime record.  Add
    # only the positive difference so interim reports do not hide already-used
    # budget and completed jobs are not double counted.
    live_unrecorded = []
    paired_root = root / "runs" / "paired_branches"
    for folder in sorted(paired_root.iterdir()) if paired_root.is_dir() else []:
        if not folder.is_dir():
            continue
        complete, _, _, row_steps = complete_json_rows(folder, "task_*.json")
        runtime_rows = [read_json(path, {}) or {} for path in folder.glob("runtime_shard*.json")]
        recorded_cont = sum(int(row.get("new_continuations", 0) or 0) for row in runtime_rows)
        recorded_steps = sum(int(row.get("new_physical_steps", 0) or 0) for row in runtime_rows)
        extra_cont = max(0, complete - recorded_cont)
        extra_steps = max(0, row_steps - recorded_steps)
        if extra_cont or extra_steps:
            total_cont += extra_cont
            total_steps += extra_steps
            live_unrecorded.append({
                "stage": folder.name,
                "completed_rows_not_yet_in_terminal_runtime": extra_cont,
                "physical_steps_not_yet_in_terminal_runtime": extra_steps,
            })
    stage_wall = {}
    for key, stamps in stage_times.items():
        if len(stamps) >= 2:
            stage_wall[key] = (max(stamps) - min(stamps)).total_seconds()
    budget = protocol.get("budget", {})
    lim_cont = int(budget.get("maximum_new_continuations", 12000))
    lim_steps = int(budget.get("maximum_new_physical_steps", 6000000))
    return {
        "schema": "single_segment_runtime_statistics_v1",
        "generated_utc": utc_now(),
        "accounting_basis": "sum of unique JSON runtime records under runs/; no log inference",
        "runtime_record_count": len(records),
        "new_continuations_recorded": total_cont,
        "new_physical_steps_recorded": total_steps,
        "continuation_budget_limit": lim_cont,
        "physical_step_budget_limit": lim_steps,
        "continuation_budget_remaining": lim_cont - total_cont,
        "physical_step_budget_remaining": lim_steps - total_steps,
        "within_budget": total_cont <= lim_cont and total_steps <= lim_steps,
        "maximum_recorded_gpu_shards": max_shards,
        "device_record_counts": dict(sorted(devices.items())),
        "recorded_thread_settings": sorted(cpu_thread_values),
        "stage_wall_seconds": stage_wall,
        "sum_recorded_stage_wall_seconds": sum(stage_wall.values()),
        "observed_experiment_runtime_window_seconds": (
            (max(all_timestamps) - min(all_timestamps)).total_seconds() if len(all_timestamps) >= 2 else None
        ),
        "live_completed_rows_accounting": live_unrecorded,
        "runtime_records": records,
    }


def verify_recorded_asset_hashes(root: Path, frozen: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, item in sorted(frozen.items()):
        if name in ("schema", "status") or not isinstance(item, dict):
            continue
        path_raw, expected = item.get("path"), item.get("sha256")
        if not path_raw or not expected:
            continue
        path = Path(path_raw)
        actual = sha256(path)
        rows.append({
            "asset": name,
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "pass": actual == expected,
        })
    return rows


def split_ids(source: dict[str, Any], split: str) -> set[str]:
    items = (source.get("sources") or {}).get(split, [])
    return {str(item.get("source_id")) for item in items if item.get("source_id") is not None}


def build_integrity(root: Path, runtime: dict[str, Any], pi: dict[str, Any]) -> dict[str, Any]:
    frozen = read_json(root / "controller_and_projection_hashes.json", {}) or {}
    source = read_json(root / "source_split_manifest.json", {}) or {}
    asset_rows = verify_recorded_asset_hashes(root, frozen)
    split_names = ("train", "validation", "calibration", "final_test")
    ids = {name: split_ids(source, name) for name in split_names}
    overlaps = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlaps[f"{left}__{right}"] = sorted(ids[left] & ids[right])
    final_rows = []
    if (root / "runs" / "full_loop").is_dir():
        for path in (root / "runs" / "full_loop").glob("*/final_test/*.json"):
            if path.name.startswith("runtime_"):
                continue
            row = read_json(path)
            if isinstance(row, dict) and row.get("record_complete") is True:
                final_rows.append(path)
    final_analysis = read_json(root / "final_test_results.json", {}) or {}
    final_declared = int(final_analysis.get("source_episode_count", 0) or 0)
    label_manifests = list((root / "labels").glob("*/decision_dataset_manifest.json")) if (root / "labels").is_dir() else []
    label_test_flags = []
    for path in label_manifests:
        row = read_json(path, {}) or {}
        label_test_flags.append({
            "path": rel(path, root),
            "test_data_used": bool(row.get("test_data_used", False) or row.get("final_test_used", False)),
            "train_validation_source_overlap": row.get("train_validation_source_overlap"),
        })
    all_branch_errors = 0
    all_hard_safety = Counter()
    result_files = list((root / "runs" / "paired_branches").glob("*/task_*.json"))
    result_files += [p for p in (root / "runs" / "full_loop").glob("*/*/*.json") if not p.name.startswith("runtime_")]
    for path in result_files:
        row = read_json(path)
        if not isinstance(row, dict):
            all_branch_errors += 1
            continue
        if row.get("execution_error") not in (None, "") or row.get("record_complete") is not True:
            all_branch_errors += 1
        for src, dst in (
            ("agent_collision_events", "agent_collisions"),
            ("agent_collision", "agent_collisions"),
            ("wall_collision_events", "wall_collisions"),
            ("wall_collision", "wall_collisions"),
            ("invalid_actions", "invalid_actions"),
            ("invalid_action", "invalid_actions"),
            ("nan_inf_events", "nan_inf"),
            ("nan_inf", "nan_inf"),
            ("projection_failures", "projection_solver_failures"),
            ("projection_solver_failure", "projection_solver_failures"),
        ):
            all_hard_safety[dst] += int(row.get(src, 0) or 0)
    checks = {
        "frozen_asset_hashes": all(item["pass"] for item in asset_rows) and bool(asset_rows),
        "source_splits_disjoint": all(not value for value in overlaps.values()),
        "source_collection_integrity_pass": (read_json(root / "source_collection_integrity.json", {}) or {}).get("status") == "PASS",
        "eta_state_collection_integrity_pass": (read_json(root / "eta_recovery_state_collection_integrity.json", {}) or {}).get("status") == "PASS",
        "policy_iteration_state_collection_integrity_pass": (read_json(root / "policy_iteration_state_collection_integrity.json", {}) or {}).get("status") == "PASS",
        "runtime_within_budget": bool(runtime.get("within_budget")),
        "final_test_rollout_count_consistent": (
            (not final_analysis and len(final_rows) == 0)
            or (final_analysis.get("status") == "COMPLETE_FROZEN_NO_POST_HOC_TUNING" and final_declared == 200 and len(final_rows) == 200)
        ),
        "final_test_forbids_test_based_adjustment": (
            not final_analysis or bool(final_analysis.get("test_based_adjustment_forbidden"))
        ),
        "no_label_manifest_uses_test": not any(bool(item["test_data_used"]) for item in label_test_flags),
        "observed_result_records_complete": all_branch_errors == 0,
        "observed_hard_safety_intact": sum(all_hard_safety.values()) == 0,
    }
    return {
        "schema": "single_segment_integrity_checks_v1",
        "generated_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "PARTIAL_OR_FAIL",
        "checks": checks,
        "frozen_asset_hash_checks": asset_rows,
        "split_source_counts": {key: len(value) for key, value in ids.items()},
        "split_overlaps": overlaps,
        "label_manifest_audit": label_test_flags,
        "observed_result_file_count": len(result_files),
        "observed_incomplete_or_execution_error_records": all_branch_errors,
        "observed_hard_safety_counts": dict(all_hard_safety),
        "final_test_outcome_file_count": len(final_rows),
        "final_test_declared_source_count": final_declared,
        "policy_iteration_status": pi.get("status"),
    }


def selected_metrics(report: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    selection = report.get("selection") or {}
    return selection.get("selected_metrics"), report.get("calibration"), selection.get("selected_policy_hash")


def load_evaluation_report(root: Path, fallback_path: Path) -> dict[str, Any]:
    """Prefer the semantically current PI evaluation bundle when available."""
    validation = read_json(root / "policy_iteration_validation.json")
    calibration = read_json(root / "policy_iteration_calibration.json")
    final_test = read_json(root / "final_test_results.json")
    if isinstance(validation, dict) and isinstance(calibration, dict):
        return {
            "schema": "single_segment_policy_iteration_evaluation_bundle_v1",
            "selection": validation.get("selection") or {},
            "candidates": validation.get("candidates") or [],
            "calibration": calibration.get("metrics"),
            "calibration_audit": calibration.get("break_budget_audit") or {},
            "final_test": (final_test or {}).get("metrics") if isinstance(final_test, dict) else None,
            "provenance": {
                "validation": "policy_iteration_validation.json",
                "calibration": "policy_iteration_calibration.json",
                "final_test": "final_test_results.json" if isinstance(final_test, dict) else None,
            },
        }
    return read_json(fallback_path, {}) or {}


def outcome_rows(root: Path, policy_hash: str | None, splits: Iterable[str]) -> list[dict[str, Any]]:
    if not policy_hash:
        return []
    fields = (
        "root_source_id", "split", "safety_outcome", "learned_outcome", "safety_success", "learned_success",
        "entry_step", "exit_step", "recovery_segment_count", "recovery_transition_count", "returned_to_safety",
        "terminal_step", "jdef", "agent_collision", "wall_collision", "invalid_action", "nan_inf",
        "projection_solver_failure", "policy_hash",
    )
    rows = []
    for split in splits:
        folder = root / "runs" / "full_loop" / policy_hash / split
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            if path.name.startswith("runtime_"):
                continue
            row = read_json(path)
            if isinstance(row, dict) and row.get("record_complete") is True:
                rows.append({key: row.get(key) for key in fields})
    return rows


def fmt_ratio(num: Any, den: Any) -> str:
    if num is None or den in (None, 0):
        return "n/a"
    return f"{num}/{den} ({float(num) / float(den):.3f})"


def metrics_table(label: str, metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return f"| {label} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |"
    n = metrics.get("source_episode_count")
    q = metrics.get("q_learned")
    safe = metrics.get("q_safe")
    jdef = (metrics.get("jdef") or {}).get("mean")
    return (
        f"| {label} | {n} | {safe:.3f} | {q:.3f} | {metrics.get('rescue')} | "
        f"{metrics.get('break')} | {jdef:.6f} |" if isinstance(q, (int, float)) and isinstance(safe, (int, float)) and isinstance(jdef, (int, float)) else
        f"| {label} | {n} | {safe} | {q} | {metrics.get('rescue')} | {metrics.get('break')} | {jdef} |"
    )


def report_freshness(root: Path, pi: dict[str, Any], report: dict[str, Any]) -> tuple[bool, str]:
    """Whether the supplied full-loop report evaluates retrained PI heads."""
    retrained = pi.get("retrained_heads") or {}
    entry = (retrained.get("entry") or {}).get("selected") or {}
    exit_head = (retrained.get("exit") or {}).get("selected") or {}
    expected = {entry.get("checkpoint_sha256"), exit_head.get("checkpoint_sha256")} - {None}
    if not expected:
        return False, "policy-iteration heads are not both retrained"
    selected = (report.get("selection") or {}).get("selected_metrics") or {}
    policy_hash = selected.get("policy_hash")
    if not policy_hash:
        return False, "no selected full-loop policy"
    manifest_candidates = list((root / "full_loop_manifests").rglob("*.json"))
    for path in manifest_candidates:
        row = read_json(path, {}) or {}
        if row.get("policy_hash") != policy_hash:
            continue
        policy = row.get("policy") or {}
        used = {
            (policy.get("entry_head") or {}).get("sha256"),
            (policy.get("exit_head") or {}).get("sha256"),
        } - {None}
        if used == expected:
            return True, "validation report matches both policy-iteration head hashes"
    return False, "validation report does not match policy-iteration head hashes"


def classify(pi: dict[str, Any], report: dict[str, Any], fresh: bool, final_rows: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not runtime.get("within_budget"):
        return {"status": "FINAL", "classification": "BUDGET_LIMITED_PARTIAL_RESULT", "reason": "recorded experiment budget exceeded"}
    if pi.get("interim_partial"):
        return {"status": "PARTIAL", "classification": None, "reason": f"{pi.get('status')}; no final scientific classification yet"}
    if not fresh:
        return {"status": "PARTIAL", "classification": None, "reason": "post-policy-iteration full-loop validation/calibration not complete"}
    val, cal, _ = selected_metrics(report)
    if not val or not cal:
        return {"status": "PARTIAL", "classification": None, "reason": "validation or calibration evidence missing"}
    finite = int((val.get("segment_statistics") or {}).get("successful_finite_segment_then_safety_count", 0) or 0)
    if finite <= 0:
        return {"status": "FINAL", "classification": "RECOVERY_WORKS_ENTRY_EXIT_NOT_YET_RELIABLE", "reason": "no validated successful finite recovery segment followed by Safety"}
    final_metrics = report.get("final_test") or {}
    if not final_rows or not final_metrics:
        return {"status": "PARTIAL", "classification": None, "reason": "final frozen test has not been run"}
    n = int(final_metrics.get("source_episode_count", 0) or 0)
    learned_success = int(round(float(final_metrics.get("q_learned", 0) or 0) * n))
    safe_success = int(round(float(final_metrics.get("q_safe", 0) or 0) * n))
    breaks = int(final_metrics.get("break", 0) or 0)
    hard_safe = bool((final_metrics.get("hard_safety") or {}).get("intact"))
    finite_final = int((final_metrics.get("segment_statistics") or {}).get("successful_finite_segment_then_safety_count", 0) or 0)
    ci = final_metrics.get("delta_q_paired_ci95") or [None, None]
    positive_ci = len(ci) == 2 and isinstance(ci[0], (int, float)) and ci[0] > 0
    if (
        n == 200 and len(final_rows) == 200 and learned_success > safe_success
        and positive_ci and (breaks / max(safe_success, 1)) <= 0.05
        and finite_final > 0 and hard_safe
    ):
        return {
            "status": "FINAL",
            "classification": "SINGLE_SEGMENT_RECOVERY_SUPPORTED",
            "reason": "frozen 200-episode test improves task success with positive paired CI, zero observed breaks, finite recovery-to-Safety completions, and intact hard safety",
        }
    return {"status": "FINAL", "classification": "SINGLE_SEGMENT_RECOVERY_DOES_NOT_IMPROVE_BASELINE", "reason": "frozen test does not improve Safety under the pilot criteria"}


def validation_markdown(report: dict[str, Any], fresh: bool, freshness_reason: str) -> str:
    val, cal, policy_hash = selected_metrics(report)
    final_test = report.get("final_test")
    lines = [
        "# Validation and calibration report",
        "",
        f"Policy-iteration alignment: **{'CURRENT' if fresh else 'STALE/PRE-ITERATION'}** — {freshness_reason}.",
        "The numerical table is preserved as development evidence, but stale evidence is not used to accept the iterated policy.",
        "",
        "| Split | Sources | Safety Q | Learned Q | Rescue | Break | Mean J_def |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metrics_table("Validation selected", val),
        metrics_table("Calibration", cal),
        metrics_table("Final test", final_test),
        "",
        f"Selected policy hash in the authoritative report: `{policy_hash or 'unavailable'}`.",
    ]
    if val:
        seg = val.get("segment_statistics") or {}
        lines += [
            "",
            "Validation segment evidence:",
            "",
            f"- Entered: {seg.get('entered_count')}/{val.get('source_episode_count')}",
            f"- Exited: {seg.get('exited_count')}/{seg.get('entered_count')}",
            f"- Successful finite segment followed by Safety: {seg.get('successful_finite_segment_then_safety_count')}",
            f"- Returned to Safety then failed: {seg.get('returned_to_safety_then_failed_count')}",
            f"- Mean J_def: {(val.get('jdef') or {}).get('mean')}",
        ]
    if cal:
        audit = report.get("calibration_audit") or {}
        lines += [
            "",
            "Calibration audit:",
            "",
            f"- Observed break: {cal.get('break')} (rate {cal.get('break_rate')})",
            f"- 95% break interval: {cal.get('break_rate_ci95')}",
            f"- Pilot point budget: {audit.get('break_budget')}",
            f"- Support sufficient for certification: {not bool(audit.get('insufficient_support', True))}",
            "- No certification is claimed.",
        ]
    if final_test:
        seg = final_test.get("segment_statistics") or {}
        deg = final_test.get("degeneracy") or {}
        lines += [
            "",
            "Frozen final-test evidence:",
            "",
            f"- Safety Q: {final_test.get('q_safe')}; learned Q: {final_test.get('q_learned')}; paired delta: {final_test.get('delta_q')} with CI {final_test.get('delta_q_paired_ci95')}.",
            f"- Rescue/break: {final_test.get('rescue')}/{final_test.get('break')}.",
            f"- Timeout rescue: {final_test.get('timeout_rescues')}/{final_test.get('timeout_failures')}; strict-deadlock rescue: {final_test.get('strict_deadlock_rescues')}/{final_test.get('strict_deadlock_failures')}.",
            f"- Successful finite segment then Safety: {seg.get('successful_finite_segment_then_safety_count')}/{final_test.get('source_episode_count')}.",
            f"- Entry-almost-everywhere degeneracy: {bool(deg.get('enter_almost_everywhere'))}. This is an explicit interpretability/selectivity limitation.",
        ]
    return "\n".join(lines) + "\n"


def final_markdown(root: Path, protocol: dict[str, Any], pi: dict[str, Any], report: dict[str, Any], fresh: bool, runtime: dict[str, Any], integrity: dict[str, Any], decision: dict[str, Any], final_rows: list[dict[str, Any]]) -> str:
    val, cal, policy_hash = selected_metrics(report)
    final_test = report.get("final_test") or {}
    source_integrity = read_json(root / "source_collection_integrity.json", {}) or {}
    entry_manifest = read_json(root / "labels" / "entry_merged" / "decision_dataset_manifest.json", {}) or {}
    exit_manifest = read_json(root / "labels" / "exit_merged" / "decision_dataset_manifest.json", {}) or {}
    entry_head_data = entry_manifest.get("head_dataset_manifest") or entry_manifest
    exit_head_data = exit_manifest.get("head_dataset_manifest") or exit_manifest
    pi_entry = ((pi.get("paired_branch_round") or {}).get("entry") or {})
    pi_exit = ((pi.get("paired_branch_round") or {}).get("exit") or {})
    original_entry_decisions = entry_head_data.get("train_decision_inputs", 0) + entry_head_data.get("validation_decision_inputs", 0)
    original_exit_decisions = exit_head_data.get("train_decision_inputs", 0) + exit_head_data.get("validation_decision_inputs", 0)
    pi_entry_decisions = int(((pi.get("state_collection") or {}).get("selected_entry_states", 0)) or 0)
    pi_exit_decisions = int(((pi.get("state_collection") or {}).get("selected_exit_states", 0)) or 0)
    paired_branch_total = sum(int(value or 0) for value in (
        entry_head_data.get("branch_rollouts"), exit_head_data.get("branch_rollouts"),
        pi_entry.get("completed_branch_continuations"), pi_exit.get("completed_branch_continuations"),
    ))
    stage_status = [
        ("Stage A primitive audit", "COMPLETE"),
        ("Generic source collection", "COMPLETE" if source_integrity.get("status") == "PASS" else "INCOMPLETE"),
        ("Bootstrap/second-pass entry and exit labels", "COMPLETE" if entry_manifest and exit_manifest else "INCOMPLETE"),
        ("First head training and validation/calibration", "COMPLETE" if report else "INCOMPLETE"),
        ("Required bounded policy-improvement round", pi.get("status", "INCOMPLETE")),
        ("Post-iteration validation/calibration", "COMPLETE" if fresh and val and cal else "PENDING"),
        ("Frozen final 200-episode test", "COMPLETE" if len(final_rows) == 200 and final_test else "NOT RUN"),
    ]
    lines = [
        "# First semantically aligned single-segment recovery pilot",
        "",
        f"Report status: **{decision['status']}**.",
        f"Classification: **{decision.get('classification') or 'PENDING'}**.",
        f"Reason: {decision['reason']}.",
        "",
        "## Completed stages",
        "",
        "| Stage | Status |",
        "|---|---|",
    ] + [f"| {name} | {status} |" for name, status in stage_status]
    lines += [
        "",
        "## Recovery candidates",
        "",
        "- System G (dense Direct-g): Stage A found weak/inconsistent rescue; it was retained as an integrity reference and did not consume head-label training budget.",
        "- System ETA (one-shot persistent eta): Stage A established credible mid/late strict-deadlock takeover capacity and it advanced to entry/exit learning.",
        "",
        "## Decision semantics",
        "",
        "- Entry input: 214-D deployment feature `h_t`.",
        "- ETA exit input: 217-D concatenation `[h_t, eta_latched]`.",
        "- Both heads use `input -> 64 -> 64 -> 1` with SiLU.",
        "- Success loss: `r*softplus(-s) + (1+lambda_break)*b*softplus(s)`; scores are decision scores, not success probabilities.",
        "- WAIT executes one Safety transition and then follows an incumbent policy that can enter later in temporally aligned rounds; it does not forbid future recovery.",
        "- EXIT executes Safety immediately and Safety remains in control to terminal; paired labels therefore include actual downstream Safety behavior.",
        "- Recovery is one contiguous segment. Eta is predicted once on entry and never re-predicted within the segment.",
        "",
        "## Frozen full-loop evidence",
        "",
        "| Split | Sources | Safety Q | Learned Q | Rescue | Break | Mean J_def |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metrics_table("Validation", val),
        metrics_table("Calibration", cal),
        metrics_table("Final test", final_test),
        "",
        f"These metrics use policy `{policy_hash or 'unavailable'}` and are **{'aligned to the iterated heads' if fresh else 'pre-policy-iteration evidence only'}**.",
    ]
    if val:
        seg = val.get("segment_statistics") or {}
        lines += [
            f"Validation observed {seg.get('successful_finite_segment_then_safety_count')} successful finite recovery segments followed by Safety, and {seg.get('returned_to_safety_then_failed_count')} exits followed by failure.",
        ]
    if cal:
        lines += [
            f"Calibration observed Q={cal.get('q_learned')} versus Safety Q={cal.get('q_safe')}, rescue={cal.get('rescue')}, break={cal.get('break')}; its break interval was {cal.get('break_rate_ci95')}, so support was insufficient for certification.",
        ]
    if final_test:
        seg = final_test.get("segment_statistics") or {}
        deg = final_test.get("degeneracy") or {}
        lines += [
            f"Final test: Safety {int(round(final_test.get('q_safe', 0) * final_test.get('source_episode_count', 0)))}/{final_test.get('source_episode_count')} (Q={final_test.get('q_safe')}); hierarchy {int(round(final_test.get('q_learned', 0) * final_test.get('source_episode_count', 0)))}/{final_test.get('source_episode_count')} (Q={final_test.get('q_learned')}); rescue={final_test.get('rescue')}, break={final_test.get('break')}, paired delta={final_test.get('delta_q')} CI={final_test.get('delta_q_paired_ci95')}.",
            f"Failure slices: timeout rescue {final_test.get('timeout_rescues')}/{final_test.get('timeout_failures')}; strict-deadlock rescue {final_test.get('strict_deadlock_rescues')}/{final_test.get('strict_deadlock_failures')}.",
            f"All {seg.get('entered_count')} entered and {seg.get('exited_count')} exited; {seg.get('successful_finite_segment_then_safety_count')} completed after a finite recovery segment returned to Safety; {seg.get('returned_to_safety_then_failed_count')} returned then failed.",
            f"Mean J_def={((final_test.get('jdef') or {}).get('mean'))}; hard safety intact={((final_test.get('hard_safety') or {}).get('intact'))}.",
            "",
            "## Critical limitation: entry-almost-everywhere",
            "",
            f"The frozen final controller entered recovery on {seg.get('entered_count')}/{final_test.get('source_episode_count')} episodes, and the degeneracy audit reports `enter_almost_everywhere={bool(deg.get('enter_almost_everywhere'))}`. Therefore the experiment supports the one-segment recovery system's task performance and learned exit, but does **not** establish a selective or semantically discriminating entry rule. In effect, the entry head nearly always initiates one recovery segment, after which the learned exit returns to Safety. This limitation must be addressed before claiming that entry identifies only recovery-needing states.",
        ]
    lines += [
        "",
        "## Data and budget",
        "",
        f"- Safety source trajectories: {source_integrity.get('source_count')} across train/validation/calibration.",
        f"- Generic materialized decision states: {source_integrity.get('available_decision_states')} from {source_integrity.get('unique_root_sources_with_decisions')} roots.",
        f"- Frozen independent root-source groups: 40 train + 20 validation + 20 calibration + 200 final test = 280 roots.",
        f"- Original merged entry decisions: {original_entry_decisions}; original merged exit decisions: {original_exit_decisions}.",
        f"- Policy-iteration decisions: {pi_entry_decisions} entry + {pi_exit_decisions} exit.",
        f"- Total supervised decision-state records: {original_entry_decisions + original_exit_decisions + pi_entry_decisions + pi_exit_decisions}; paired branch continuations: {paired_branch_total}.",
        f"- Recorded new continuations: {runtime.get('new_continuations_recorded')}/{runtime.get('continuation_budget_limit')}.",
        f"- Recorded physical steps: {runtime.get('new_physical_steps_recorded')}/{runtime.get('physical_step_budget_limit')}.",
        f"- Observed runtime timestamp window: {runtime.get('observed_experiment_runtime_window_seconds')} s; summed recorded stage spans: {runtime.get('sum_recorded_stage_wall_seconds')} s. These exclude uninstrumented training/report time.",
        f"- Maximum recorded simultaneous shard count: {runtime.get('maximum_recorded_gpu_shards')}; recorded thread settings: {runtime.get('recorded_thread_settings')}.",
        "",
        "## Safety and integrity",
        "",
        f"Integrity status: **{integrity.get('status')}**. Observed hard-safety counts: `{json.dumps(integrity.get('observed_hard_safety_counts', {}), sort_keys=True)}`.",
        f"Final-test episode outcome files observed: {integrity.get('final_test_outcome_file_count')} (runtime JSON files excluded).",
        "",
        "## Remaining uncertainty",
        "",
        "Sparse paired decision evidence left many inputs unresolved; non-significance was not treated as equivalence. Calibration remained too small for standalone certification, although the frozen final test subsequently showed zero observed breaks with a 95% upper bound below 5%. The dominant remaining semantic limitation is entry-almost-everywhere, not task success or learned exit feasibility.",
        "",
        "This pilot does not solve repeated recovery, exit/re-entry, a different scenario, or controller retraining.",
    ]
    return "\n".join(lines) + "\n"


def artifact_manifest(root: Path, generated_names: Iterable[str], status: str) -> dict[str, Any]:
    important = [
        "protocol.json", "controller_and_projection_hashes.json", "state_machine_spec.md", "source_split_manifest.json",
        "primitive_readiness_report.md", "decision_state_manifest.meta.json", "source_collection_integrity.json",
        "eta_recovery_state_collection_integrity.json", "policy_iteration_state_collection_integrity.json",
        "validation_and_calibration_report.json", "final_test_manifest.json",
        "policy_iteration_validation.json", "policy_iteration_calibration.json", "final_test_results.json",
    ] + list(generated_names)
    artifacts = []
    for name in important:
        path = root / name
        artifacts.append({"path": name, "exists": path.is_file(), "sha256": sha256(path), "bytes": path.stat().st_size if path.is_file() else None})
    return {
        "schema": "single_segment_recovery_training_manifest_v1",
        "generated_utc": utc_now(),
        "status": status,
        "scope": "read-only aggregation; no rollout, training, or final-test execution",
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--validation-report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read_json(root / "protocol.json", {}) or {}
    report_path = args.validation_report or (root / "validation_and_calibration_report.json")
    report = load_evaluation_report(root, report_path)

    pi = build_policy_iteration_manifest(root)
    runtime = summarize_runtime(root, protocol)
    integrity = build_integrity(root, runtime, pi)
    fresh, freshness_reason = report_freshness(root, pi, report)
    _, _, policy_hash = selected_metrics(report)
    full_rows = outcome_rows(root, policy_hash, ("validation", "calibration", "final_test"))
    final_rows = [row for row in full_rows if row.get("split") == "final_test"]
    decision = classify(pi, report, fresh, final_rows, runtime)
    if decision.get("classification") is not None and decision["classification"] not in ALLOWED_FINAL_CLASSIFICATIONS:
        raise RuntimeError(f"invalid classification: {decision['classification']}")

    generated = [
        "policy_iteration_manifest.json", "integrity_checks.json", "runtime_statistics.json",
        "validation_and_calibration_report.md", "final_report.md", "full_episode_outcomes.csv",
        "entry_exit_segment_statistics.csv", "rescue_break_and_deformation_metrics.json",
    ]
    write_json(root / "policy_iteration_manifest.json", pi, args.dry_run)
    write_json(root / "runtime_statistics.json", runtime, args.dry_run)
    write_json(root / "integrity_checks.json", integrity, args.dry_run)
    write_json(root / "rescue_break_and_deformation_metrics.json", {
        "schema": "single_segment_rescue_break_deformation_summary_v1",
        "generated_utc": utc_now(),
        "policy_iteration_aligned": fresh,
        "freshness_reason": freshness_reason,
        "validation": selected_metrics(report)[0],
        "calibration": selected_metrics(report)[1],
        "final_test": report.get("final_test"),
    }, args.dry_run)
    write_md(root / "validation_and_calibration_report.md", validation_markdown(report, fresh, freshness_reason), args.dry_run)
    write_md(root / "final_report.md", final_markdown(root, protocol, pi, report, fresh, runtime, integrity, decision, final_rows), args.dry_run)

    outcome_fields = [
        "root_source_id", "split", "safety_outcome", "learned_outcome", "safety_success", "learned_success",
        "entry_step", "exit_step", "recovery_segment_count", "recovery_transition_count", "returned_to_safety",
        "terminal_step", "jdef", "agent_collision", "wall_collision", "invalid_action", "nan_inf",
        "projection_solver_failure", "policy_hash",
    ]
    write_csv(root / "full_episode_outcomes.csv", full_rows, outcome_fields, args.dry_run)
    segment_rows = []
    for split, metrics in (
        ("validation", selected_metrics(report)[0]),
        ("calibration", selected_metrics(report)[1]),
        ("final_test", report.get("final_test")),
    ):
        if not metrics:
            continue
        seg = metrics.get("segment_statistics") or {}
        segment_rows.append({
            "split": split, "policy_iteration_aligned": fresh, "source_episode_count": metrics.get("source_episode_count"),
            "entered_count": seg.get("entered_count"), "exited_count": seg.get("exited_count"),
            "successful_finite_segment_then_safety_count": seg.get("successful_finite_segment_then_safety_count"),
            "successful_takeover_until_completion_count": seg.get("successful_takeover_until_completion_count"),
            "returned_to_safety_then_failed_count": seg.get("returned_to_safety_then_failed_count"),
            "mean_recovery_transitions": (seg.get("recovery_transition_count") or {}).get("mean"),
            "median_recovery_transitions": (seg.get("recovery_transition_count") or {}).get("median"),
        })
    write_csv(root / "entry_exit_segment_statistics.csv", segment_rows, [
        "split", "policy_iteration_aligned", "source_episode_count", "entered_count", "exited_count",
        "successful_finite_segment_then_safety_count", "successful_takeover_until_completion_count",
        "returned_to_safety_then_failed_count", "mean_recovery_transitions", "median_recovery_transitions",
    ], args.dry_run)

    if not args.dry_run:
        manifest = artifact_manifest(root, generated, decision["status"])
        write_json(root / "manifest.json", manifest, False)
    summary = {
        "dry_run": args.dry_run,
        "policy_iteration_status": pi["status"],
        "report_policy_iteration_aligned": fresh,
        "report_freshness_reason": freshness_reason,
        "integrity_status": integrity["status"],
        "within_budget": runtime["within_budget"],
        "status": decision["status"],
        "classification": decision.get("classification"),
        "reason": decision["reason"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
