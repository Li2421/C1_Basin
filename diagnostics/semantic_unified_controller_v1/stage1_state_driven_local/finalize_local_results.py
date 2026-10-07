"""Fail-closed finalizer for complete matched Stage-1 S/L branch results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1"
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage1_state_driven_local"
sys.path[:0] = [str(BASE), str(HERE)]

from decision_orchestration import build_head_dataset, file_hash  # noqa: E402
from local_common import HORIZON, content_hash, verify_content_hash  # noqa: E402


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    os.replace(temporary, path)


def validate_result(row: dict[str, Any], task: dict[str, Any], manifest: dict[str, Any]) -> None:
    if row.get("schema") != "semantic_local_paired_branch_result_v1" or row.get("record_complete") is not True:
        raise RuntimeError((task["task_id"], "incomplete/wrong result schema"))
    if row.get("execution_error") is not None or row.get("manifest_content_sha256") != manifest["content_sha256"]:
        raise RuntimeError((task["task_id"], "execution/manifest mismatch"))
    for key in (
        "task_id", "task_index", "decision_id", "pass_id", "action", "root_source_id",
        "split", "state_id", "absolute_step", "current_flow_realization_id",
        "future_stream_id", "future_rollout_id", "future_rng_state_hash", "downstream_policy_hash",
    ):
        if row.get(key) != task.get(key):
            raise RuntimeError((task["task_id"], "task/result mismatch", key))
    if row.get("state_hash") != task["state_sha256"] or row.get("mode") != "NORMAL":
        raise RuntimeError((task["task_id"], "state/mode mismatch"))
    outcome = row.get("outcome")
    flags = [bool(row.get(name)) for name in ("success", "deadlock", "timeout", "collision", "other_failure")]
    if outcome not in {"success", "deadlock", "timeout", "collision"} or sum(flags) != 1:
        raise RuntimeError((task["task_id"], "invalid terminal outcome"))
    transitions = int(row["physical_transition_count"])
    if transitions != int(row["remaining_physical_steps"]) or transitions <= 0:
        raise RuntimeError((task["task_id"], "physical-step accounting"))
    if transitions > HORIZON - int(task["absolute_step"]):
        raise RuntimeError((task["task_id"], "horizon extension"))
    if int(row["terminal_global_step"]) != int(task["absolute_step"]) + transitions:
        raise RuntimeError((task["task_id"], "absolute-time mismatch"))
    if int(row["flow_sample_count"]) != transitions or int(row["head_query_count"]) != transitions:
        raise RuntimeError((task["task_id"], "per-step query count mismatch"))
    if int(row["direct_query_count"]) != int(row["local_action_count"]):
        raise RuntimeError((task["task_id"], "Direct-g query not one-for-one with local events"))
    if int(row["first_decision_override_calls"]) < 1:
        raise RuntimeError((task["task_id"], "paired action never consumed"))
    for key in ("invalid_actions", "nan_inf_events", "projection_failures"):
        if int(row.get(key, 0)) != 0:
            raise RuntimeError((task["task_id"], key))
    for key in ("remaining_jdef", "runtime_seconds"):
        if not np.isfinite(float(row[key])) or float(row[key]) < 0:
            raise RuntimeError((task["task_id"], key))


def finalize(manifest_path: Path, result_directory: Path, output_directory: Path) -> dict[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError((output_directory, "refusing overwrite"))
    manifest = json.loads(manifest_path.read_text())
    verify_content_hash(manifest, manifest_path)
    tasks = list(manifest["tasks"])
    expected_names = {f"task_{int(task['task_index']):05d}.json" for task in tasks}
    actual_names = {path.name for path in result_directory.glob("task_*.json")}
    if actual_names != expected_names:
        raise RuntimeError(("result coverage mismatch", len(expected_names), len(actual_names)))
    by_index = {int(task["task_index"]): task for task in tasks}
    rows = []
    for index in range(len(tasks)):
        row = json.loads((result_directory / f"task_{index:05d}.json").read_text())
        validate_result(row, by_index[index], manifest)
        rows.append(row)
    futures = int(manifest["matched_future_semantics"]["count_per_decision"])
    dataset = build_head_dataset(
        manifest["decisions"], rows, head_kind="entry_g",
        expected_futures_per_input=futures, require_validation=True,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    outcomes_jsonl = output_directory / "paired_branch_outcomes.jsonl"
    atomic_text(outcomes_jsonl, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    outcomes_csv = output_directory / "paired_branch_outcomes.csv"
    atomic_csv(outcomes_csv, rows)
    evidence_rows = []
    for split, metadata, evidence in (
        ("train", dataset.train_rows, dataset.train_evidence),
        ("validation", dataset.validation_rows, dataset.validation_evidence),
    ):
        for source, item in zip(metadata, evidence, strict=True):
            evidence_rows.append({
                "decision_id": source["decision_id"], "pass_id": source["pass_id"],
                "downstream_policy_hash": source["downstream_policy_hash"],
                "root_source_id": source["root_source_id"], "split": split, **asdict(item),
            })
    uncertainty = output_directory / "label_uncertainty.csv"
    atomic_csv(uncertainty, evidence_rows)
    report = output_directory / "label_uncertainty_report.md"
    unresolved = sum(bool(row["unresolved"]) for row in evidence_rows)
    atomic_text(report, "\n".join([
        f"# Stage-1 S/L label audit: {manifest['stage']}", "",
        f"- complete branch rows: {len(rows)}", f"- decision inputs: {len(evidence_rows)}",
        f"- matched futures/input: {futures}", f"- unresolved inputs: {unresolved}", "",
        "No-discordance at n=32 is retained as unresolved; it is not treated as proven equivalence.", "",
    ]))
    snapshot = output_directory / "decision_dataset_snapshot.npz"
    np.savez(
        snapshot, train_features=dataset.train_features, validation_features=dataset.validation_features,
        train_r=np.asarray([item.r for item in dataset.train_evidence]),
        train_b=np.asarray([item.b for item in dataset.train_evidence]),
        validation_r=np.asarray([item.r for item in dataset.validation_evidence]),
        validation_b=np.asarray([item.b for item in dataset.validation_evidence]),
    )
    result = {
        "schema": "single_segment_finalized_head_dataset_v1",
        "semantic_schema": "semantic_local_finalized_head_dataset_v1",
        "status": "COMPLETE_FROZEN_READY_FOR_TRAINING", "stage": manifest["stage"],
        "pass_id": manifest["pass_id"], "head_kind": "entry_g",
        "source_branch_manifest": str(manifest_path.resolve()),
        "source_branch_manifest_sha256": file_hash(manifest_path),
        "source_branch_manifest_content_sha256": manifest["content_sha256"],
        "result_directory": str(result_directory.resolve()), "result_count": len(rows),
        "expected_result_count": len(tasks), "decision_count": len(manifest["decisions"]),
        "expected_futures_per_input": futures,
        "paired_branch_outcomes_jsonl": str(outcomes_jsonl.resolve()),
        "paired_branch_outcomes_jsonl_sha256": file_hash(outcomes_jsonl),
        "paired_branch_outcomes_csv": str(outcomes_csv.resolve()),
        "paired_branch_outcomes_csv_sha256": file_hash(outcomes_csv),
        "label_uncertainty_csv": str(uncertainty.resolve()),
        "label_uncertainty_csv_sha256": file_hash(uncertainty),
        "label_uncertainty_report": str(report.resolve()),
        "label_uncertainty_report_sha256": file_hash(report),
        "dataset_snapshot": str(snapshot.resolve()), "dataset_snapshot_sha256": file_hash(snapshot),
        "head_dataset_manifest": dataset.manifest, "all_results_complete": True,
        "execution_error_count": 0, "train_validation_source_overlap": 0,
        "outcome_counts": {name: sum(row["outcome"] == name for row in rows) for name in ("success", "deadlock", "timeout", "collision")},
        "final_test_used": False, "training_launched": False,
    }
    result["content_sha256"] = content_hash(result)
    atomic_json(output_directory / "decision_dataset_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.manifest, args.result_directory, args.output_directory), indent=2))


if __name__ == "__main__":
    main()
