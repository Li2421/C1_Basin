"""Fail-closed finalizer for matched single-segment branch outcomes.

This script performs no rollout and no training.  It accepts a stage only
after every frozen task has one complete, semantically matching result and
every future stream has both action branches.  Its JSONL output is the typed,
authoritative input to the separate head-training driver; CSV/Markdown files
are human-readable audits only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decision_orchestration import build_head_dataset, content_hash, file_hash


HERE = Path(__file__).resolve().parent
HORIZON = 850
FROZEN_STAGE_TASK_COUNTS = {"exit_bootstrap_never_exit": 1536}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_semantic_hash(payload: Mapping[str, Any], path: Path) -> None:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if not claimed or canonical_hash(body) != claimed:
        raise RuntimeError((str(path), "semantic hash mismatch"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, allow_nan=False)
                        if isinstance(value, (dict, list, tuple)) else value
                    )
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _exact_equal(result: Mapping[str, Any], task: Mapping[str, Any], key: str) -> None:
    if result.get(key) != task.get(key):
        raise RuntimeError((task.get("task_id"), f"result/task mismatch: {key}"))


def validate_result(result: Mapping[str, Any], task: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    task_id = task["task_id"]
    if result.get("schema") != "single_segment_paired_branch_result_v1":
        raise RuntimeError((task_id, "result schema mismatch"))
    if result.get("record_complete") is not True or result.get("execution_error") is not None:
        raise RuntimeError((task_id, "incomplete/errored branch result", result.get("execution_error")))
    if result.get("manifest_content_sha256") != manifest["content_sha256"]:
        raise RuntimeError((task_id, "result belongs to a different frozen manifest"))
    for key in (
        "task_id", "task_index", "decision_id", "action", "root_source_id", "split",
        "state_id", "absolute_step", "current_flow_realization_id", "future_stream_id",
        "future_rollout_id", "future_rng_state_hash", "downstream_policy_hash",
    ):
        _exact_equal(result, task, key)
    if result.get("pass_id") != manifest["pass_id"]:
        raise RuntimeError((task_id, "pass lineage mismatch"))
    if result.get("state_hash") != task["state_sha256"]:
        raise RuntimeError((task_id, "state hash mismatch"))
    expected_mode = "RECOVERY" if manifest["decision_kind"] == "exit" else "SAFETY_BEFORE"
    if result.get("mode") != expected_mode:
        raise RuntimeError((task_id, "decision mode mismatch"))
    outcome = str(result.get("outcome"))
    if outcome not in {"success", "deadlock", "timeout", "collision"}:
        raise RuntimeError((task_id, "non-physical/unknown terminal outcome", outcome))
    flags = {
        "success": bool(result.get("success")),
        "deadlock": bool(result.get("deadlock")),
        "timeout": bool(result.get("timeout")),
        "collision": bool(result.get("collision")),
    }
    if sum(flags.values()) != 1 or not flags[outcome]:
        raise RuntimeError((task_id, "outcome indicators are not one-hot"))
    for key in ("remaining_jdef", "runtime_seconds"):
        value = float(result[key])
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError((task_id, f"invalid {key}", value))
    transitions = int(result["physical_transition_count"])
    if transitions != int(result["remaining_physical_steps"]):
        raise RuntimeError((task_id, "physical-step accounting mismatch"))
    if transitions <= 0 or transitions > HORIZON - int(task["absolute_step"]):
        raise RuntimeError((task_id, "remaining horizon violated", transitions))
    if int(result["terminal_global_step"]) != int(task["absolute_step"]) + transitions:
        raise RuntimeError((task_id, "absolute time was not preserved"))
    if outcome == "timeout" and int(result["terminal_global_step"]) != HORIZON:
        raise RuntimeError((task_id, "timeout occurred before official horizon"))
    if int(result["flow_sample_count"]) != transitions:
        raise RuntimeError((task_id, "one Flow sample per physical transition violated"))
    if int(result["first_decision_override_calls"]) < 1:
        raise RuntimeError((task_id, "paired action was not consumed"))
    for key in ("invalid_actions", "nan_inf_events", "projection_failures"):
        if int(result.get(key, 0)) != 0:
            raise RuntimeError((task_id, f"numerical/control failure: {key}"))
    if manifest["decision_kind"] == "exit":
        expected_eta_hash = canonical_hash(task["eta_latched"])
        if result.get("eta_latched_hash") != expected_eta_hash or int(result["eta_query_count"]) != 0:
            raise RuntimeError((task_id, "latched eta was not preserved"))


def load_complete_results(manifest_path: Path, result_directory: Path, *, require_task_count: int | None) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    verify_semantic_hash(manifest, manifest_path)
    if manifest.get("status") != "FROZEN_READY_FOR_EXECUTION":
        raise RuntimeError("branch manifest is not frozen for execution")
    tasks = list(manifest.get("tasks", []))
    if len(tasks) != int(manifest.get("task_count", -1)):
        raise RuntimeError("manifest task count mismatch")
    frozen_count = FROZEN_STAGE_TASK_COUNTS.get(str(manifest.get("stage")))
    expected = require_task_count if require_task_count is not None else frozen_count
    if expected is not None and len(tasks) != int(expected):
        raise RuntimeError((manifest.get("stage"), "unexpected frozen task count", len(tasks), expected))
    expected_names = {f"task_{int(task['task_index']):05d}.json" for task in tasks}
    actual_names = {path.name for path in result_directory.glob("task_*.json")}
    if actual_names != expected_names:
        raise RuntimeError(("result file coverage mismatch", sorted(expected_names - actual_names)[:5], sorted(actual_names - expected_names)[:5]))
    by_index = {int(task["task_index"]): task for task in tasks}
    if len(by_index) != len(tasks) or set(by_index) != set(range(len(tasks))):
        raise RuntimeError("task indices are not unique/contiguous")
    results: list[dict] = []
    for index in range(len(tasks)):
        path = result_directory / f"task_{index:05d}.json"
        row = json.loads(path.read_text())
        validate_result(row, by_index[index], manifest)
        results.append(row)
    if len({row["task_id"] for row in results}) != len(results):
        raise RuntimeError("duplicate result task IDs")
    if sum(int(row["physical_transition_count"]) for row in results) > int(manifest["budget"]["maximum_physical_steps"]):
        raise RuntimeError("actual branch steps exceed frozen maximum")
    return manifest, results


def _decision_evidence_rows(dataset: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, metadata, evidence in (
        ("train", dataset.train_rows, dataset.train_evidence),
        ("validation", dataset.validation_rows, dataset.validation_evidence),
    ):
        for decision, item in zip(metadata, evidence, strict=True):
            rows.append(
                {
                    "target_lineage_id": item.decision_id,
                    "decision_id": decision["decision_id"],
                    "pass_id": decision["pass_id"],
                    "downstream_policy_hash": decision["downstream_policy_hash"],
                    "root_source_id": decision["root_source_id"],
                    "split": split,
                    **asdict(item),
                }
            )
    return rows


def _report_markdown(stage: str, dataset: Any, evidence_rows: Sequence[Mapping[str, Any]], result_count: int) -> str:
    statuses: dict[str, int] = {}
    for row in evidence_rows:
        statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
    equivalents = sum(bool(row["equivalence_supported"]) for row in evidence_rows)
    unresolved = sum(bool(row["unresolved"]) for row in evidence_rows)
    lines = [
        f"# Label uncertainty report: {stage}", "",
        f"- Complete branch rows: {result_count}",
        f"- Matched decision inputs: {len(evidence_rows)}",
        f"- Futures per input: {dataset.manifest['expected_futures_per_input']}",
        f"- Train / validation inputs: {dataset.manifest['train_decision_inputs']} / {dataset.manifest['validation_decision_inputs']}",
        f"- Unresolved inputs: {unresolved}",
        f"- Success-equivalence supported within epsilon_Q=0.02: {equivalents}", "",
        "## Status counts", "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(statuses.items()))
    lines += [
        "", "## Interpretation guardrail", "",
        "Thirty-two matched futures with no discordant successes remain unresolved under the frozen conservative paired interval. ",
        "They are not evidence of epsilon_Q=0.02 equivalence, and therefore do not activate the deformation tie-break.", "",
    ]
    return "\n".join(lines)


def finalize(
    manifest_path: Path,
    result_directory: Path,
    output_directory: Path,
    *,
    require_task_count: int | None = None,
) -> dict[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError((output_directory, "refusing to overwrite finalized stage artifacts"))
    manifest, results = load_complete_results(
        manifest_path, result_directory, require_task_count=require_task_count
    )
    head_kind = f"{manifest['decision_kind']}_eta"
    futures = int(manifest["matched_future_semantics"]["count_per_decision"])
    dataset = build_head_dataset(
        manifest["decisions"], results, head_kind=head_kind,
        expected_futures_per_input=futures,
        require_validation=not str(manifest["stage"]).startswith("policy_iteration_"),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    outcomes_jsonl = output_directory / "paired_branch_outcomes.jsonl"
    outcomes_csv = output_directory / "paired_branch_outcomes.csv"
    uncertainty_csv = output_directory / "label_uncertainty.csv"
    report_path = output_directory / "label_uncertainty_report.md"
    snapshot_path = output_directory / "decision_dataset_snapshot.npz"
    _atomic_jsonl(outcomes_jsonl, results)
    _atomic_csv(outcomes_csv, results)
    evidence_rows = _decision_evidence_rows(dataset)
    _atomic_csv(uncertainty_csv, evidence_rows)
    _atomic_text(report_path, _report_markdown(manifest["stage"], dataset, evidence_rows, len(results)))
    _atomic_npz(
        snapshot_path,
        train_features=dataset.train_features,
        validation_features=dataset.validation_features,
        train_r=np.asarray([item.r for item in dataset.train_evidence]),
        train_b=np.asarray([item.b for item in dataset.train_evidence]),
        validation_r=np.asarray([item.r for item in dataset.validation_evidence]),
        validation_b=np.asarray([item.b for item in dataset.validation_evidence]),
    )
    final_manifest = {
        "schema": "single_segment_finalized_head_dataset_v1",
        "status": "COMPLETE_FROZEN_READY_FOR_TRAINING",
        "stage": manifest["stage"],
        "pass_id": manifest["pass_id"],
        "head_kind": head_kind,
        "source_branch_manifest": str(manifest_path.resolve()),
        "source_branch_manifest_sha256": file_hash(manifest_path),
        "source_branch_manifest_content_sha256": manifest["content_sha256"],
        "result_directory": str(result_directory.resolve()),
        "result_count": len(results),
        "expected_result_count": int(manifest["task_count"]),
        "decision_count": int(manifest["decision_count"]),
        "expected_futures_per_input": futures,
        "paired_branch_outcomes_jsonl": str(outcomes_jsonl.resolve()),
        "paired_branch_outcomes_jsonl_sha256": file_hash(outcomes_jsonl),
        "paired_branch_outcomes_csv": str(outcomes_csv.resolve()),
        "paired_branch_outcomes_csv_sha256": file_hash(outcomes_csv),
        "label_uncertainty_csv": str(uncertainty_csv.resolve()),
        "label_uncertainty_csv_sha256": file_hash(uncertainty_csv),
        "label_uncertainty_report": str(report_path.resolve()),
        "label_uncertainty_report_sha256": file_hash(report_path),
        "dataset_snapshot": str(snapshot_path.resolve()),
        "dataset_snapshot_sha256": file_hash(snapshot_path),
        "head_dataset_manifest": dataset.manifest,
        "all_results_complete": True,
        "execution_error_count": 0,
        "outcome_counts": {
            name: sum(row["outcome"] == name for row in results)
            for name in ("success", "deadlock", "timeout", "collision")
        },
        "train_validation_source_overlap": 0,
        "final_test_used": False,
        "training_launched": False,
    }
    final_manifest["content_sha256"] = content_hash(final_manifest)
    _atomic_json(output_directory / "decision_dataset_manifest.json", final_manifest)
    return final_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--require-task-count", type=int)
    args = parser.parse_args()
    result = finalize(
        args.manifest, args.result_directory, args.output_directory,
        require_task_count=args.require_task_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
