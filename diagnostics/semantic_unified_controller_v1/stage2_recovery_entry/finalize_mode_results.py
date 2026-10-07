"""Fail-closed aggregation and conservative three-way Stage-2 targets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
BASE = ROOT / "diagnostics/single_segment_recovery_training_v1"
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage2_recovery_entry"
sys.path[:0] = [str(BASE), str(HERE)]

from decision_learning import paired_success_difference_interval  # noqa: E402
from mode_common import ACTION_NAMES, FUTURES, HORIZON, content_hash, sha256, verify_content_hash  # noqa: E402


EPSILON_Q = 0.02
CI_LEVEL = 0.90


@dataclass(frozen=True)
class ThreeWayEvidence:
    decision_id: str
    n_futures: int
    q_safety: float
    q_local: float
    q_recovery: float
    mean_jdef_safety: float
    mean_jdef_local: float
    mean_jdef_recovery: float
    local_rescue_vs_safety: int
    local_break_vs_safety: int
    recovery_rescue_vs_safety: int
    recovery_break_vs_safety: int
    target_action: int | None
    status: str
    unresolved: bool
    strict_support_margin: float | None
    pairwise: dict[str, Any]


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
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    os.replace(temporary, path)


def validate_result(row: dict[str, Any], task: dict[str, Any], manifest: dict[str, Any]) -> None:
    if row.get("schema") != "semantic_mode_paired_branch_result_v1" or row.get("record_complete") is not True:
        raise RuntimeError((task["task_id"], "incomplete/wrong result schema"))
    if row.get("execution_error") is not None or row.get("manifest_content_sha256") != manifest["content_sha256"]:
        raise RuntimeError((task["task_id"], "execution/manifest mismatch"))
    for key in (
        "task_id", "task_index", "decision_id", "pass_id", "action", "action_name",
        "root_source_id", "split", "state_id", "absolute_step",
        "current_flow_realization_id", "future_stream_id", "future_rollout_id",
        "future_rng_state_hash", "downstream_policy_hash",
    ):
        if row.get(key) != task.get(key):
            raise RuntimeError((task["task_id"], "task/result mismatch", key))
    if row.get("state_hash") != task["state_sha256"] or row.get("mode") != "NORMAL":
        raise RuntimeError((task["task_id"], "state/mode mismatch"))
    flags = [bool(row.get(name)) for name in ("success", "deadlock", "timeout", "collision", "other_failure")]
    if row.get("outcome") not in {"success", "deadlock", "timeout", "collision", "other"} or sum(flags) != 1:
        raise RuntimeError((task["task_id"], "terminal outcome mismatch"))
    transitions = int(row["physical_transition_count"])
    if transitions <= 0 or transitions != int(row["remaining_physical_steps"]):
        raise RuntimeError((task["task_id"], "step accounting"))
    if transitions > HORIZON - int(task["absolute_step"]):
        raise RuntimeError((task["task_id"], "horizon extension"))
    if int(row["terminal_global_step"]) != int(task["absolute_step"]) + transitions:
        raise RuntimeError((task["task_id"], "absolute-time mismatch"))
    if int(row["flow_sample_count"]) != transitions:
        raise RuntimeError((task["task_id"], "Flow sample/transition mismatch"))
    action = int(task["action"])
    if action in (0, 1):
        if int(row["eta_query_count"]) != 0 or int(row["recovery_transition_count"]) != 0:
            raise RuntimeError((task["task_id"], "S/L branch entered recovery"))
        if int(row["local_head_query_count"]) != transitions:
            raise RuntimeError((task["task_id"], "S/L downstream policy not queried every step"))
        if int(row["direct_query_count"]) != int(row["local_action_count"]):
            raise RuntimeError((task["task_id"], "local query/action mismatch"))
        if row["eta_latched"] is not None:
            raise RuntimeError((task["task_id"], "S/L branch contains eta"))
    else:
        if int(row["eta_query_count"]) != 1 or int(row["recovery_transition_count"]) != transitions:
            raise RuntimeError((task["task_id"], "R branch is not one-shot-eta dense recovery"))
        if int(row["direct_query_count"]) != 0 or int(row["local_action_count"]) != 0:
            raise RuntimeError((task["task_id"], "R branch mixed local actions"))
        eta = np.asarray(row["eta_latched"], dtype=np.float64)
        if eta.shape != (3,) or not np.isfinite(eta).all():
            raise RuntimeError((task["task_id"], "invalid latched eta"))
    for key in ("invalid_actions", "nan_inf_events", "projection_failures"):
        if int(row.get(key, 0)) != 0:
            raise RuntimeError((task["task_id"], key))
    for key in ("remaining_jdef", "runtime_seconds", "current_replay_max_abs"):
        if not np.isfinite(float(row[key])) or float(row[key]) < 0:
            raise RuntimeError((task["task_id"], key))


def pairwise_summary(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    """Return B-minus-A paired evidence."""
    by_stream_a = {row["future_stream_id"]: row for row in rows_a}
    by_stream_b = {row["future_stream_id"]: row for row in rows_b}
    if set(by_stream_a) != set(by_stream_b):
        raise RuntimeError("three-way future coverage mismatch")
    rescue = break_count = both_success = both_fail = 0
    for stream in sorted(by_stream_a):
        first, second = bool(by_stream_a[stream]["success"]), bool(by_stream_b[stream]["success"])
        if not first and second: rescue += 1
        elif first and not second: break_count += 1
        elif first: both_success += 1
        else: both_fail += 1
    n = len(by_stream_a)
    lower, upper = paired_success_difference_interval(rescue, break_count, n, confidence=CI_LEVEL)
    return {
        "n": n, "both_fail": both_fail, "b_rescues_a": rescue,
        "b_breaks_a": break_count, "both_success": both_success,
        "delta_q_b_minus_a": (rescue-break_count)/n,
        "ci_level": CI_LEVEL, "ci_lower": lower, "ci_upper": upper,
        "practical_advantage_supported": lower > EPSILON_Q,
        "equivalence_supported": lower >= -EPSILON_Q and upper <= EPSILON_Q,
    }


def summarize(decision_id: str, by_action: dict[int, list[dict[str, Any]]]) -> ThreeWayEvidence:
    if set(by_action) != set(ACTION_NAMES) or any(len(rows) != FUTURES for rows in by_action.values()):
        raise RuntimeError((decision_id, "incomplete three-way evidence"))
    pairs: dict[str, Any] = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pairs[f"{ACTION_NAMES[right]}_vs_{ACTION_NAMES[left]}"] = pairwise_summary(by_action[left], by_action[right])

    def candidate_vs_other(candidate: int, other: int) -> dict[str, Any]:
        return pairwise_summary(by_action[other], by_action[candidate])

    supported: list[tuple[int, float]] = []
    for candidate in ACTION_NAMES:
        comparisons = [candidate_vs_other(candidate, other) for other in ACTION_NAMES if other != candidate]
        if all(item["practical_advantage_supported"] for item in comparisons):
            supported.append((candidate, min(float(item["ci_lower"]) for item in comparisons)))
    all_equivalent = all(item["equivalence_supported"] for item in pairs.values())
    if len(supported) == 1:
        target, margin = supported[0]
        status, unresolved = f"{ACTION_NAMES[target]}_STRICTLY_SUPPORTED", False
    elif all_equivalent:
        # Safety is the predeclared conservative action when success is proven equivalent.
        target, margin = 0, None
        status, unresolved = "ALL_ACTIONS_SUCCESS_EQUIVALENT_PREFER_SAFETY", False
    else:
        target, margin = None, None
        status, unresolved = "UNRESOLVED_THREE_WAY_SUCCESS_ORDER", True
    q = {action: float(np.mean([bool(row["success"]) for row in rows])) for action, rows in by_action.items()}
    j = {action: float(np.mean([float(row["remaining_jdef"]) for row in rows])) for action, rows in by_action.items()}
    sl = pairwise_summary(by_action[0], by_action[1])
    sr = pairwise_summary(by_action[0], by_action[2])
    return ThreeWayEvidence(
        decision_id=decision_id, n_futures=FUTURES,
        q_safety=q[0], q_local=q[1], q_recovery=q[2],
        mean_jdef_safety=j[0], mean_jdef_local=j[1], mean_jdef_recovery=j[2],
        local_rescue_vs_safety=int(sl["b_rescues_a"]), local_break_vs_safety=int(sl["b_breaks_a"]),
        recovery_rescue_vs_safety=int(sr["b_rescues_a"]), recovery_break_vs_safety=int(sr["b_breaks_a"]),
        target_action=target, status=status, unresolved=unresolved,
        strict_support_margin=margin, pairwise=pairs,
    )


def finalize(manifest_path: Path, result_directory: Path, output_directory: Path) -> dict[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError((output_directory, "refusing overwrite"))
    manifest = json.loads(manifest_path.read_text())
    verify_content_hash(manifest, manifest_path)
    tasks = list(manifest["tasks"])
    expected = {f"task_{int(task['task_index']):05d}.json" for task in tasks}
    actual = {path.name for path in result_directory.glob("task_*.json")}
    if actual != expected:
        raise RuntimeError(("result coverage mismatch", len(expected), len(actual)))
    rows: list[dict[str, Any]] = []
    by_index = {int(task["task_index"]): task for task in tasks}
    for index in range(len(tasks)):
        row = json.loads((result_directory / f"task_{index:05d}.json").read_text())
        validate_result(row, by_index[index], manifest)
        rows.append(row)
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["decision_id"], {}).setdefault(int(row["action"]), []).append(row)
    decision_by_id = {row["decision_id"]: row for row in manifest["decisions"]}
    if set(grouped) != set(decision_by_id):
        raise RuntimeError("decision/result coverage mismatch")
    evidence = {decision_id: summarize(decision_id, grouped[decision_id]) for decision_id in sorted(grouped)}
    evidence_rows = []
    for decision_id, item in evidence.items():
        source = decision_by_id[decision_id]
        evidence_rows.append({
            "decision_id": decision_id, "pass_id": source["pass_id"],
            "downstream_policy_hash": source["downstream_policy_hash"],
            "root_source_id": source["root_source_id"], "split": source["split"],
            **asdict(item),
        })
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_text(output_directory / "paired_branch_outcomes.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    atomic_csv(output_directory / "paired_branch_outcomes.csv", rows)
    atomic_csv(output_directory / "three_way_label_uncertainty.csv", evidence_rows)
    ordered = [decision_by_id[item["decision_id"]] for item in evidence_rows]
    features = np.stack([np.asarray(row["feature"], dtype=np.float32) for row in ordered])
    targets = np.asarray([-1 if row["target_action"] is None else int(row["target_action"]) for row in evidence_rows], dtype=np.int8)
    break_vs_safety = np.asarray([
        [0.0, row["local_break_vs_safety"]/FUTURES, row["recovery_break_vs_safety"]/FUTURES]
        for row in evidence_rows
    ], dtype=np.float32)
    split = np.asarray([row["split"] for row in evidence_rows])
    snapshot = output_directory / "mode_dataset_snapshot.npz"
    np.savez(snapshot, features=features, targets=targets, break_vs_safety=break_vs_safety,
             splits=split, decision_ids=np.asarray([row["decision_id"] for row in evidence_rows]))
    train_roots = {row["root_source_id"] for row in evidence_rows if row["split"] == "train"}
    val_roots = {row["root_source_id"] for row in evidence_rows if row["split"] == "validation"}
    if train_roots & val_roots:
        raise RuntimeError("root-source split leakage")
    unresolved = sum(bool(row["unresolved"]) for row in evidence_rows)
    atomic_text(output_directory / "three_way_label_uncertainty_report.md", "\n".join([
        f"# Stage-2 three-way label audit: {manifest['stage']}", "",
        f"- Complete branch rows: {len(rows)}",
        f"- Decision inputs: {len(evidence_rows)}",
        f"- Matched futures/input/action: {FUTURES}",
        f"- Unresolved inputs: {unresolved}",
        f"- Practical success margin: {EPSILON_Q:.3f}",
        "- A class is assigned only when it beats both alternatives with a conservative paired CI,",
        "  or when all three are supported equivalent and Safety is preferred.",
        "- Unsupported comparisons remain unresolved and receive no training class.", "",
    ]))
    result = {
        "schema": "semantic_mode_finalized_dataset_v1",
        "status": "COMPLETE_FROZEN_READY_FOR_TRAINING",
        "stage": manifest["stage"], "pass_id": manifest["pass_id"],
        "input_dimension": 214, "output_actions": [ACTION_NAMES[key] for key in sorted(ACTION_NAMES)],
        "source_branch_manifest": str(manifest_path.resolve()),
        "source_branch_manifest_sha256": sha256(manifest_path),
        "source_branch_manifest_content_sha256": manifest["content_sha256"],
        "result_count": len(rows), "decision_count": len(evidence_rows),
        "expected_futures_per_input_per_action": FUTURES,
        "dataset_snapshot": str(snapshot.resolve()), "dataset_snapshot_sha256": sha256(snapshot),
        "three_way_label_uncertainty_csv": str((output_directory / "three_way_label_uncertainty.csv").resolve()),
        "resolved_train": sum(row["split"] == "train" and not row["unresolved"] for row in evidence_rows),
        "resolved_validation": sum(row["split"] == "validation" and not row["unresolved"] for row in evidence_rows),
        "unresolved_count": unresolved,
        "epsilon_q": EPSILON_Q, "paired_ci_level": CI_LEVEL,
        "deformation_tiebreak_used": False,
        "complete_paired_branches": True, "train_validation_source_overlap": 0,
        "all_results_complete": True, "execution_error_count": 0,
        "training_launched": False, "final_test_used": False,
    }
    result["content_sha256"] = content_hash(result)
    atomic_json(output_directory / "mode_dataset_manifest.json", result)
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
