"""Freeze matched S-now versus L-now branch tasks for Stage 1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from local_common import (
    BASE, DIRECT_CHECKPOINT, DIRECT_SHA256, FUTURES, FUTURE_ROOT_SEED, HERE,
    HORIZON, MAX_CONTINUATIONS, MAX_PHYSICAL_STEPS, content_hash, sha256,
    verify_content_hash,
)


STAGES = {
    "local_bootstrap_safety": {"max_inputs": 24, "downstream": "SAFETY_ONLY"},
    "local_second_pass": {"max_inputs": 12, "downstream": "LEARNED_REQUIRED"},
}
SOURCE_META = BASE / "decision_state_manifest.meta.json"
MANIFEST_DIR = HERE / "branch_manifests"
LOCAL_COMMON = HERE / "local_common.py"
BRANCH_RUNNER = HERE / "run_local_branches.py"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_states() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(SOURCE_META.read_text())
    verify_content_hash(payload, SOURCE_META)
    if payload.get("status") != "COMPLETE_FROZEN" or payload.get("decision_kind") != "entry":
        raise RuntimeError("generic Safety-anchor manifest is not frozen/compatible")
    rows = [dict(row) for row in payload["states"] if row["split"] in {"train", "validation"}]
    roots: dict[str, str] = {}
    counts: dict[str, int] = {}
    required = {
        "state_id", "root_source_id", "split", "state_file", "state_sha256",
        "absolute_step", "feature", "current_u_flow", "current_u_safe",
        "current_flow_realization_id", "current_flow_key_data",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise RuntimeError((row.get("state_id"), "missing state fields", sorted(missing)))
        if sha256(row["state_file"]) != row["state_sha256"]:
            raise RuntimeError((row["state_id"], "state hash mismatch"))
        if len(row["feature"]) != 214 or not row.get("selection_outcome_blind"):
            raise RuntimeError((row["state_id"], "non-generic/malformed anchor"))
        previous = roots.setdefault(row["root_source_id"], row["split"])
        if previous != row["split"]:
            raise RuntimeError((row["root_source_id"], "source crosses splits"))
        counts[row["root_source_id"]] = counts.get(row["root_source_id"], 0) + 1
    if max(counts.values(), default=0) > 2:
        raise RuntimeError("source collection exceeds two generic anchors/source")
    return payload, rows


def _rank(stage: str, row: dict[str, Any]) -> str:
    return content_hash(["semantic_local_stage1_v1", stage, row["root_source_id"], row["state_id"]])


def _select(rows: list[dict[str, Any]], stage: str, maximum: int) -> list[dict[str, Any]]:
    quotas = {"train": maximum * 2 // 3, "validation": maximum - maximum * 2 // 3}
    selected: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        candidates = sorted((row for row in rows if row["split"] == split), key=lambda row: _rank(stage, row))
        unique, extras, seen = [], [], set()
        for row in candidates:
            (unique if row["root_source_id"] not in seen else extras).append(row)
            seen.add(row["root_source_id"])
        selected.extend((unique + extras)[:quotas[split]])
    return selected


def _head_record(stage: str, checkpoint: Path | None) -> dict[str, Any]:
    expected = STAGES[stage]["downstream"]
    if expected == "SAFETY_ONLY":
        if checkpoint is not None:
            raise ValueError("bootstrap downstream does not accept a head")
        return {"kind": "SAFETY_ONLY", "checkpoint": None, "sha256": None}
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError("second pass requires a frozen learned S/L head")
    return {"kind": "LEARNED_HEAD", "checkpoint": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}


def future_id(stage: str, decision_id: str, index: int) -> int:
    return int(content_hash(["semantic_local_future_v1", stage, decision_id, index])[:8], 16)


def _existing_budget(exclude: Path) -> tuple[int, int]:
    continuations = steps = 0
    for path in MANIFEST_DIR.glob("*.json") if MANIFEST_DIR.exists() else ():
        if path.resolve() == exclude.resolve():
            continue
        value = json.loads(path.read_text())
        verify_content_hash(value, path)
        if value.get("schema") == "semantic_local_paired_branch_manifest_v1":
            continuations += int(value["budget"]["new_continuations"])
            steps += int(value["budget"]["maximum_physical_steps"])
    return continuations, steps


def build(stage: str, output: Path, downstream_checkpoint: Path | None = None) -> dict[str, Any]:
    if stage not in STAGES or output.exists():
        raise ValueError((stage, output, "unknown stage or output exists"))
    if sha256(DIRECT_CHECKPOINT) != DIRECT_SHA256:
        raise RuntimeError("high-water Direct-g checkpoint hash mismatch")
    source, rows = load_states()
    selected = _select(rows, stage, int(STAGES[stage]["max_inputs"]))
    downstream = _head_record(stage, downstream_checkpoint)
    downstream_hash = content_hash(downstream)
    decisions, tasks = [], []
    used_ids: set[int] = set()
    for state in selected:
        decision_id = f"{stage}|{state['state_id']}"
        decision = {
            "decision_id": decision_id, "pass_id": stage,
            "root_source_id": state["root_source_id"], "split": state["split"],
            "state_id": state["state_id"], "state_file": state["state_file"],
            "state_sha256": state["state_sha256"], "absolute_step": int(state["absolute_step"]),
            "feature": state["feature"], "current_u_flow": state["current_u_flow"],
            "current_u_safe": state["current_u_safe"],
            "current_flow_realization_id": state["current_flow_realization_id"],
            "current_flow_key_data": state["current_flow_key_data"],
            "downstream_policy_hash": downstream_hash,
        }
        decisions.append(decision)
        for future_index in range(FUTURES):
            rollout_id = future_id(stage, decision_id, future_index)
            if rollout_id in used_ids:
                raise RuntimeError("future rollout-id collision")
            used_ids.add(rollout_id)
            stream = f"{decision_id}|future{future_index:02d}|{rollout_id:08x}"
            rng_hash = content_hash({
                "root": FUTURE_ROOT_SEED, "rollout_id": rollout_id,
                "first_future_step": int(state["absolute_step"]) + 1,
            })
            for action in (0, 1):
                tasks.append({
                    **decision, "task_index": len(tasks), "task_id": f"{stream}|a{action}",
                    "action": action, "future_index": future_index,
                    "future_rollout_id": rollout_id, "future_stream_id": stream,
                    "future_rng_state_hash": rng_hash,
                })
    prior_continuations, prior_steps = _existing_budget(output)
    new_continuations = len(tasks)
    maximum_steps = sum(HORIZON - int(task["absolute_step"]) for task in tasks)
    if prior_continuations + new_continuations > MAX_CONTINUATIONS:
        raise RuntimeError("Stage-1 continuation budget exceeded")
    if prior_steps + maximum_steps > MAX_PHYSICAL_STEPS:
        raise RuntimeError("Stage-1 physical-step budget exceeded")
    result = {
        "schema": "semantic_local_paired_branch_manifest_v1",
        "status": "FROZEN_READY_FOR_EXECUTION", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage, "pass_id": stage, "decision_kind": "local", "system": "STATE_DRIVEN_LOCAL",
        "source_state_manifest": str(SOURCE_META), "source_state_manifest_sha256": sha256(SOURCE_META),
        "source_state_manifest_content_sha256": source["content_sha256"],
        "source_split_manifest_sha256": source["source_manifest_sha256"],
        "direct_g": {"path": str(DIRECT_CHECKPOINT), "sha256": DIRECT_SHA256},
        "implementation_hashes": {
            "local_common": {"path": str(LOCAL_COMMON), "sha256": sha256(LOCAL_COMMON)},
            "branch_runner": {"path": str(BRANCH_RUNNER), "sha256": sha256(BRANCH_RUNNER)},
        },
        "downstream_policy": downstream, "downstream_policy_hash": downstream_hash,
        "action_semantics": {"0": "SAFETY_NOW then frozen downstream S/L policy", "1": "LOCAL_CORRECTION_NOW then frozen downstream S/L policy"},
        "matched_future_semantics": {
            "count_per_decision": FUTURES, "future_root_seed": FUTURE_ROOT_SEED,
            "current_flow_shared_across_actions": True, "future_stream_shared_across_actions": True,
            "policy_inference_does_not_consume_flow_rng": True,
        },
        "selection": {
            "generic_outcome_blind": True, "maximum_two_anchors_per_source": True,
            "method": "deterministic hash, source-diverse first", "selected_inputs": len(decisions),
            "split_counts": {split: sum(row["split"] == split for row in decisions) for split in ("train", "validation")},
        },
        "budget": {
            "new_continuations": new_continuations, "maximum_physical_steps": maximum_steps,
            "global_after_continuations": prior_continuations + new_continuations,
            "global_after_maximum_steps": prior_steps + maximum_steps,
            "global_limits": {"continuations": MAX_CONTINUATIONS, "physical_steps": MAX_PHYSICAL_STEPS},
        },
        "decision_count": len(decisions), "task_count": len(tasks),
        "decisions": decisions, "tasks": tasks,
    }
    result["content_sha256"] = content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--downstream-checkpoint", type=Path)
    args = parser.parse_args()
    result = build(args.stage, args.output, args.downstream_checkpoint)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"tasks", "decisions"}}, indent=2))


if __name__ == "__main__":
    main()
