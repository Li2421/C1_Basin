"""Freeze matched S/L/R Stage-2 branch tasks from a generic anchor manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from mode_common import (
    ACTION_NAMES, DIRECT_CHECKPOINT, DIRECT_SHA256, ETA_CHECKPOINT, ETA_SHA256,
    FUTURES, FUTURE_ROOT_SEED, HERE, HORIZON, LOCAL_HEAD,
    LOCAL_THRESHOLD_PROBABILITY, MAX_BRANCH_CONTINUATIONS, MAX_PHYSICAL_STEPS,
    PILOT_ROOT, SEMANTIC_ROOT, content_hash, logit, sha256, verify_content_hash,
)


DEFAULT_STATE_MANIFEST = PILOT_ROOT / "decision_state_manifest.meta.json"
MANIFEST_DIR = HERE / "branch_manifests"
STAGES = {
    "mode_bootstrap_local_downstream": 60,
    "mode_confirmation_local_downstream": 30,
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_states(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    verify_content_hash(payload, path)
    if payload.get("status") not in {"COMPLETE_FROZEN", "FROZEN_COMPLETE", "COMPLETE_AND_FROZEN"}:
        raise RuntimeError((path, "generic state manifest is not complete/frozen"))
    if payload.get("decision_kind") not in {"entry", "normal", "mode"}:
        raise RuntimeError((path, "not a NORMAL-state manifest"))
    rows = [dict(row) for row in payload.get("states", []) if row.get("split") in {"train", "validation"}]
    required = {
        "state_id", "root_source_id", "split", "state_file", "state_sha256",
        "absolute_step", "feature", "current_u_flow", "current_u_safe",
        "current_flow_realization_id", "current_flow_key_data",
    }
    root_splits: dict[str, str] = {}
    root_counts: dict[str, int] = {}
    state_ids: set[str] = set()
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise RuntimeError((row.get("state_id"), "missing state fields", sorted(missing)))
        if row["state_id"] in state_ids:
            raise RuntimeError((row["state_id"], "duplicate state ID"))
        state_ids.add(row["state_id"])
        if not bool(row.get("selection_outcome_blind")):
            raise RuntimeError((row["state_id"], "anchor selection was not outcome-blind"))
        if len(row["feature"]) != 214 or not 0 <= int(row["absolute_step"]) < HORIZON:
            raise RuntimeError((row["state_id"], "invalid feature/time"))
        if sha256(row["state_file"]) != row["state_sha256"]:
            raise RuntimeError((row["state_id"], "state hash mismatch"))
        prior = root_splits.setdefault(row["root_source_id"], row["split"])
        if prior != row["split"]:
            raise RuntimeError((row["root_source_id"], "root crosses splits"))
        root_counts[row["root_source_id"]] = root_counts.get(row["root_source_id"], 0) + 1
    if not rows or max(root_counts.values(), default=0) > 2:
        raise RuntimeError("generic anchor manifest is empty or exceeds two anchors/source")
    return payload, rows


def _rank(stage: str, row: dict[str, Any]) -> str:
    return content_hash(["semantic_mode_stage2_v1", stage, row["root_source_id"], row["state_id"]])


def select_states(rows: list[dict[str, Any]], stage: str, maximum: int) -> list[dict[str, Any]]:
    quotas = {"train": maximum * 2 // 3, "validation": maximum - maximum * 2 // 3}
    selected: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        candidates = sorted((row for row in rows if row["split"] == split), key=lambda row: _rank(stage, row))
        unique, extras, seen = [], [], set()
        for row in candidates:
            target = unique if row["root_source_id"] not in seen else extras
            target.append(row)
            seen.add(row["root_source_id"])
        selected.extend((unique + extras)[:quotas[split]])
    return selected


def future_id(stage: str, decision_id: str, index: int) -> int:
    return int(content_hash(["semantic_mode_future_v1", stage, decision_id, index])[:8], 16)


def prior_branch_budget(exclude: Path) -> tuple[int, int, list[str]]:
    continuations = steps = 0
    included: list[str] = []
    roots = [SEMANTIC_ROOT / "stage1_state_driven_local/branch_manifests", MANIFEST_DIR]
    for root in roots:
        for path in sorted(root.glob("*.json")) if root.exists() else ():
            if path.resolve() == exclude.resolve():
                continue
            value = json.loads(path.read_text())
            verify_content_hash(value, path)
            budget = value.get("budget", {})
            if "new_continuations" in budget and "maximum_physical_steps" in budget:
                continuations += int(budget["new_continuations"])
                steps += int(budget["maximum_physical_steps"])
                included.append(str(path.resolve()))
    return continuations, steps, included


def build(stage: str, output: Path, state_manifest: Path = DEFAULT_STATE_MANIFEST) -> dict[str, Any]:
    if stage not in STAGES or output.exists():
        raise ValueError((stage, output, "unknown stage or output exists"))
    for path, expected in ((DIRECT_CHECKPOINT, DIRECT_SHA256), (ETA_CHECKPOINT, ETA_SHA256)):
        if sha256(path) != expected:
            raise RuntimeError((path, "checkpoint hash mismatch"))
    if not LOCAL_HEAD.is_file():
        raise FileNotFoundError(LOCAL_HEAD)
    source, rows = load_states(state_manifest)
    selected = select_states(rows, stage, STAGES[stage])
    if not selected or not {row["split"] for row in selected} >= {"train", "validation"}:
        raise RuntimeError("both train and validation anchors are required")
    downstream = {
        "kind": "FROZEN_STATE_DRIVEN_LOCAL", "checkpoint": str(LOCAL_HEAD.resolve()),
        "sha256": sha256(LOCAL_HEAD), "threshold_probability": LOCAL_THRESHOLD_PROBABILITY,
        "threshold_logit": logit(LOCAL_THRESHOLD_PROBABILITY),
    }
    downstream_hash = content_hash(downstream)
    decisions, tasks, rollout_ids = [], [], set()
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
            if rollout_id in rollout_ids:
                raise RuntimeError("future rollout ID collision")
            rollout_ids.add(rollout_id)
            stream = f"{decision_id}|future{future_index:02d}|{rollout_id:08x}"
            rng_hash = content_hash({
                "root": FUTURE_ROOT_SEED, "rollout_id": rollout_id,
                "first_future_step": int(state["absolute_step"]) + 1,
            })
            for action in ACTION_NAMES:
                tasks.append({
                    **decision, "task_index": len(tasks), "task_id": f"{stream}|a{action}",
                    "action": action, "action_name": ACTION_NAMES[action],
                    "future_index": future_index, "future_rollout_id": rollout_id,
                    "future_stream_id": stream, "future_rng_state_hash": rng_hash,
                })
    prior_n, prior_steps, included = prior_branch_budget(output)
    new_n = len(tasks)
    max_steps = sum(HORIZON - int(task["absolute_step"]) for task in tasks)
    if prior_n + new_n > MAX_BRANCH_CONTINUATIONS or prior_steps + max_steps > MAX_PHYSICAL_STEPS:
        raise RuntimeError(("semantic-program budget exceeded", prior_n + new_n, prior_steps + max_steps))
    implementation = {
        "mode_common": HERE / "mode_common.py",
        "branch_runner": HERE / "run_mode_branches.py",
    }
    result = {
        "schema": "semantic_mode_paired_branch_manifest_v1",
        "status": "FROZEN_READY_FOR_EXECUTION", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage, "pass_id": stage, "decision_kind": "normal_mode_3way",
        "source_state_manifest": str(state_manifest.resolve()),
        "source_state_manifest_sha256": sha256(state_manifest),
        "source_state_manifest_content_sha256": source["content_sha256"],
        "direct_g": {"path": str(DIRECT_CHECKPOINT), "sha256": DIRECT_SHA256},
        "structured_eta": {"path": str(ETA_CHECKPOINT), "sha256": ETA_SHA256},
        "downstream_policy": downstream, "downstream_policy_hash": downstream_hash,
        "actions": {str(key): value for key, value in ACTION_NAMES.items()},
        "action_semantics": {
            "0": "Safety now, then the frozen state-driven S/L policy",
            "1": "one Direct-g correction now, then the same frozen state-driven S/L policy",
            "2": "predict eta once now, then persistent dense structured-eta recovery to terminal",
        },
        "contains_periodic_timing": False, "recovery_exit_enabled": False,
        "matched_future_semantics": {
            "count_per_decision": FUTURES, "future_root_seed": FUTURE_ROOT_SEED,
            "current_flow_shared_across_actions": True, "future_stream_shared_across_actions": True,
            "policy_inference_does_not_consume_flow_rng": True,
        },
        "selection": {
            "generic_outcome_blind": True, "maximum_two_anchors_per_source": True,
            "state_manifest_is_replaceable_cli_input": True,
            "method": "deterministic hash, source-diverse first", "selected_inputs": len(decisions),
            "split_counts": {split: sum(row["split"] == split for row in decisions) for split in ("train", "validation")},
        },
        "implementation_hashes": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in implementation.items()
        },
        "budget": {
            "new_continuations": new_n, "maximum_physical_steps": max_steps,
            "prior_manifest_continuations": prior_n, "prior_manifest_maximum_steps": prior_steps,
            "global_after_continuations": prior_n + new_n,
            "global_after_maximum_steps": prior_steps + max_steps,
            "included_prior_branch_manifests": included,
            "global_limits": {"continuations": MAX_BRANCH_CONTINUATIONS, "physical_steps": MAX_PHYSICAL_STEPS},
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
    parser.add_argument("--state-manifest", type=Path, default=DEFAULT_STATE_MANIFEST)
    args = parser.parse_args()
    result = build(args.stage, args.output, args.state_manifest)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"tasks", "decisions"}}, indent=2))


if __name__ == "__main__":
    main()
