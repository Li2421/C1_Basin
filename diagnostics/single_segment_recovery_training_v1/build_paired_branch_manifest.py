"""Freeze bounded, matched ETA branch-label tasks for Stage B/C.

The input state manifest must already be complete and frozen.  This script does
not inspect any counterfactual outcome and never launches a rollout.  It picks
states by source-diverse deterministic hashing, assigns exactly 32 future
streams per input, and emits paired action-0/action-1 tasks with a single
downstream-policy hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1")
PROTOCOL = HERE / "protocol.json"
SPLITS = HERE / "source_split_manifest.json"
MANIFEST_DIR = HERE / "branch_manifests"
HORIZON = 850
FUTURES = 32
MAX_CONTINUATIONS = 12_000
MAX_STEPS = 6_000_000
FUTURE_ROOT_SEED = 2026100201

STAGES: dict[str, dict[str, Any]] = {
    "exit_bootstrap_never_exit": {
        "decision_kind": "exit", "max_inputs": 24,
        "entry_policy": "NOT_APPLICABLE", "exit_policy": "NEVER_EXIT",
    },
    "exit_second_pass": {
        "decision_kind": "exit", "max_inputs": 12,
        "entry_policy": "NOT_APPLICABLE", "exit_policy": "LEARNED_REQUIRED",
    },
    "entry_bootstrap_no_entry": {
        "decision_kind": "entry", "max_inputs": 24,
        "entry_policy": "NEVER_ENTER", "exit_policy": "LEARNED_REQUIRED",
    },
    "entry_second_pass": {
        "decision_kind": "entry", "max_inputs": 12,
        "entry_policy": "LEARNED_REQUIRED", "exit_policy": "LEARNED_REQUIRED",
    },
    "policy_iteration_exit": {
        "decision_kind": "exit", "max_inputs": 12,
        "entry_policy": "NOT_APPLICABLE", "exit_policy": "LEARNED_REQUIRED",
    },
    "policy_iteration_entry": {
        "decision_kind": "entry", "max_inputs": 12,
        "entry_policy": "LEARNED_REQUIRED", "exit_policy": "LEARNED_REQUIRED",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def verify_semantic_hash(payload: dict, path: Path) -> None:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if not claimed or canonical_hash(body) != claimed:
        raise RuntimeError((str(path), "semantic hash mismatch"))


def load_frozen_states(path: Path, decision_kind: str) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    verify_semantic_hash(payload, path)
    status = str(payload.get("status", ""))
    if status not in {"COMPLETE_FROZEN", "FROZEN_COMPLETE", "COMPLETE_AND_FROZEN"}:
        raise RuntimeError((path, "state collector is not complete and frozen", status))
    if payload.get("decision_kind") != decision_kind:
        raise RuntimeError((path, "decision-kind mismatch", payload.get("decision_kind"), decision_kind))
    rows = payload.get("states")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError((path, "no collected decision states"))
    required = {
        "state_id", "root_source_id", "split", "state_file", "state_sha256",
        "absolute_step", "feature", "current_u_flow", "current_u_safe",
        "current_flow_realization_id",
        "current_flow_key_data",
    }
    seen = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError((row.get("state_id"), "missing fields", sorted(missing)))
        if row["state_id"] in seen:
            raise RuntimeError((row["state_id"], "duplicate state ID"))
        seen.add(row["state_id"])
        if row["split"] not in ("train", "validation"):
            raise RuntimeError((row["state_id"], "labeling state outside train/validation"))
        state_file = Path(row["state_file"])
        if not state_file.is_file() or sha256(state_file) != row["state_sha256"]:
            raise RuntimeError((row["state_id"], "state file/hash mismatch"))
        if len(row["feature"]) != 214:
            raise RuntimeError((row["state_id"], "feature dimension"))
        if not 0 <= int(row["absolute_step"]) < HORIZON:
            raise RuntimeError((row["state_id"], "invalid absolute step"))
        if decision_kind == "exit":
            for name in ("eta_latched", "entry_step", "recovery_transitions"):
                if name not in row:
                    raise RuntimeError((row["state_id"], "missing exit-state memory", name))
            if len(row["eta_latched"]) != 3 or int(row["recovery_transitions"]) < 1:
                raise RuntimeError((row["state_id"], "invalid ETA recovery memory"))
            if int(row["entry_step"]) >= int(row["absolute_step"]):
                raise RuntimeError((row["state_id"], "exit query must follow a recovery transition"))
    return payload, rows


def policy_record(kind: str, checkpoint: str | None) -> dict[str, Any]:
    if kind in {"NEVER_EXIT", "NEVER_ENTER", "NOT_APPLICABLE"}:
        if checkpoint is not None:
            raise ValueError((kind, "does not accept a checkpoint"))
        return {"kind": kind, "checkpoint": None, "sha256": None}
    if kind != "LEARNED_REQUIRED" or checkpoint is None:
        raise ValueError((kind, "requires a learned checkpoint"))
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"kind": "LEARNED_HEAD", "checkpoint": str(path), "sha256": sha256(path)}


def _rank(stage: str, row: dict) -> str:
    return hashlib.sha256(
        f"single_segment_branch_state_v1|{stage}|{row['root_source_id']}|{row['state_id']}".encode()
    ).hexdigest()


def source_diverse_select(rows: list[dict], stage: str, maximum: int) -> list[dict]:
    """Select 2/3 train, 1/3 validation, one state/root before any reuse."""

    # The single bounded policy-improvement round is collected exclusively on
    # TRAIN sources.  Bootstrap/second-pass collections retain their frozen
    # 2/3 train, 1/3 validation split.
    if stage.startswith("policy_iteration_"):
        quotas = {"train": maximum, "validation": 0}
    else:
        train_quota = maximum * 2 // 3
        quotas = {"train": train_quota, "validation": maximum - train_quota}
    selected: list[dict] = []
    for split in ("train", "validation"):
        candidates = sorted((row for row in rows if row["split"] == split), key=lambda row: _rank(stage, row))
        first_by_root: list[dict] = []
        extras: list[dict] = []
        roots = set()
        for row in candidates:
            if row["root_source_id"] not in roots:
                roots.add(row["root_source_id"])
                first_by_root.append(row)
            else:
                extras.append(row)
        selected.extend((first_by_root + extras)[:quotas[split]])
    return selected


def future_rollout_id(stage: str, decision_id: str, index: int) -> int:
    digest = hashlib.sha256(
        f"single_segment_future_v1|{stage}|{decision_id}|{index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "little")


def existing_budget(exclude: Path) -> tuple[int, int, list[str]]:
    continuations = steps = 0
    paths: list[str] = []
    if not MANIFEST_DIR.exists():
        return continuations, steps, paths
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        if path.resolve() == exclude.resolve():
            continue
        try:
            payload = json.loads(path.read_text())
            verify_semantic_hash(payload, path)
        except Exception:
            continue
        if payload.get("schema") != "single_segment_paired_branch_manifest_v1":
            continue
        continuations += int(payload["budget"]["new_continuations"])
        steps += int(payload["budget"]["maximum_physical_steps"])
        paths.append(str(path))
    return continuations, steps, paths


def source_collection_budget() -> tuple[int, int, list[str]]:
    """Return pre-branch continuation/step use, counted once globally."""

    continuations = 0
    steps = 0
    assets: list[str] = []
    for path in sorted((HERE / "runs").glob("source_collection_runtime_shard*.json")):
        payload = json.loads(path.read_text())
        steps += int(payload.get("new_physical_steps", 0))
        assets.append(str(path))
    # The completed collector freezes this integrity artifact alongside the
    # exit-state manifest.  Count the collection once from the audit's actual
    # use (rather than summing its per-shard records as well).  Retain the
    # older finalizer output only as a mutually exclusive compatibility path.
    recovery_audit = HERE / "eta_recovery_state_collection_integrity.json"
    legacy_recovery = HERE / "eta_recovery_decision_state_manifest.json"
    if recovery_audit.is_file():
        payload = json.loads(recovery_audit.read_text())
        verify_semantic_hash(payload, recovery_audit)
        if (payload.get("schema") != "single_segment_eta_recovery_state_collection_integrity_v1"
                or payload.get("status") != "PASS"):
            raise RuntimeError((recovery_audit, "recovery collection audit is not a frozen PASS"))
        budget = payload["budget_use"]
        continuations += int(budget["new_continuations"])
        steps += int(budget["new_physical_simulation_steps"])
        assets.append(str(recovery_audit))
    elif legacy_recovery.is_file():
        payload = json.loads(legacy_recovery.read_text())
        verify_semantic_hash(payload, legacy_recovery)
        budget = payload["collection_budget"]
        continuations += int(budget["new_prefix_continuations"])
        steps += int(budget["new_physical_steps"])
        assets.append(str(legacy_recovery))
    policy_iteration_audit = HERE / "policy_iteration_state_collection_integrity.json"
    if policy_iteration_audit.is_file():
        payload = json.loads(policy_iteration_audit.read_text())
        verify_semantic_hash(payload, policy_iteration_audit)
        if (payload.get("schema")
                != "single_segment_policy_iteration_state_collection_integrity_v1"
                or payload.get("status") != "PASS"):
            raise RuntimeError((policy_iteration_audit, "policy-iteration collection audit is not PASS"))
        budget = payload["budget_use"]
        continuations += int(budget["new_continuations"])
        steps += int(budget["new_physical_steps"])
        assets.append(str(policy_iteration_audit))
    return continuations, steps, assets


def build_manifest(
    *, stage: str, state_manifest: Path, output: Path,
    entry_checkpoint: str | None = None, exit_checkpoint: str | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(("unknown stage", stage))
    if output.exists():
        raise RuntimeError((output, "refusing to overwrite frozen manifest"))
    protocol = json.loads(PROTOCOL.read_text())
    splits = json.loads(SPLITS.read_text())
    verify_semantic_hash(splits, SPLITS)
    config = STAGES[stage]
    source_payload, rows = load_frozen_states(state_manifest, config["decision_kind"])
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for prior_path in sorted(MANIFEST_DIR.glob("*.json")):
        prior = json.loads(prior_path.read_text())
        verify_semantic_hash(prior, prior_path)
        if prior.get("schema") == "single_segment_paired_branch_manifest_v1" and prior.get("stage") == stage:
            raise RuntimeError((stage, "duplicate frozen stage manifest", str(prior_path)))
    selected = source_diverse_select(rows, stage, int(config["max_inputs"]))
    if not selected:
        raise RuntimeError("no eligible states after deterministic bounded selection")
    entry_policy = policy_record(config["entry_policy"], entry_checkpoint)
    exit_policy = policy_record(config["exit_policy"], exit_checkpoint)
    downstream = {
        "system": "ETA", "entry": entry_policy, "exit": exit_policy,
        "one_segment_only": True,
    }
    downstream_hash = canonical_hash(downstream)
    decisions = []
    tasks = []
    used_future_ids: set[int] = set()
    for state in selected:
        decision_id = f"{stage}|{state['state_id']}"
        decision = {
            "pass_id": stage,
            "decision_id": decision_id,
            "decision_kind": config["decision_kind"],
            "root_source_id": state["root_source_id"],
            "split": state["split"],
            "state_id": state["state_id"],
            "state_file": state["state_file"],
            "state_sha256": state["state_sha256"],
            "absolute_step": int(state["absolute_step"]),
            "feature": state["feature"],
            "current_u_flow": state["current_u_flow"],
            "current_u_safe": state["current_u_safe"],
            "current_flow_realization_id": state["current_flow_realization_id"],
            "current_flow_key_data": state.get("current_flow_key_data"),
            "eta_latched": state.get("eta_latched"),
            "entry_step": state.get("entry_step"),
            "recovery_transitions": state.get("recovery_transitions", 0),
            "downstream_policy_hash": downstream_hash,
        }
        decisions.append(decision)
        for future_index in range(FUTURES):
            rollout_id = future_rollout_id(stage, decision_id, future_index)
            if rollout_id in used_future_ids:
                raise RuntimeError(("future rollout ID collision", rollout_id))
            used_future_ids.add(rollout_id)
            stream_id = f"{decision_id}|future{future_index:02d}|{rollout_id:08x}"
            future_rng_state_hash = canonical_hash({
                "root": FUTURE_ROOT_SEED,
                "rollout_id": rollout_id,
                "first_future_step": int(state["absolute_step"]) + 1,
            })
            for action in (0, 1):
                tasks.append({
                    "task_index": len(tasks), "task_id": f"{stream_id}|a{action}",
                    **decision, "action": action, "future_index": future_index,
                    "future_rollout_id": rollout_id, "future_stream_id": stream_id,
                    "future_rng_state_hash": future_rng_state_hash,
                })
    new_continuations = len(tasks)
    maximum_steps = sum(HORIZON - int(task["absolute_step"]) for task in tasks)
    prior_continuations, prior_steps, prior_paths = existing_budget(output)
    source_continuations, source_steps, source_assets = source_collection_budget()
    if source_continuations + prior_continuations + new_continuations > MAX_CONTINUATIONS:
        raise RuntimeError("global continuation budget would be exceeded")
    if source_steps + prior_steps + maximum_steps > MAX_STEPS:
        raise RuntimeError("global physical-step budget would be exceeded")
    result = {
        "schema": "single_segment_paired_branch_manifest_v1",
        "status": "FROZEN_READY_FOR_EXECUTION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "pass_id": stage,
        "decision_kind": config["decision_kind"],
        "system": "ETA",
        "state_source_manifest": str(state_manifest.resolve()),
        "state_source_manifest_sha256": sha256(state_manifest),
        "state_source_manifest_content_sha256": source_payload["content_sha256"],
        "protocol_path": str(PROTOCOL), "protocol_sha256": sha256(PROTOCOL),
        "source_split_manifest_path": str(SPLITS), "source_split_manifest_sha256": sha256(SPLITS),
        "selection": {
            "outcome_blind": True, "method": "deterministic hash rank; source-diverse first",
            "maximum_inputs": int(config["max_inputs"]),
            "selected_inputs": len(decisions),
            "split_counts": {name: sum(row["split"] == name for row in decisions) for name in ("train", "validation")},
            "uses_future_failure_or_counterfactual_outcome": False,
        },
        "matched_future_semantics": {
            "count_per_decision": FUTURES,
            "current_flow_action_frozen_from_decision_state": True,
            "current_flow_shared_across_actions": True,
            "future_rollout_id_shared_across_actions": True,
            "after_divergence_same_keys_but_state_conditional_actions_may_differ": True,
            "flow_rng_consumed_by_policy_only_not_bookkeeping": True,
            "future_root_seed": FUTURE_ROOT_SEED,
        },
        "action_semantics": {
            "exit": {"0": "CONTINUE_NOW then frozen exit policy", "1": "EXIT_NOW then Safety forever"},
            "entry": {"0": "WAIT_NOW then frozen incumbent entry/exit policies", "1": "ENTER_NOW then frozen exit policy"},
        }[config["decision_kind"]],
        "downstream_policy": downstream,
        "downstream_policy_hash": downstream_hash,
        "budget": {
            "new_continuations": new_continuations,
            "maximum_physical_steps": maximum_steps,
            "source_collection_continuations": source_continuations,
            "source_collection_physical_steps": source_steps,
            "source_collection_assets": source_assets,
            "prior_frozen_manifest_continuations": prior_continuations,
            "prior_frozen_manifest_maximum_steps": prior_steps,
            "global_after_continuations": source_continuations + prior_continuations + new_continuations,
            "global_after_maximum_steps": source_steps + prior_steps + maximum_steps,
            "global_limits": {"continuations": MAX_CONTINUATIONS, "physical_steps": MAX_STEPS},
            "prior_manifests": prior_paths,
        },
        "decision_count": len(decisions), "task_count": len(tasks),
        "decisions": decisions, "tasks": tasks,
    }
    result["content_sha256"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--state-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-checkpoint")
    parser.add_argument("--exit-checkpoint")
    args = parser.parse_args()
    result = build_manifest(
        stage=args.stage, state_manifest=args.state_manifest, output=args.output,
        entry_checkpoint=args.entry_checkpoint, exit_checkpoint=args.exit_checkpoint,
    )
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("tasks", "decisions")}, indent=2))


if __name__ == "__main__":
    main()
