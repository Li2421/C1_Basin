"""Freeze paired N/R continuation tasks without executing any rollout.

The initial manifest contains 16 matched streams per generic queried state.
An optional, one-shot confirmation manifest adds streams 16--31 only for
states whose initial paired 90% confidence interval crosses a predeclared
decision cut.  Initial plus confirmation work is fail-closed at 12,000 branch
continuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from branch_common import atomic_json, canonical_hash, file_hash, read_csv, verify_semantic_hash


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/recovery_entry_identifiability_v1"
BUDGET_PATH = HERE / "experiment_budget.json"
PROTOCOL_PATH = HERE / "protocol.md"
FINAL_POLICY_MANIFEST = (
    ROOT / "diagnostics/single_segment_recovery_training_v1/"
    "full_loop_manifests/final_test/policy_iteration_eta.json"
)
FROZEN_HASHES = ROOT / "diagnostics/single_segment_recovery_training_v1/controller_and_projection_hashes.json"
ETA_CHECKPOINT = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
STATE_MACHINE = ROOT / "diagnostics/single_segment_recovery_training_v1/state_machine.py"

ETA_SHA256 = "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095"
EXIT_SHA256 = "7b3bd0d9caff3bfa4baed95d227a6484b5f6f176d7c5b21a33715acb151db694"
EXIT_THRESHOLD_LOGIT = -1.0986122886681098
EXIT_THRESHOLD_PROBABILITY = 0.25
HORIZON = 850
FUTURE_ROOT_SEED = 2026102603
INITIAL_FUTURES = 16
CONFIRMATION_FUTURES = 16
MAX_CONTINUATIONS = 12_000
MAX_ESCALATED_STATES = 135
ESCALATION_CUTS = (-0.05, 0.0, 0.05)


def _load_budget() -> dict[str, Any]:
    value = json.loads(BUDGET_PATH.read_text())
    expected = {
        "initial_matched_futures_per_state": INITIAL_FUTURES,
        "confirmation_block_additional_futures": CONFIRMATION_FUTURES,
        "maximum_new_branch_continuations": MAX_CONTINUATIONS,
        "maximum_escalated_states": MAX_ESCALATED_STATES,
        "initial_branches_per_future": 2,
    }
    for key, target in expected.items():
        if int(value.get(key, -1)) != target:
            raise RuntimeError((BUDGET_PATH, "frozen budget mismatch", key, value.get(key), target))
    if value.get("frozen_before_new_outcomes") is not True:
        raise RuntimeError((BUDGET_PATH, "budget was not frozen before outcomes"))
    return value


def _authoritative_policy() -> dict[str, Any]:
    policy_manifest = json.loads(FINAL_POLICY_MANIFEST.read_text())
    verify_semantic_hash(policy_manifest, FINAL_POLICY_MANIFEST)
    if policy_manifest.get("status") != "FROZEN_READY_FOR_EXECUTION":
        raise RuntimeError((FINAL_POLICY_MANIFEST, "final policy is not frozen"))
    policy = policy_manifest.get("policy", {})
    exit_head = policy.get("exit_head", {})
    required = {
        "sha256": EXIT_SHA256,
        "input_dimension": 217,
        "threshold_logit": EXIT_THRESHOLD_LOGIT,
        "threshold_probability": EXIT_THRESHOLD_PROBABILITY,
    }
    for key, expected in required.items():
        if exit_head.get(key) != expected:
            raise RuntimeError((FINAL_POLICY_MANIFEST, "authoritative exit mismatch", key,
                                exit_head.get(key), expected))
    exit_path = Path(exit_head["path"])
    if file_hash(exit_path) != EXIT_SHA256:
        raise RuntimeError((exit_path, "exit checkpoint hash mismatch"))
    if file_hash(ETA_CHECKPOINT) != ETA_SHA256:
        raise RuntimeError((ETA_CHECKPOINT, "structured-eta checkpoint hash mismatch"))
    implementation = policy_manifest.get("implementation_hashes", {}).get("state_machine", {})
    if (Path(implementation.get("path", "")).resolve() != STATE_MACHINE.resolve()
            or file_hash(STATE_MACHINE) != implementation.get("sha256")):
        raise RuntimeError((STATE_MACHINE, "authoritative state-machine hash mismatch"))
    hashes = json.loads(FROZEN_HASHES.read_text())
    if hashes.get("status") != "PASS":
        raise RuntimeError((FROZEN_HASHES, "controller hash audit is not PASS"))
    for name in ("flowbc", "environment", "projection_constraints", "projection_retry",
                 "event_priority_and_monitor", "startup_feature_builder", "eta_basis",
                 "structured_eta_recovery"):
        record = hashes[name]
        if file_hash(Path(record["path"])) != record["sha256"]:
            raise RuntimeError((name, "frozen controller asset hash mismatch"))
    return {
        "system": "PERSISTENT_STRUCTURED_ETA",
        "eta_checkpoint": {"path": str(ETA_CHECKPOINT), "sha256": ETA_SHA256},
        "exit_head": {
            "path": str(exit_path.resolve()), "sha256": EXIT_SHA256,
            "input_dimension": 217,
            "input_semantics": "concat(current_h_214, immutable_eta_latched_3)",
            "threshold_logit": EXIT_THRESHOLD_LOGIT,
            "threshold_probability": EXIT_THRESHOLD_PROBABILITY,
            "decision": "exit iff logit >= threshold_logit",
            "checkpoint_local_threshold_ignored": True,
        },
        "state_machine": {"path": str(STATE_MACHINE), "sha256": file_hash(STATE_MACHINE)},
        "controller_hash_manifest": {"path": str(FROZEN_HASHES), "sha256": file_hash(FROZEN_HASHES)},
        "authoritative_final_policy_manifest": {
            "path": str(FINAL_POLICY_MANIFEST), "sha256": file_hash(FINAL_POLICY_MANIFEST),
            "content_sha256": policy_manifest["content_sha256"],
        },
        "one_segment_only": True,
        "no_reentry": True,
    }


def _load_states(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    verify_semantic_hash(payload, path)
    if payload.get("schema") != "recovery_entry_queried_state_manifest_v1":
        raise RuntimeError((path, "queried-state schema mismatch"))
    if payload.get("status") != "COMPLETE_FROZEN":
        raise RuntimeError((path, "queried states are not complete and frozen"))
    rows = payload.get("states")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError((path, "queried-state manifest has no states"))
    required = {
        "state_id", "root_source_id", "state_file", "state_sha256", "absolute_step",
        "feature", "current_u_flow", "current_u_safe", "current_flow_key_data",
        "current_flow_realization_id",
    }
    state_ids: set[str] = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError((row.get("state_id"), "missing queried-state fields", sorted(missing)))
        state_id = str(row["state_id"])
        if state_id in state_ids:
            raise RuntimeError((state_id, "duplicate queried state"))
        state_ids.add(state_id)
        state_path = Path(row["state_file"])
        if not state_path.is_file() or file_hash(state_path) != row["state_sha256"]:
            raise RuntimeError((state_id, "state file/hash mismatch"))
        feature = np.asarray(row["feature"], dtype=np.float64)
        if feature.shape != (214,) or not np.isfinite(feature).all():
            raise RuntimeError((state_id, "invalid 214-D feature", feature.shape))
        if np.asarray(row["current_u_flow"]).shape != (2, 2):
            raise RuntimeError((state_id, "invalid current Flow action"))
        if np.asarray(row["current_u_safe"]).shape != (2, 2):
            raise RuntimeError((state_id, "invalid current Safety action"))
        key = np.asarray(row["current_flow_key_data"])
        if key.shape != (2,) or np.any(key < 0) or np.any(key > np.iinfo(np.uint32).max):
            raise RuntimeError((state_id, "invalid current Flow RNG key"))
        step = int(row["absolute_step"])
        if not 0 <= step < HORIZON:
            raise RuntimeError((state_id, "invalid absolute step", step))
        for flag in ("selection_outcome_blind", "location_independent", "terminal_relative_independent"):
            if flag in row and row[flag] is not True:
                raise RuntimeError((state_id, "generic-state guardrail failed", flag))
    if rows != sorted(rows, key=lambda item: str(item["state_id"])):
        raise RuntimeError((path, "states must be frozen in state_id order"))
    return payload, rows


def future_rollout_id(state_id: str, future_index: int) -> int:
    digest = hashlib.sha256(
        f"recovery_entry_identifiability_v1|{FUTURE_ROOT_SEED}|{state_id}|{future_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "little")


def _decision(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": str(row["state_id"]),
        "state_id": str(row["state_id"]),
        "root_source_id": str(row["root_source_id"]),
        "development_cohort": str(row.get("development_cohort", "development")),
        "state_file": str(Path(row["state_file"]).resolve()),
        "state_sha256": row["state_sha256"],
        "absolute_step": int(row["absolute_step"]),
        "feature": row["feature"],
        "current_u_flow": row["current_u_flow"],
        "current_u_safe": row["current_u_safe"],
        "current_flow_key_data": [int(item) for item in row["current_flow_key_data"]],
        "current_flow_realization_id": row["current_flow_realization_id"],
    }


def _tasks(decisions: Sequence[Mapping[str, Any]], future_start: int,
           future_stop: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    used: dict[int, tuple[str, int]] = {}
    for decision in decisions:
        for future_index in range(future_start, future_stop):
            rollout_id = future_rollout_id(str(decision["state_id"]), future_index)
            owner = used.setdefault(rollout_id, (str(decision["state_id"]), future_index))
            if owner != (str(decision["state_id"]), future_index):
                raise RuntimeError(("future rollout-id collision", rollout_id, owner))
            stream_id = f"{decision['state_id']}|future{future_index:02d}|{rollout_id:08x}"
            rng_hash = canonical_hash({
                "future_root_seed": FUTURE_ROOT_SEED,
                "future_rollout_id": rollout_id,
                "first_future_absolute_step": int(decision["absolute_step"]) + 1,
                "current_flow_key_data": decision["current_flow_key_data"],
            })
            for branch in ("N", "R"):
                tasks.append({
                    "task_index": len(tasks),
                    "task_id": f"{stream_id}|{branch}",
                    **dict(decision),
                    "branch": branch,
                    "future_index": future_index,
                    "future_stream_id": stream_id,
                    "future_rollout_id": rollout_id,
                    "future_rng_state_hash": rng_hash,
                })
    return tasks


def _base_manifest(*, wave: str, state_manifest: Path, state_payload: Mapping[str, Any],
                   decisions: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]],
                   initial_task_count: int, selected_by_uncertainty: bool) -> dict[str, Any]:
    maximum_steps = sum(HORIZON - int(task["absolute_step"]) for task in tasks)
    total_reserved = initial_task_count + (len(tasks) if wave == "confirmation" else 0)
    if len(tasks) > MAX_CONTINUATIONS or total_reserved > MAX_CONTINUATIONS:
        raise RuntimeError(("global branch-continuation budget exceeded", total_reserved,
                            MAX_CONTINUATIONS))
    result = {
        "schema": "recovery_entry_paired_branch_manifest_v1",
        "status": "FROZEN_READY_FOR_EXECUTION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "wave": wave,
        "state_source_manifest": str(state_manifest.resolve()),
        "state_source_manifest_sha256": file_hash(state_manifest),
        "state_source_manifest_content_sha256": state_payload["content_sha256"],
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": file_hash(PROTOCOL_PATH)},
        "experiment_budget": {"path": str(BUDGET_PATH), "sha256": file_hash(BUDGET_PATH)},
        "recovery_policy": _authoritative_policy(),
        "action_semantics": {
            "N": "CONTINUE_SAFETY from queried state through terminal",
            "R": "ENTER frozen persistent structured-eta now; frozen learned exit; Safety forever; no re-entry",
        },
        "matched_future_semantics": {
            "future_root_seed": FUTURE_ROOT_SEED,
            "current_flow_key_is_saved_continuation_key": True,
            "current_flow_realization_shared_exactly_between_N_R": True,
            "future_rollout_id_shared_exactly_between_N_R": True,
            "future_step_keys_shared_exactly_between_N_R": True,
            "post_divergence_actions_are_state_conditional": True,
            "flow_rng_consumed_only_by_physical_policy_steps": True,
            "horizon_and_absolute_time_preserved": True,
        },
        "selection": {
            "generic_frozen_states_only": True,
            "future_outcome_not_used_for_initial_selection": wave == "initial",
            "selected_by_predeclared_uncertainty_rule": selected_by_uncertainty,
        },
        "budget": {
            "new_branch_continuations_this_wave": len(tasks),
            "initial_branch_continuations": initial_task_count,
            "global_reserved_after_this_wave": total_reserved,
            "global_maximum_new_branch_continuations": MAX_CONTINUATIONS,
            "remaining_after_this_wave": MAX_CONTINUATIONS - total_reserved,
            "maximum_physical_steps_this_wave": maximum_steps,
        },
        "decision_count": len(decisions),
        "task_count": len(tasks),
        "decisions": list(decisions),
        "tasks": list(tasks),
    }
    result["content_sha256"] = canonical_hash(result)
    return result


def build_initial(state_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError((output, "refusing to overwrite frozen branch manifest"))
    _load_budget()
    state_payload, state_rows = _load_states(state_manifest)
    decisions = [_decision(row) for row in state_rows]
    tasks = _tasks(decisions, 0, INITIAL_FUTURES)
    if len(tasks) > MAX_CONTINUATIONS:
        raise RuntimeError(("initial block alone exceeds budget", len(tasks), MAX_CONTINUATIONS))
    result = _base_manifest(
        wave="initial", state_manifest=state_manifest, state_payload=state_payload,
        decisions=decisions, tasks=tasks, initial_task_count=len(tasks),
        selected_by_uncertainty=False,
    )
    result["budget"]["maximum_full_confirmation_blocks"] = min(
        MAX_ESCALATED_STATES,
        (MAX_CONTINUATIONS - len(tasks)) // (2 * CONFIRMATION_FUTURES),
    )
    result["adaptive_confirmation"] = {
        "not_automatically_launched": True,
        "one_shot_only": True,
        "additional_matched_futures": CONFIRMATION_FUTURES,
        "future_indices": [INITIAL_FUTURES, INITIAL_FUTURES + CONFIRMATION_FUTURES - 1],
        "paired_ci_level": 0.90,
        "eligible_if_initial_delta_ci_crosses_any": list(ESCALATION_CUTS),
        "ranking": "descending initial paired-CI width, then SHA256(state_id)",
        "maximum_escalated_states": result["budget"]["maximum_full_confirmation_blocks"],
    }
    result["content_sha256"] = canonical_hash(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    return result


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def build_confirmation(initial_manifest_path: Path, uncertainty_csv: Path,
                       output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError((output, "refusing to overwrite frozen confirmation manifest"))
    _load_budget()
    initial = json.loads(initial_manifest_path.read_text())
    verify_semantic_hash(initial, initial_manifest_path)
    if initial.get("schema") != "recovery_entry_paired_branch_manifest_v1" or initial.get("wave") != "initial":
        raise RuntimeError((initial_manifest_path, "not an initial branch manifest"))
    if initial.get("status") != "FROZEN_READY_FOR_EXECUTION":
        raise RuntimeError((initial_manifest_path, "initial branch manifest is not executable"))
    finalization_path = uncertainty_csv.parent / "branch_finalization_manifest.json"
    if not finalization_path.is_file():
        raise RuntimeError((finalization_path, "initial branch finalization provenance is required"))
    finalization = json.loads(finalization_path.read_text())
    verify_semantic_hash(finalization, finalization_path)
    if (finalization.get("status") != "COMPLETE"
            or finalization.get("waves") != ["initial"]
            or finalization.get("all_results_complete") is not True
            or finalization.get("all_pairs_matched") is not True):
        raise RuntimeError((finalization_path, "initial branch finalization is not a complete paired audit"))
    sources = finalization.get("source_manifests", [])
    uncertainty_record = finalization.get("outputs", {}).get("label_uncertainty", {})
    if (len(sources) != 1
            or sources[0].get("content_sha256") != initial["content_sha256"]
            or Path(uncertainty_record.get("path", "")).resolve() != uncertainty_csv.resolve()
            or uncertainty_record.get("sha256") != file_hash(uncertainty_csv)):
        raise RuntimeError((finalization_path, "uncertainty/initial-manifest lineage mismatch"))
    evidence = read_csv(uncertainty_csv)
    by_id = {str(row["state_id"]): row for row in evidence}
    if len(by_id) != len(evidence) or set(by_id) != {row["state_id"] for row in initial["decisions"]}:
        raise RuntimeError((uncertainty_csv, "initial uncertainty/state coverage mismatch"))
    eligible = [row for row in evidence if _truth(row["escalation_eligible"])]
    eligible.sort(key=lambda row: (-float(row["delta_q_ci90_width"]),
                                  hashlib.sha256(row["state_id"].encode()).hexdigest()))
    capacity = min(
        MAX_ESCALATED_STATES,
        (MAX_CONTINUATIONS - int(initial["task_count"])) // (2 * CONFIRMATION_FUTURES),
    )
    chosen_ids = {row["state_id"] for row in eligible[:capacity]}
    decisions = [row for row in initial["decisions"] if row["state_id"] in chosen_ids]
    tasks = _tasks(decisions, INITIAL_FUTURES, INITIAL_FUTURES + CONFIRMATION_FUTURES)
    initial_rollout_ids = {int(task["future_rollout_id"]) for task in initial["tasks"]}
    confirmation_rollout_ids = {int(task["future_rollout_id"]) for task in tasks}
    overlap = initial_rollout_ids & confirmation_rollout_ids
    if overlap:
        raise RuntimeError(("future rollout-id collision across initial/confirmation waves",
                            sorted(overlap)[:5]))
    state_manifest = Path(initial["state_source_manifest"])
    state_payload = json.loads(state_manifest.read_text())
    verify_semantic_hash(state_payload, state_manifest)
    result = _base_manifest(
        wave="confirmation", state_manifest=state_manifest, state_payload=state_payload,
        decisions=decisions, tasks=tasks, initial_task_count=int(initial["task_count"]),
        selected_by_uncertainty=True,
    )
    result["parent_initial_manifest"] = {
        "path": str(initial_manifest_path.resolve()),
        "sha256": file_hash(initial_manifest_path),
        "content_sha256": initial["content_sha256"],
    }
    result["initial_uncertainty"] = {
        "path": str(uncertainty_csv.resolve()), "sha256": file_hash(uncertainty_csv),
        "finalization_manifest": str(finalization_path.resolve()),
        "finalization_manifest_sha256": file_hash(finalization_path),
        "eligible_states": len(eligible), "selected_states": len(decisions),
        "eligible_but_deferred_due_budget": max(0, len(eligible) - len(decisions)),
        "selection_rule": "90% paired DeltaQ CI crosses -0.05, 0, or +0.05; width desc; state hash",
    }
    result["content_sha256"] = canonical_hash(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser("initial")
    initial.add_argument("--state-manifest", type=Path, required=True)
    initial.add_argument("--output", type=Path, default=HERE / "branch_seed_manifest.json")
    confirmation = subparsers.add_parser("confirmation")
    confirmation.add_argument("--initial-manifest", type=Path, required=True)
    confirmation.add_argument("--initial-label-uncertainty", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, default=HERE / "branch_seed_manifest_confirmation.json")
    args = parser.parse_args()
    if args.command == "initial":
        result = build_initial(args.state_manifest, args.output)
    else:
        result = build_confirmation(
            args.initial_manifest, args.initial_label_uncertainty, args.output
        )
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"tasks", "decisions"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
