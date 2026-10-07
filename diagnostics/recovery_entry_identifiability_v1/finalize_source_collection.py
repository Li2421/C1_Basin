"""Fail-closed finalizer for development Safety roots and queried states."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/recovery_entry_identifiability_v1"
SOURCE = HERE / "development_source_manifest.json"
PROTOCOL = HERE / "collection_protocol.json"
OUTPUT_JSON = HERE / "queried_state_manifest.json"
OUTPUT_CSV = HERE / "queried_state_manifest.csv"
INTEGRITY = HERE / "source_collection_integrity.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)
    os.replace(temporary, path)


def load_materialized_state(anchor: dict[str, Any]) -> dict[str, Any]:
    state_file = Path(anchor["state_file"])
    decision_file = Path(anchor["decision_input_file"])
    if sha256(state_file) != anchor["state_sha256"]:
        raise RuntimeError((anchor["state_id"], "augmented-state hash mismatch"))
    if sha256(decision_file) != anchor["decision_input_sha256"]:
        raise RuntimeError((anchor["state_id"], "decision-input hash mismatch"))
    with np.load(state_file, allow_pickle=False) as payload:
        required = {
            "positions", "velocities", "step", "error_history", "history_start_step",
            "candidate_since", "stuck_timer", "max_stuck_timer", "ever_candidate_deadlock",
            "first_success_step", "first_deadlock_step", "first_wall_collision_step",
            "first_agent_collision_step", "done", "complete_real_history",
        }
        if not required.issubset(payload.files):
            raise RuntimeError((anchor["state_id"], "incomplete augmented-state fields"))
        global_step = int(np.asarray(payload["step"]).item())
        history = np.asarray(payload["error_history"], dtype=np.float64)
        if history.shape != (global_step + 1, 2) or not np.isfinite(history).all():
            raise RuntimeError((anchor["state_id"], "incomplete real monitor history", history.shape))
        if bool(np.asarray(payload["done"]).item()):
            raise RuntimeError((anchor["state_id"], "terminal state was materialized"))
        if not bool(np.asarray(payload["complete_real_history"]).item()):
            raise RuntimeError((anchor["state_id"], "synthetic or truncated history"))
    with np.load(decision_file, allow_pickle=False) as payload:
        feature = np.asarray(payload["feature"], dtype=np.float64)
        u_flow = np.asarray(payload["u_flow"], dtype=np.float64)
        u_safe = np.asarray(payload["u_safe"], dtype=np.float64)
        flow_key = np.asarray(payload["flow_step_key"], dtype=np.uint32)
        observation = np.asarray(payload["observation"], dtype=np.float64)
        requested_step = int(np.asarray(payload["requested_global_step"]).item())
    if feature.shape != (214,) or not np.isfinite(feature).all():
        raise RuntimeError((anchor["state_id"], "invalid 214-D deployment feature", feature.shape))
    if u_flow.shape != (2, 2) or u_safe.shape != (2, 2) or flow_key.shape != (2,):
        raise RuntimeError((anchor["state_id"], "invalid current decision tensors"))
    if global_step != int(anchor["actual_global_step"]) or global_step != requested_step:
        raise RuntimeError((anchor["state_id"], "global-step mismatch"))
    state = dict(anchor)
    state.update({
        "schema": "recovery_entry_queried_state_v1",
        "development_cohort": "development",
        "decision_kind": "Safety_vs_enter_persistent_eta_now",
        "absolute_step": global_step,
        "global_step": global_step,
        "feature": feature.tolist(),
        "h_t": feature.tolist(),
        "current_u_flow": u_flow.tolist(),
        "current_u_safe": u_safe.tolist(),
        "current_observation": observation.tolist(),
        "current_flow_key_data": flow_key.astype(int).tolist(),
        "current_flow_realization_id": (
            f"root={anchor['flow_root_seed']}|rollout={anchor['flow_rollout_id']}|"
            f"absolute_step={global_step}|key={flow_key.astype(int).tolist()}"
        ),
        "feature_dimension": 214,
        "selection_outcome_blind": True,
        "selection_location_independent": True,
        "selection_terminal_relative_independent": True,
        "selection_failure_type_independent": True,
        "selection_recovery_outcome_independent": True,
    })
    return state


def finalize() -> dict[str, Any]:
    if OUTPUT_JSON.exists() or OUTPUT_CSV.exists() or INTEGRITY.exists():
        raise RuntimeError("refusing to overwrite frozen finalized source artifacts")
    source = json.loads(SOURCE.read_text())
    source_body = {key: value for key, value in source.items() if key != "content_sha256"}
    if content_hash(source_body) != source["content_sha256"]:
        raise RuntimeError("development source manifest semantic hash mismatch")
    source_rows = source["sources"]["development"]
    expected_ids = {row["source_id"] for row in source_rows}
    if len(source_rows) != 120 or len(expected_ids) != 120:
        raise RuntimeError("development source manifest does not contain 120 independent roots")

    results: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    observed_anchor_ids: set[str] = set()
    for path in sorted((HERE / "runs/safety_raw/development").glob("*.json")):
        row = json.loads(path.read_text())
        if not row.get("record_complete") or row.get("source_id") not in expected_ids:
            raise RuntimeError((path, "invalid development source result"))
        if row["source_manifest_sha256"] != sha256(SOURCE):
            raise RuntimeError((path, "source-manifest hash mismatch"))
        if row.get("execution_error") is not None:
            raise RuntimeError((path, "Safety source execution error"))
        anchors_path = Path(row["anchor_rows_file"])
        if sha256(anchors_path) != row["anchor_rows_sha256"]:
            raise RuntimeError((path, "anchor-row hash mismatch"))
        anchors = json.loads(anchors_path.read_text())["anchors"]
        if len(anchors) != 2 or {int(item["anchor_slot"]) for item in anchors} != {0, 1}:
            raise RuntimeError((path, "expected exactly two predeclared anchor records"))
        for anchor in anchors:
            state_id = anchor["state_id"]
            if state_id in observed_anchor_ids:
                raise RuntimeError((state_id, "duplicate queried-state ID"))
            observed_anchor_ids.add(state_id)
            if anchor["available"]:
                states.append(load_materialized_state(anchor))
            else:
                if anchor.get("no_replacement") is not True:
                    raise RuntimeError((state_id, "unavailable anchor was replaceable"))
                unavailable.append(anchor)
        results.append(row)

    if len(results) != 120 or {row["source_id"] for row in results} != expected_ids:
        raise RuntimeError(("incomplete development source collection", len(results)))
    if len(observed_anchor_ids) != 240 or len(states) + len(unavailable) != 240:
        raise RuntimeError(("incomplete predeclared anchor accounting", len(states), len(unavailable)))
    if len({row["root_source_id"] for row in states}) > 120:
        raise RuntimeError("queried state has an unknown root")

    manifest = {
        "schema": "recovery_entry_queried_state_manifest_v1",
        "status": "COMPLETE_FROZEN",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "final_test_used": False,
        "source_manifest_path": str(SOURCE),
        "source_manifest_sha256": sha256(SOURCE),
        "protocol_sha256": sha256(PROTOCOL),
        "root_source_count": len(results),
        "requested_state_count": 240,
        "state_count": len(states),
        "unavailable_request_count": len(unavailable),
        "source_grouping_key": "root_source_id",
        "selection_rule": (
            "two predeclared source-hash-seeded uniform absolute transitions per root; "
            "materialize only nonterminal states; unavailable requests not replaced"
        ),
        "feature_dimension": 214,
        "states": sorted(states, key=lambda row: row["state_id"]),
    }
    manifest["content_sha256"] = content_hash(manifest)
    atomic_json(OUTPUT_JSON, manifest)

    csv_rows = sorted(states, key=lambda row: row["state_id"])
    atomic_csv(OUTPUT_CSV, csv_rows, [
        "state_id", "root_source_id", "development_cohort", "episode_index", "rollout_id",
        "anchor_slot", "requested_global_step", "actual_global_step", "absolute_step",
        "state_file", "state_sha256", "decision_input_file", "decision_input_sha256",
        "feature_dimension", "flow_root_seed", "flow_rollout_id", "flow_global_step",
        "current_flow_realization_id", "complete_real_monitor_history",
        "monitor_history_start_step", "monitor_history_length", "selection_outcome_blind",
        "selection_location_independent", "selection_terminal_relative_independent",
    ])

    integrity = {
        "schema": "recovery_entry_source_collection_integrity_v1",
        "status": "PASS",
        "source_rollouts": len(results),
        "independent_root_sources": len(expected_ids),
        "requested_generic_states": 240,
        "materialized_nonterminal_states": len(states),
        "unavailable_predeclared_requests": len(unavailable),
        "outcomes": {
            name: sum(row["outcome"] == name for row in results)
            for name in ("success", "timeout", "deadlock", "collision", "other")
        },
        "source_execution_errors": 0,
        "hard_safety_or_numerical_events": sum(
            row["wall_collision_events"] + row["agent_collision_events"]
            + row["projection_failures"] + row["invalid_actions"] + row["nan_inf_events"]
            for row in results
        ),
        "new_source_physical_steps": sum(row["terminal_step"] for row in results),
        "queried_state_manifest_sha256": sha256(OUTPUT_JSON),
        "queried_state_manifest_content_sha256": manifest["content_sha256"],
        "final_test_used": False,
        "branch_rollouts_executed_by_finalizer": 0,
    }
    atomic_json(INTEGRITY, integrity)
    return integrity


def main() -> None:
    print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
