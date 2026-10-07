"""Freeze aggregate ETA recovery-state manifest for Stage-B exit labels."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
PLAN = HERE / "eta_recovery_state_collection_plan.json"
OUTPUT = HERE / "exit_state_manifest.json"
UNAVAILABLE = HERE / "unavailable_exit_state_requests.csv"
INTEGRITY = HERE / "eta_recovery_state_collection_integrity.json"
REFERENCE = ROOT / "diagnostics/recovery_takeover_primitive_v1/development_manifest.json"

sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT)]
from pilot_common import canonical_json_hash, sha256  # noqa: E402
from diagnostics.single_segment_recovery_training_v1.full_state_io import restore_full_history  # noqa: E402
from single_integrator.environment import Config  # noqa: E402


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def verify(payload: dict[str, Any], label: str) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_json_hash(body):
        raise RuntimeError((label, "semantic hash mismatch"))


def main() -> None:
    if any(path.exists() for path in (OUTPUT, UNAVAILABLE, INTEGRITY)):
        raise RuntimeError("refusing to overwrite frozen ETA recovery-state aggregate")
    plan = json.loads(PLAN.read_text())
    verify(plan, "plan")
    if plan.get("status") != "FROZEN_BEFORE_RECOVERY_OUTCOMES" or len(plan["tasks"]) != 24:
        raise RuntimeError("invalid collection plan")
    config = Config(**json.loads(REFERENCE.read_text())["environment"])
    task_by_index = {int(task["task_index"]): task for task in plan["tasks"]}
    rows: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for task_index in range(24):
        path = HERE / "runs/eta_recovery_state_rows" / f"task_{task_index:03d}.json"
        if not path.is_file():
            raise RuntimeError((task_index, "missing collection row"))
        row = json.loads(path.read_text())
        task = task_by_index[task_index]
        if not row.get("record_complete") or row["task_id"] != task["task_id"]:
            raise RuntimeError((task_index, "incomplete/task mismatch"))
        if row["collection_plan_sha256"] != sha256(PLAN) or row["eta_query_count"] != 1:
            raise RuntimeError((task_index, "plan or eta-latch mismatch"))
        if row["split"] not in ("train", "validation"):
            raise RuntimeError((task_index, "forbidden split", row["split"]))
        if int(row["requested_recovery_transitions"]) != int(task["recovery_depth_transitions"]):
            raise RuntimeError((task_index, "depth mismatch"))
        if not row["available"]:
            if not row.get("no_replacement") or row["executed_recovery_transitions"] > row["requested_recovery_transitions"]:
                raise RuntimeError((task_index, "invalid unavailable semantics"))
            unavailable.append(row)
            continue
        required = {
            "state_id", "root_source_id", "split", "state_file", "state_sha256", "absolute_step",
            "feature", "current_u_flow", "current_u_safe", "current_flow_realization_id",
            "current_flow_key_data", "eta_latched", "entry_step", "recovery_transitions",
        }
        if required - set(row):
            raise RuntimeError((task_index, "missing output fields", sorted(required - set(row))))
        if row["executed_recovery_transitions"] != row["requested_recovery_transitions"]:
            raise RuntimeError((task_index, "available state did not execute planned depth"))
        if int(row["absolute_step"]) != int(row["entry_step"]) + int(row["recovery_transitions"]):
            raise RuntimeError((task_index, "absolute-time mismatch"))
        if int(row["recovery_transitions"]) < 1 or int(row["entry_step"]) >= int(row["absolute_step"]):
            raise RuntimeError((task_index, "invalid recovery memory"))
        state_path, input_path = Path(row["state_file"]), Path(row["decision_input_file"])
        if sha256(state_path) != row["state_sha256"] or sha256(input_path) != row["decision_input_sha256"]:
            raise RuntimeError((task_index, "derived artifact hash mismatch"))
        env = restore_full_history(state_path, config)
        if env.step_count != int(row["absolute_step"]) or env.done:
            raise RuntimeError((task_index, "invalid recovered augmented state"))
        with np.load(input_path, allow_pickle=False) as data:
            comparisons = {
                "feature": row["feature"], "u_flow": row["current_u_flow"],
                "u_safe": row["current_u_safe"], "flow_step_key": row["current_flow_key_data"],
                "eta_latched": row["eta_latched"],
            }
            for key, expected in comparisons.items():
                if not np.array_equal(np.asarray(data[key]), np.asarray(expected, dtype=np.asarray(data[key]).dtype)):
                    raise RuntimeError((task_index, key, "sidecar mismatch"))
            if int(data["entry_step"]) != int(row["entry_step"]) or int(data["recovery_transitions"]) != int(row["recovery_transitions"]):
                raise RuntimeError((task_index, "sidecar recovery memory mismatch"))
        rows.append(row)
    if len({row["root_source_id"] for row in rows + unavailable}) != 24:
        raise RuntimeError("source diversity was not preserved")

    payload = {
        "schema": "single_segment_eta_exit_state_manifest_v1",
        "status": "COMPLETE_FROZEN", "decision_kind": "exit",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "collection_plan": str(PLAN), "collection_plan_sha256": sha256(PLAN),
        "collection_plan_content_sha256": plan["content_sha256"],
        "selection": plan["selection"], "recovery_depth_rule": plan["recovery_depth_rule"],
        "requested_state_count": 24, "state_count": len(rows), "unavailable_count": len(unavailable),
        "states": rows,
    }
    payload["content_sha256"] = canonical_json_hash(payload)
    atomic_json(OUTPUT, payload)
    atomic_csv(UNAVAILABLE, unavailable)

    runtimes = []
    for shard in range(2):
        path = HERE / "runs" / f"eta_recovery_state_runtime_shard{shard}.json"
        if not path.is_file():
            raise RuntimeError(("missing runtime shard", shard))
        runtimes.append(json.loads(path.read_text()))
    continuation_count = sum(int(row["new_continuations"]) for row in runtimes)
    step_count = sum(int(row["new_physical_steps"]) for row in runtimes)
    if continuation_count != 24 or step_count != sum(int(row["executed_recovery_transitions"]) for row in rows + unavailable):
        raise RuntimeError("runtime/budget accounting mismatch")
    integrity = {
        "schema": "single_segment_eta_recovery_state_collection_integrity_v1",
        "status": "PASS", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha256(PLAN), "plan_content_sha256": plan["content_sha256"],
        "all_24_tasks_accounted": len(rows) + len(unavailable) == 24,
        "available_states": len(rows), "unavailable_terminal_before_depth": len(unavailable),
        "split_counts_requested": {split: sum(task["split"] == split for task in plan["tasks"])
                                   for split in ("train", "validation")},
        "split_counts_available": {split: sum(row["split"] == split for row in rows)
                                   for split in ("train", "validation")},
        "calibration_or_final_sources": 0,
        "unique_root_sources_requested": len({task["root_source_id"] for task in plan["tasks"]}),
        "eta_queries": sum(int(row["eta_query_count"]) for row in rows + unavailable),
        "all_eta_latched_once": all(row["eta_query_count"] == 1 and row["eta_unchanged_while_active"]
                                    for row in rows + unavailable),
        "all_entry_flow_reused": all(row.get("entry_current_flow_reused") for row in rows),
        "all_round_trips_exact": all(row["round_trip_audit"]["passed"] and
                                     row["round_trip_audit"]["history_start_step"] == 0 for row in rows),
        "all_current_inputs_frozen": True,
        "maximum_projection_linear_violation": max((row["maximum_linear_violation"] for row in rows + unavailable), default=0.0),
        "maximum_projection_speed_excess": max((row["maximum_speed_excess"] for row in rows + unavailable), default=0.0),
        "budget_use": {"new_continuations": continuation_count, "new_physical_simulation_steps": step_count,
                       "global_limits": {"continuations": 12000, "physical_steps": 6000000}},
        "exit_state_manifest": {"path": str(OUTPUT), "sha256": sha256(OUTPUT),
                                "content_sha256": payload["content_sha256"]},
        "unavailable_csv": {"path": str(UNAVAILABLE), "sha256": sha256(UNAVAILABLE)},
        "runtime_shards": runtimes,
    }
    integrity["content_sha256"] = canonical_json_hash(integrity)
    atomic_json(INTEGRITY, integrity)
    print(json.dumps({key: value for key, value in integrity.items() if key != "runtime_shards"}, indent=2))


if __name__ == "__main__":
    main()
