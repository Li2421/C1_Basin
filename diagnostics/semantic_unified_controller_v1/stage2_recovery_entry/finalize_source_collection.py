"""Fail-closed finalizer for the frozen Stage-2 generic Safety anchors."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from mode_common import HERE, content_hash, sha256


SOURCE = HERE / "source_split_manifest.json"
PROTOCOL = HERE / "collection_protocol.json"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    states = []
    source_ids = {
        row["source_id"]
        for split in ("train", "validation", "calibration")
        for row in source["sources"][split]
    }
    for path in sorted((HERE / "runs/safety_raw").glob("*/*.json")):
        row = json.loads(path.read_text())
        if not row.get("record_complete") or row["source_id"] not in source_ids:
            raise RuntimeError((path, "invalid source result"))
        if row["source_manifest_sha256"] != sha256(SOURCE):
            raise RuntimeError((path, "source manifest mismatch"))
        rows.append(row)
        anchors = json.loads(Path(row["anchor_rows_file"]).read_text())
        if sha256(row["anchor_rows_file"]) != row["anchor_rows_sha256"]:
            raise RuntimeError((path, "anchor-row hash mismatch"))
        for anchor in anchors["anchors"]:
            if not anchor["available"]:
                continue
            state_file = Path(anchor["state_file"])
            decision_file = Path(anchor["decision_input_file"])
            if sha256(state_file) != anchor["state_sha256"] or sha256(decision_file) != anchor["decision_input_sha256"]:
                raise RuntimeError((anchor["state_id"], "saved-state hash mismatch"))
            with np.load(decision_file, allow_pickle=False) as values:
                feature = np.asarray(values["feature"], dtype=np.float64)
                u_flow = np.asarray(values["u_flow"], dtype=np.float64)
                u_safe = np.asarray(values["u_safe"], dtype=np.float64)
                flow_key = np.asarray(values["flow_step_key"], dtype=np.uint32)
            if feature.shape != (214,) or u_flow.shape != (2, 2) or u_safe.shape != (2, 2) or flow_key.shape != (2,):
                raise RuntimeError((anchor["state_id"], "decision tensor shape mismatch"))
            state = dict(anchor)
            state.update({
                "decision_kind": "normal",
                "absolute_step": int(anchor["actual_global_step"]),
                "feature": feature.tolist(),
                "current_u_flow": u_flow.tolist(),
                "current_u_safe": u_safe.tolist(),
                "current_flow_key_data": flow_key.astype(int).tolist(),
                "current_flow_realization_id": (
                    f"root={anchor['flow_root_seed']}|rollout={anchor['flow_rollout_id']}|"
                    f"absolute_step={anchor['actual_global_step']}|key={flow_key.astype(int).tolist()}"
                ),
            })
            states.append(state)
    if len(rows) != 160 or {row["source_id"] for row in rows} != source_ids:
        raise RuntimeError(("incomplete source collection", len(rows), len(source_ids)))
    if len(states) != 126 or any(not row["selection_outcome_blind"] for row in states):
        raise RuntimeError(("unexpected anchor set", len(states)))
    if any(row["execution_error"] is not None for row in rows):
        raise RuntimeError("source rollout execution error")

    manifest = {
        "schema": "semantic_stage2_normal_state_manifest_v1",
        "status": "COMPLETE_FROZEN",
        "decision_kind": "normal",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256(SOURCE),
        "protocol_sha256": sha256(PROTOCOL),
        "root_source_count": len(rows),
        "state_count": len(states),
        "split_state_counts": {
            split: sum(row["split"] == split for row in states)
            for split in ("train", "validation", "calibration")
        },
        "selection_rule": "one predeclared deterministic hash-uniform absolute transition per root; unavailable anchors not replaced",
        "states": sorted(states, key=lambda row: row["state_id"]),
    }
    manifest["content_sha256"] = content_hash(manifest)
    atomic_json(HERE / "normal_state_manifest.json", manifest)

    fields = [
        "source_id", "split", "outcome", "success", "timeout", "deadlock", "collision",
        "other_failure", "terminal_step", "materialized_anchor_count", "unavailable_anchor_count",
        "wall_collision_events", "agent_collision_events", "projection_failures", "invalid_actions",
        "nan_inf_events", "runtime_seconds",
    ]
    output = HERE / "safety_source_results.csv"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in sorted(rows, key=lambda row: row["source_id"]))
    os.replace(temporary, output)

    runtimes = [json.loads(path.read_text()) for path in sorted((HERE / "runs").glob("source_collection_runtime_shard*.json"))]
    audit = {
        "schema": "semantic_stage2_source_collection_integrity_v1",
        "status": "PASS",
        "source_rollouts": len(rows),
        "outcomes": {name: sum(row["outcome"] == name for row in rows) for name in ("success", "timeout", "deadlock", "collision", "other")},
        "available_states": len(states),
        "unavailable_predeclared_anchors": sum(row["unavailable_anchor_count"] for row in rows),
        "source_execution_errors": 0,
        "hard_safety_or_numerical_events": sum(
            row["wall_collision_events"] + row["agent_collision_events"] + row["projection_failures"]
            + row["invalid_actions"] + row["nan_inf_events"] for row in rows
        ),
        "new_physical_steps": sum(row["terminal_step"] for row in rows),
        "manifest_sha256": sha256(HERE / "normal_state_manifest.json"),
        "manifest_content_sha256": manifest["content_sha256"],
        "runtime_shards": len(runtimes),
        "final_test_used": False,
    }
    atomic_json(HERE / "source_collection_integrity.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
