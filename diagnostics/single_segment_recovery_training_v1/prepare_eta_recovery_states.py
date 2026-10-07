"""Predeclare outcome-blind ETA recovery-state collection tasks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path("/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1")
ENTRY_MANIFEST = HERE / "decision_state_manifest.meta.json"
OUTPUT = HERE / "eta_recovery_state_collection_plan.json"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(payload: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload.get("content_sha256") != canonical_hash(body):
        raise RuntimeError("entry decision manifest semantic hash mismatch")
    if payload.get("status") != "COMPLETE_FROZEN" or payload.get("decision_kind") != "entry":
        raise RuntimeError("entry decision manifest is not complete/frozen")


def rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"single_segment_eta_recovery_collection_v1|{row['split']}|{row['root_source_id']}|{row['state_id']}".encode()
    ).hexdigest()


def recovery_depth(state_id: str) -> int:
    digest = hashlib.sha256(f"single_segment_eta_recovery_depth_v1|{state_id}".encode()).digest()
    return 1 + int.from_bytes(digest[:4], "little") % 32


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError((OUTPUT, "refusing to overwrite frozen plan"))
    source = json.loads(ENTRY_MANIFEST.read_text())
    verify(source)
    selected: list[dict[str, Any]] = []
    for split, quota in (("train", 16), ("validation", 8)):
        candidates = sorted((row for row in source["states"] if row["split"] == split), key=rank)
        first_by_root: list[dict[str, Any]] = []
        seen_roots: set[str] = set()
        for row in candidates:
            if row["root_source_id"] not in seen_roots:
                seen_roots.add(row["root_source_id"])
                first_by_root.append(row)
        if len(first_by_root) < quota:
            raise RuntimeError((split, "insufficient source-diverse states", len(first_by_root), quota))
        selected.extend(first_by_root[:quota])
    tasks = []
    for index, state in enumerate(selected):
        depth = recovery_depth(state["state_id"])
        tasks.append({
            "task_index": index,
            "task_id": f"eta_recovery_state|{state['state_id']}|depth{depth:02d}",
            "root_source_id": state["root_source_id"], "split": state["split"],
            "entry_state_id": state["state_id"], "entry_state_file": state["state_file"],
            "entry_state_sha256": state["state_sha256"], "entry_absolute_step": int(state["absolute_step"]),
            "flow_root_seed": int(state["flow_root_seed"]), "flow_rollout_id": int(state["flow_rollout_id"]),
            "entry_feature": state["feature"], "entry_u_flow": state["current_u_flow"],
            "entry_u_safe": state["current_u_safe"], "entry_flow_key": state["current_flow_key_data"],
            "entry_flow_realization_id": state["current_flow_realization_id"],
            "recovery_depth_transitions": depth,
        })
    payload = {
        "schema": "single_segment_eta_recovery_state_collection_plan_v1",
        "status": "FROZEN_BEFORE_RECOVERY_OUTCOMES",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "entry_state_manifest": str(ENTRY_MANIFEST), "entry_state_manifest_sha256": sha256(ENTRY_MANIFEST),
        "entry_state_manifest_content_sha256": source["content_sha256"],
        "selection": {
            "outcome_blind": True, "counterfactual_outcome_blind": True,
            "location_independent": True, "source_diverse": True,
            "method": "deterministic SHA256 rank; at most one anchor per root source",
            "split_quotas": {"train": 16, "validation": 8},
            "calibration_selected": 0, "final_test_selected": 0,
        },
        "recovery_depth_rule": {
            "range_inclusive": [1, 32],
            "method": "1 + uint32_le(SHA256('single_segment_eta_recovery_depth_v1|'+state_id)[:4]) mod 32",
            "outcome_blind": True,
            "terminal_before_query": "unavailable without replacement",
        },
        "system": "ETA", "eta_prediction": "once at entry then latched",
        "dense_recovery": True, "task_count": len(tasks), "tasks": tasks,
    }
    payload["content_sha256"] = canonical_hash(payload)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({key: value for key, value in payload.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
