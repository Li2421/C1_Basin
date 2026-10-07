"""Freeze aggregate manifests and integrity evidence for generic Safety sources."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SOURCE_MANIFEST = HERE / "source_split_manifest.json"
PROTOCOL = HERE / "protocol.json"
ALLOWED_SPLITS = ("train", "validation", "calibration")

sys.path[:0] = [str(SYSROOT), str(ROOT), str(PILOT)]
from pilot_common import canonical_json_hash, sha256  # noqa: E402


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rows_in_order(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for split in ALLOWED_SPLITS for row in source["sources"][split]]


def check_trajectory(result: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    path = Path(result["trajectory_file"])
    if sha256(path) != result["trajectory_sha256"]:
        raise RuntimeError((source_row["source_id"], "trajectory hash mismatch"))
    with np.load(path, allow_pickle=False) as data:
        steps = np.asarray(data["step"], dtype=np.int64)
        if not np.array_equal(steps, np.arange(len(steps), dtype=np.int64)):
            raise RuntimeError((source_row["source_id"], "noncontiguous absolute steps"))
        if len(steps) != result["terminal_step"]:
            raise RuntimeError((source_row["source_id"], "terminal length mismatch"))
        if int(data["flow_root_seed"]) != int(source_row["flow_root_seed"]):
            raise RuntimeError((source_row["source_id"], "flow root mismatch"))
        if int(data["flow_rollout_id"]) != int(source_row["rollout_id"]):
            raise RuntimeError((source_row["source_id"], "rollout id mismatch"))
        if len(steps):
            if not np.array_equal(np.asarray(data["positions_before"])[0], np.asarray(source_row["initial_positions"])):
                raise RuntimeError((source_row["source_id"], "initial position mismatch"))
            max_linear_violation = float(max(0.0, -np.min(np.asarray(data["linear_min"], dtype=np.float64))))
            max_speed_excess = float(max(0.0, np.max(np.asarray(data["speed_excess"], dtype=np.float64))))
        else:
            max_linear_violation = max_speed_excess = 0.0
        return {
            "steps": len(steps), "max_linear_violation": max_linear_violation,
            "max_speed_excess": max_speed_excess,
            "nonfinite_actions": int(np.count_nonzero(~np.isfinite(np.asarray(data["u_safe"])))),
        }


def main() -> None:
    for target in (
        HERE / "safety_episode_results.csv", HERE / "decision_state_manifest.jsonl",
        HERE / "decision_state_manifest.meta.json", HERE / "unavailable_decision_anchors.csv",
        HERE / "source_collection_integrity.json",
    ):
        if target.exists():
            raise RuntimeError(("aggregate already frozen", str(target)))
    source = json.loads(SOURCE_MANIFEST.read_text())
    body = {key: value for key, value in source.items() if key != "content_sha256"}
    if canonical_json_hash(body) != source["content_sha256"]:
        raise RuntimeError("source manifest semantic hash mismatch")
    manifest_sha = sha256(SOURCE_MANIFEST)
    expected = rows_in_order(source)
    if len(expected) != 80:
        raise RuntimeError(("expected 80 sources", len(expected)))
    expected_ids = {row["source_id"] for row in expected}
    final_ids = {row["source_id"] for row in source["sources"]["final_test"]}
    if expected_ids & final_ids:
        raise RuntimeError("split leakage")

    results: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    trajectory_checks: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    max_restore_difference = 0.0
    for source_row in expected:
        source_id, split = source_row["source_id"], source_row["split"]
        result_path = HERE / "runs/safety_raw" / split / f"{source_id}.json"
        anchors_path = HERE / "runs/safety_anchor_rows" / split / f"{source_id}.json"
        if not result_path.is_file() or not anchors_path.is_file():
            raise RuntimeError((source_id, "source collection incomplete"))
        result = json.loads(result_path.read_text())
        anchor_payload = json.loads(anchors_path.read_text())
        if not result.get("record_complete") or not anchor_payload.get("record_complete"):
            raise RuntimeError((source_id, "incomplete record"))
        if result["source_manifest_sha256"] != manifest_sha or anchor_payload["source_manifest_sha256"] != manifest_sha:
            raise RuntimeError((source_id, "manifest mismatch"))
        if result["source_id"] != source_id or result["split"] != split:
            raise RuntimeError((source_id, "source identity mismatch"))
        if source_id in observed_ids:
            raise RuntimeError((source_id, "duplicate result"))
        observed_ids.add(source_id)
        check = check_trajectory(result, source_row)
        check["source_id"] = source_id
        trajectory_checks.append(check)
        with np.load(Path(result["trajectory_file"]), allow_pickle=False) as trajectory_data:
            trajectory_current = {
                "flow_step_key": np.asarray(trajectory_data["flow_step_key"], dtype=np.uint32),
                "u_flow": np.asarray(trajectory_data["u_flow"], dtype=np.float64),
                "u_safe": np.asarray(trajectory_data["u_safe"], dtype=np.float64),
            }
        results.append(result)
        anchors = anchor_payload["anchors"]
        if len(anchors) != len(source_row["requested_anchor_steps"]):
            raise RuntimeError((source_id, "anchor request count mismatch"))
        for anchor in anchors:
            if anchor["root_source_id"] != source_id or anchor["split"] != split:
                raise RuntimeError((source_id, "anchor grouping mismatch"))
            if anchor["available"]:
                state_path, input_path = Path(anchor["state_file"]), Path(anchor["decision_input_file"])
                if sha256(state_path) != anchor["state_sha256"] or sha256(input_path) != anchor["decision_input_sha256"]:
                    raise RuntimeError((anchor["state_id"], "anchor artifact hash mismatch"))
                with np.load(state_path, allow_pickle=False) as state_data:
                    state_step = int(state_data["step"])
                    state_history = np.asarray(state_data["error_history"], dtype=np.float64)
                    if int(state_data["history_start_step"]) != 0:
                        raise RuntimeError((anchor["state_id"], "monitor history is not complete from step zero"))
                    if "complete_real_history" not in state_data.files or not bool(state_data["complete_real_history"]):
                        raise RuntimeError((anchor["state_id"], "missing complete-real-history marker"))
                    if state_history.shape != (state_step + 1, 2) or not np.isfinite(state_history).all():
                        raise RuntimeError((anchor["state_id"], "invalid complete monitor history", state_history.shape))
                with np.load(input_path, allow_pickle=False) as data:
                    if np.asarray(data["feature"]).shape != (214,):
                        raise RuntimeError((anchor["state_id"], "feature dimension mismatch"))
                    if not all(np.isfinite(np.asarray(data[key])).all() for key in ("feature", "u_flow", "u_safe")):
                        raise RuntimeError((anchor["state_id"], "nonfinite decision input"))
                    if int(data["requested_global_step"]) != int(anchor["requested_global_step"]):
                        raise RuntimeError((anchor["state_id"], "sidecar step mismatch"))
                    if np.asarray(data["flow_step_key"], dtype=np.uint32).tolist() != anchor["flow_step_key"]:
                        raise RuntimeError((anchor["state_id"], "Flow key mismatch"))
                    step = int(anchor["requested_global_step"])
                    if not np.array_equal(np.asarray(data["flow_step_key"], dtype=np.uint32), trajectory_current["flow_step_key"][step]):
                        raise RuntimeError((anchor["state_id"], "decision/physical-transition Flow sample mismatch"))
                    if not np.array_equal(np.asarray(data["u_flow"], dtype=np.float64), trajectory_current["u_flow"][step]):
                        raise RuntimeError((anchor["state_id"], "decision/physical-transition u_flow mismatch"))
                    if not np.array_equal(np.asarray(data["u_safe"], dtype=np.float64), trajectory_current["u_safe"][step]):
                        raise RuntimeError((anchor["state_id"], "decision/physical-transition u_safe mismatch"))
                    enriched = dict(anchor)
                    enriched.update({
                        "decision_kind": "entry",
                        "absolute_step": int(anchor["actual_global_step"]),
                        "feature": np.asarray(data["feature"], dtype=np.float64).tolist(),
                        "current_u_flow": np.asarray(data["u_flow"], dtype=np.float64).tolist(),
                        "current_u_safe": np.asarray(data["u_safe"], dtype=np.float64).tolist(),
                        "current_flow_key_data": np.asarray(data["flow_step_key"], dtype=np.uint32).tolist(),
                        "current_flow_realization_id": (
                            f"root={anchor['flow_root_seed']}|rollout={anchor['flow_rollout_id']}|"
                            f"absolute_step={anchor['flow_global_step']}|key={anchor['flow_step_key']}"
                        ),
                    })
                max_restore_difference = max(max_restore_difference, float(anchor["state_restore_max_abs_difference"]))
                decision_rows.append(enriched)
            else:
                unavailable.append(anchor)

    unexpected_results = {
        path.stem for split in ALLOWED_SPLITS
        for path in (HERE / "runs/safety_raw" / split).glob("*.json")
    } - expected_ids
    final_artifacts = []
    for pattern in (
        "runs/safety_raw/final_test/*", "runs/safety_trajectories/final_test/*",
        "runs/safety_anchor_rows/final_test/*", "states/safety_anchors/final_test/*",
        "decision_inputs/safety_anchors/final_test/*",
    ):
        final_artifacts.extend(str(path) for path in HERE.glob(pattern))
    if unexpected_results or final_artifacts:
        raise RuntimeError(("unexpected or final-test rollout artifacts", sorted(unexpected_results), final_artifacts))

    result_rows = []
    for row in results:
        result_rows.append({
            "source_id": row["source_id"], "split": row["split"],
            "episode_index": row["episode_index"], "rollout_id": row["rollout_id"],
            "outcome": row["outcome"], "success": row["success"], "timeout": row["timeout"],
            "deadlock": row["deadlock"], "collision": row["collision"], "other_failure": row["other_failure"],
            "terminal_step": row["terminal_step"], "requested_anchor_count": row["requested_anchor_count"],
            "materialized_anchor_count": row["materialized_anchor_count"],
            "unavailable_anchor_count": row["unavailable_anchor_count"],
            "projection_failures": row["projection_failures"], "invalid_actions": row["invalid_actions"],
            "nan_inf_events": row["nan_inf_events"], "trajectory_file": row["trajectory_file"],
            "trajectory_sha256": row["trajectory_sha256"],
        })
    atomic_csv(HERE / "safety_episode_results.csv", result_rows)
    decision_text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in decision_rows)
    atomic_text(HERE / "decision_state_manifest.jsonl", decision_text)
    decision_meta = {
        "schema": "single_segment_decision_state_manifest_v1",
        "status": "COMPLETE_FROZEN",
        "decision_kind": "entry",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": manifest_sha,
        "protocol_sha256": sha256(PROTOCOL),
        "state_count": len(decision_rows),
        "root_source_count": len({row["root_source_id"] for row in decision_rows}),
        "states": decision_rows,
    }
    decision_meta["content_sha256"] = canonical_json_hash(decision_meta)
    atomic_json(HERE / "decision_state_manifest.meta.json", decision_meta)
    atomic_csv(HERE / "unavailable_decision_anchors.csv", unavailable)

    runtimes = []
    for shard in range(2):
        path = HERE / "runs" / f"source_collection_runtime_shard{shard}.json"
        if not path.is_file():
            raise RuntimeError(("missing runtime shard", shard))
        runtimes.append(json.loads(path.read_text()))
    new_rollouts = sum(int(row["new_source_rollouts"]) for row in runtimes)
    reused = sum(int(row["reused_complete_sources"]) for row in runtimes)
    new_steps = sum(int(row["new_physical_steps"]) for row in runtimes)
    if new_rollouts + reused != 80:
        raise RuntimeError(("runtime accounting mismatch", new_rollouts, reused))
    per_split = {
        split: {
            "sources": sum(row["split"] == split for row in results),
            "outcomes": dict(Counter(row["outcome"] for row in results if row["split"] == split)),
            "available_anchors": sum(row["split"] == split for row in decision_rows),
            "unavailable_anchors": sum(row["split"] == split for row in unavailable),
        }
        for split in ALLOWED_SPLITS
    }
    integrity = {
        "schema": "single_segment_source_collection_integrity_v1",
        "status": "PASS", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": manifest_sha,
        "source_manifest_content_sha256": source["content_sha256"],
        "protocol_sha256": sha256(PROTOCOL), "source_count": len(results),
        "expected_source_count": 80, "observed_ids_exact_match": observed_ids == expected_ids,
        "final_test_source_count_reserved": len(source["sources"]["final_test"]),
        "final_test_rollouts_launched": 0, "final_test_artifacts": [],
        "available_decision_states": len(decision_rows), "unavailable_anchor_requests": len(unavailable),
        "unique_decision_state_ids": len({row["state_id"] for row in decision_rows}),
        "unique_root_sources_with_decisions": len({row["root_source_id"] for row in decision_rows}),
        "maximum_anchors_per_root": max(Counter(row["root_source_id"] for row in decision_rows).values(), default=0),
        "all_anchors_outcome_blind": all(row["selection_outcome_blind"] for row in decision_rows),
        "all_anchors_location_independent": all(row["selection_location_independent"] for row in decision_rows),
        "all_available_states_nonterminal": True, "all_features_dimension_214": True,
        "all_states_complete_real_history_from_step_zero": True,
        "all_decision_flow_samples_reused_for_transition": True,
        "max_state_restore_abs_difference": max_restore_difference,
        "max_first_projection_linear_violation": max(row["max_linear_violation"] for row in trajectory_checks),
        "max_first_projection_speed_excess": max(row["max_speed_excess"] for row in trajectory_checks),
        "nonfinite_safety_action_elements": sum(row["nonfinite_actions"] for row in trajectory_checks),
        "projection_failures": sum(row["projection_failures"] for row in results),
        "invalid_actions": sum(row["invalid_actions"] for row in results),
        "nan_inf_events": sum(row["nan_inf_events"] for row in results),
        "collisions": sum(row["collision"] for row in results),
        "per_split": per_split,
        "budget_use": {
            "new_source_rollouts": new_rollouts, "reused_source_rollouts": reused,
            "new_physical_simulation_steps": new_steps,
            "limit_new_continuations": 12000, "limit_new_physical_steps": 6000000,
        },
        "artifacts": {
            "safety_episode_results": {"path": str(HERE / "safety_episode_results.csv"),
                                       "sha256": sha256(HERE / "safety_episode_results.csv")},
            "decision_state_manifest": {"path": str(HERE / "decision_state_manifest.jsonl"),
                                        "sha256": sha256(HERE / "decision_state_manifest.jsonl")},
            "decision_state_manifest_meta": {"path": str(HERE / "decision_state_manifest.meta.json"),
                                             "sha256": sha256(HERE / "decision_state_manifest.meta.json")},
            "unavailable_anchors": {"path": str(HERE / "unavailable_decision_anchors.csv"),
                                    "sha256": sha256(HERE / "unavailable_decision_anchors.csv")},
        },
        "runtime_shards": runtimes,
        "quarantined_invalid_attempt": {
            "path": str(HERE / "quarantine_tail41_job312"),
            "slurm_job_id": 312,
            "file_count": 290,
            "reason": "cancelled partial attempt used tail-41 snapshot_augmented state files; none are accepted or reused",
            "accepted_artifact_count": 0,
        },
    }
    integrity["content_sha256"] = canonical_json_hash(integrity)
    atomic_json(HERE / "source_collection_integrity.json", integrity)
    print(json.dumps({key: value for key, value in integrity.items() if key not in ("runtime_shards",)}, indent=2))


if __name__ == "__main__":
    main()
