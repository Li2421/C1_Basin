"""Strict matched three-controller summary for Stage-2 development evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from mode_common import HERE, content_hash, sha256, verify_content_hash


SOURCE_MANIFEST = HERE / "source_split_manifest.json"


def wilson(k: int, n: int) -> list[float | None]:
    if n == 0:
        return [None, None]
    z = NormalDist().inv_cdf(.975); p = k/n; d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half, center+half]


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.quantile(array, .95))}


def load_rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(path.read_text()); verify_content_hash(manifest, path)
    if (manifest.get("schema") != "semantic_stage2_full_loop_manifest_v1"
            or manifest.get("status") != "FROZEN_READY_FOR_EXECUTION"
            or manifest.get("split") not in {"validation", "calibration"}
            or manifest.get("final_test_forbidden") is not True
            or manifest.get("contains_periodic_timing") is not False):
        raise RuntimeError((path, "invalid or non-development full-loop manifest"))
    if manifest.get("controller") not in {"SAFETY", "SEMANTIC_LOCAL", "SEMANTIC_MODE"}:
        raise RuntimeError((path, "invalid controller"))
    source = json.loads(SOURCE_MANIFEST.read_text())
    verify_content_hash(source, SOURCE_MANIFEST)
    if (manifest.get("source_manifest") != str(SOURCE_MANIFEST)
            or manifest.get("source_manifest_sha256") != sha256(SOURCE_MANIFEST)
            or manifest.get("source_manifest_content_sha256") != source["content_sha256"]):
        raise RuntimeError((path, "unauthenticated frozen source manifest"))
    expected_tasks = [{
        "task_index": index, "source_id": row["source_id"], "root_source_id": row["source_id"],
        "split": manifest["split"], "episode_index": int(row["episode_index"]),
        "rollout_id": int(row["rollout_id"]), "flow_root_seed": int(row["flow_root_seed"]),
        "initial_positions": row["initial_positions"],
    } for index, row in enumerate(source["sources"][manifest["split"]])]
    if manifest.get("tasks") != expected_tasks:
        raise RuntimeError((path, "tasks do not equal frozen source split"))
    directory = HERE / "runs/full_loop" / manifest["split"] / manifest["policy_hash"]
    rows: dict[str, dict[str, Any]] = {}
    for task in manifest["tasks"]:
        result_path = directory / f"task_{int(task['task_index']):04d}.json"
        row = json.loads(result_path.read_text())
        if (not row.get("record_complete") or row.get("manifest_content_sha256") != manifest["content_sha256"]
                or row.get("source_id") != task["source_id"]
                or row.get("root_source_id") != task["root_source_id"]
                or row.get("split") != manifest["split"]
                or row.get("controller") != manifest["controller"]
                or row.get("policy_hash") != manifest["policy_hash"]
                or int(row.get("task_index", -1)) != int(task["task_index"])
                or int(row["flow_sample_count"]) != int(row["physical_transition_count"])):
            raise RuntimeError((result_path, "incomplete or mismatched result"))
        rows[row["source_id"]] = row
    if len(rows) != len(manifest["tasks"]):
        raise RuntimeError("duplicate source IDs")
    expected_files = {f"task_{int(task['task_index']):04d}.json" for task in manifest["tasks"]}
    actual_files = {item.name for item in directory.glob("task_*.json")}
    if actual_files != expected_files:
        raise RuntimeError((directory, "result file coverage mismatch"))
    return manifest, rows


def controller_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows); success = sum(row["success"] for row in rows)
    return {
        "episodes": n, "success": success, "q": success/n, "q_wilson95": wilson(success, n),
        "deadlock": sum(row["deadlock"] for row in rows),
        "timeout": sum(row["timeout"] for row in rows),
        "collision": sum(row["collision"] for row in rows),
        "other_failure": sum(row["other_failure"] for row in rows),
        "jdef": describe([float(row["jdef"]) for row in rows]),
        "local_actions": describe([float(row["local_action_count"]) for row in rows]),
        "recovery_entries": sum(row["entry_step"] is not None for row in rows),
        "entry_step": describe([float(row["entry_step"]) for row in rows if row["entry_step"] is not None]),
        "hard_safety": {
            "agent_collision_events": sum(row["agent_collision_events"] for row in rows),
            "wall_collision_events": sum(row["wall_collision_events"] for row in rows),
            "invalid_actions": sum(row["invalid_actions"] for row in rows),
            "nan_inf_events": sum(row["nan_inf_events"] for row in rows),
            "projection_failures": sum(row["projection_failures"] for row in rows),
        },
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    loaded = [load_rows(path) for path in paths]
    manifests = {manifest["controller"]: manifest for manifest, _ in loaded}
    rows_by_controller = {manifest["controller"]: rows for manifest, rows in loaded}
    if set(manifests) != {"SAFETY", "SEMANTIC_LOCAL", "SEMANTIC_MODE"}:
        raise RuntimeError(("three required controllers absent", sorted(manifests)))
    splits = {manifest["split"] for manifest in manifests.values()}
    source_manifest_hashes = {
        (manifest.get("source_manifest_sha256"), manifest.get("source_manifest_content_sha256"))
        for manifest in manifests.values()
    }
    sources = [set(rows) for rows in rows_by_controller.values()]
    if len(splits) != 1 or len(source_manifest_hashes) != 1 or any(item != sources[0] for item in sources[1:]):
        raise RuntimeError("full-loop comparison is not source matched")
    task_identity: dict[str, tuple[Any, ...]] = {}
    for manifest in manifests.values():
        for task in manifest["tasks"]:
            identity = (
                task["root_source_id"], task["split"], int(task["episode_index"]),
                int(task["rollout_id"]), int(task["flow_root_seed"]),
                tuple(tuple(float(value) for value in agent) for agent in task["initial_positions"]),
            )
            prior = task_identity.setdefault(task["source_id"], identity)
            if prior != identity:
                raise RuntimeError((task["source_id"], "cross-controller source identity mismatch"))
    ordered = sorted(sources[0])
    safety = rows_by_controller["SAFETY"]
    paired = {}
    for controller in ("SEMANTIC_LOCAL", "SEMANTIC_MODE"):
        learned = rows_by_controller[controller]
        rescue = sum(not safety[s]["success"] and learned[s]["success"] for s in ordered)
        broken = sum(safety[s]["success"] and not learned[s]["success"] for s in ordered)
        timeouts = [s for s in ordered if safety[s]["timeout"]]
        deadlocks = [s for s in ordered if safety[s]["deadlock"]]
        paired[controller] = {
            "rescue": rescue, "break": broken, "net_rescue": rescue-broken,
            "timeout_rescue": sum(learned[s]["success"] for s in timeouts),
            "timeout_total": len(timeouts),
            "strict_deadlock_rescue": sum(learned[s]["success"] for s in deadlocks),
            "strict_deadlock_total": len(deadlocks),
        }
    result = {
        "schema": "semantic_stage2_full_loop_comparison_v1", "split": next(iter(splits)),
        "controllers": {
            name: controller_metrics([rows_by_controller[name][s] for s in ordered])
            for name in ("SAFETY", "SEMANTIC_LOCAL", "SEMANTIC_MODE")
        },
        "safety_referenced": paired, "matched_source_count": len(ordered),
        "source_episode_level_inference": True, "final_test_used": False,
        "contains_periodic_timing": False,
    }
    result["content_sha256"] = content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
