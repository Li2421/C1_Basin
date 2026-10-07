"""Freeze development full-episode evaluation for semantic S/L or H8 reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from local_common import (
    BASE, DIRECT_CHECKPOINT, DIRECT_SHA256, HERE, HORIZON,
    MAX_CONTINUATIONS, MAX_PHYSICAL_STEPS, content_hash, sha256, verify_content_hash,
)


SOURCE_MANIFEST = BASE / "source_split_manifest.json"
EXTERNAL_DEVELOPMENT_MANIFEST = Path(
    "/home/zhihan/research/Basin_C1/diagnostics/gphi_h8_fresh_unseen_generalization_v1/fresh_test_manifest.json"
)
MANIFEST_DIR = HERE / "full_loop_manifests"
THRESHOLDS = (0.25, 0.5, 0.75)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def inspect_head(path: Path, threshold_probability: float) -> dict[str, Any]:
    if threshold_probability not in THRESHOLDS:
        raise ValueError((threshold_probability, THRESHOLDS))
    with np.load(path, allow_pickle=False) as values:
        if np.asarray(values["normalization_mean"]).shape != (214,):
            raise RuntimeError("local head is not 214-D")
        if np.asarray(values["layer_0_weight"]).shape != (214, 64):
            raise RuntimeError("local head architecture mismatch")
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "threshold_probability": threshold_probability,
        "threshold_logit": float(math.log(threshold_probability/(1-threshold_probability))),
    }


def _budget(exclude: Path) -> tuple[int, int]:
    continuations = steps = 0
    branch_dir = HERE / "branch_manifests"
    for path in branch_dir.glob("*.json") if branch_dir.exists() else ():
        value = json.loads(path.read_text()); verify_content_hash(value, path)
        continuations += int(value["budget"]["new_continuations"])
        steps += int(value["budget"]["maximum_physical_steps"])
    for path in MANIFEST_DIR.glob("*.json") if MANIFEST_DIR.exists() else ():
        if path.resolve() == exclude.resolve():
            continue
        value = json.loads(path.read_text()); verify_content_hash(value, path)
        continuations += int(value["budget"]["new_continuations"])
        steps += int(value["budget"]["maximum_physical_steps"])
    return continuations, steps


def build(*, controller: str, split: str, output: Path,
          head_checkpoint: Path | None = None, threshold: float | None = None) -> dict[str, Any]:
    if output.exists() or split not in {"train", "validation", "calibration", "development_external"}:
        raise ValueError((output, split, "output exists or forbidden split"))
    controller = controller.upper()
    if controller not in {"SEMANTIC_LOCAL", "H8_REFERENCE"}:
        raise ValueError(controller)
    if sha256(DIRECT_CHECKPOINT) != DIRECT_SHA256:
        raise RuntimeError("Direct-g hash mismatch")
    source_path = SOURCE_MANIFEST
    source = json.loads(source_path.read_text()); verify_content_hash(source, source_path)
    if split == "development_external":
        source_path = EXTERNAL_DEVELOPMENT_MANIFEST
        source = json.loads(source_path.read_text()); verify_content_hash(source, source_path)
        if source.get("episode_count") != 200 or source.get("frozen_before_rollout") is not True:
            raise RuntimeError("external development cohort is not the frozen 200-episode asset")
        rows = [{**dict(row), "split": split} for row in source["episodes"]]
    else:
        rows = [dict(row) for row in source["sources"][split]]
    local_head = None
    if controller == "SEMANTIC_LOCAL":
        if head_checkpoint is None or threshold is None:
            raise ValueError("semantic local needs head + predeclared threshold")
        local_head = inspect_head(head_checkpoint, float(threshold))
    elif head_checkpoint is not None or threshold is not None:
        raise ValueError("H8 reference accepts no learned local head")
    policy = {
        "controller": controller, "direct_g": {"path": str(DIRECT_CHECKPOINT), "sha256": DIRECT_SHA256},
        "local_head": local_head,
        "contains_periodic_timing": controller == "H8_REFERENCE",
        "h8_is_external_reference_only": controller == "H8_REFERENCE",
    }
    prior_continuations, prior_steps = _budget(output)
    new_continuations, maximum_steps = len(rows), len(rows) * HORIZON
    if prior_continuations + new_continuations > MAX_CONTINUATIONS or prior_steps + maximum_steps > MAX_PHYSICAL_STEPS:
        raise RuntimeError("Stage-1 global budget exceeded")
    tasks = [{
        "task_index": index, "source_id": row["source_id"], "root_source_id": row["source_id"],
        "split": split, "episode_index": int(row["episode_index"]),
        "rollout_id": int(row["rollout_id"]), "flow_root_seed": int(row["flow_root_seed"]),
        "initial_positions": row["initial_positions"],
    } for index, row in enumerate(rows)]
    result = {
        "schema": "semantic_local_full_loop_manifest_v1", "status": "FROZEN_READY_FOR_EXECUTION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "controller": controller,
        "split": split, "policy": policy, "policy_hash": content_hash(policy),
        "source_manifest": str(source_path), "source_manifest_sha256": sha256(source_path),
        "source_manifest_content_sha256": source["content_sha256"],
        "implementation_hashes": {
            "local_common": {"path": str(HERE / "local_common.py"), "sha256": sha256(HERE / "local_common.py")},
            "full_loop_runner": {"path": str(HERE / "run_local_full_loop.py"), "sha256": sha256(HERE / "run_local_full_loop.py")},
        },
        "task_count": len(tasks), "tasks": tasks,
        "matched_flow_semantics": "same source Flow root/rollout ID as frozen Safety baseline",
        "final_test_forbidden": True,
        "budget": {
            "new_continuations": new_continuations, "maximum_physical_steps": maximum_steps,
            "global_after_continuations": prior_continuations + new_continuations,
            "global_after_maximum_steps": prior_steps + maximum_steps,
            "global_limits": {"continuations": MAX_CONTINUATIONS, "physical_steps": MAX_PHYSICAL_STEPS},
        },
    }
    result["content_sha256"] = content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=("SEMANTIC_LOCAL", "H8_REFERENCE"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "calibration", "development_external"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    result = build(controller=args.controller, split=args.split, output=args.output,
                   head_checkpoint=args.head_checkpoint, threshold=args.threshold)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
