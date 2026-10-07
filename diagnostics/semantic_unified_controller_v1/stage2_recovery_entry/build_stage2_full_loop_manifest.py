"""Freeze validation/calibration full-episode Stage-2 evaluation manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from mode_common import (
    DIRECT_CHECKPOINT, DIRECT_SHA256, ETA_CHECKPOINT, ETA_SHA256, HERE,
    LOCAL_HEAD, LOCAL_THRESHOLD_PROBABILITY, content_hash, sha256,
    verify_content_hash,
)


SOURCE_MANIFEST = HERE / "source_split_manifest.json"
CONTROLLERS = {"SAFETY", "SEMANTIC_LOCAL", "SEMANTIC_MODE"}
ALLOWED_SPLITS = {"validation", "calibration"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def inspect_mode_head(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        shapes = [np.asarray(values[f"layer_{i}_weight"]).shape for i in range(3)]
        mean_shape = np.asarray(values["normalization_mean"]).shape
    if shapes != [(214, 64), (64, 64), (64, 3)] or mean_shape != (214,):
        raise RuntimeError((path, "mode-head architecture mismatch", shapes, mean_shape))
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def build(*, controller: str, split: str, output: Path,
          mode_checkpoint: Path | None = None) -> dict[str, Any]:
    controller = controller.upper()
    if output.exists() or controller not in CONTROLLERS or split not in ALLOWED_SPLITS:
        raise ValueError((output, controller, split))
    if sha256(DIRECT_CHECKPOINT) != DIRECT_SHA256 or sha256(ETA_CHECKPOINT) != ETA_SHA256:
        raise RuntimeError("frozen learned checkpoint mismatch")
    if not LOCAL_HEAD.is_file():
        raise RuntimeError("frozen Stage-1 local head missing")
    source = json.loads(SOURCE_MANIFEST.read_text())
    verify_content_hash(source, SOURCE_MANIFEST)
    if source.get("status") != "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION" or source.get("final_test_generated") is not False:
        raise RuntimeError("source manifest is not the frozen development asset")
    rows = [dict(row) for row in source["sources"][split]]
    mode_head = None
    if controller == "SEMANTIC_MODE":
        if mode_checkpoint is None:
            raise ValueError("SEMANTIC_MODE requires --mode-checkpoint")
        mode_head = inspect_mode_head(mode_checkpoint)
    elif mode_checkpoint is not None:
        raise ValueError("only SEMANTIC_MODE accepts a mode checkpoint")
    local_head = {
        "path": str(LOCAL_HEAD.resolve()), "sha256": sha256(LOCAL_HEAD),
        "threshold_probability": LOCAL_THRESHOLD_PROBABILITY,
        "threshold_logit": float(math.log(LOCAL_THRESHOLD_PROBABILITY / (1-LOCAL_THRESHOLD_PROBABILITY))),
    }
    policy = {
        "controller": controller,
        "direct_g": {"path": str(DIRECT_CHECKPOINT), "sha256": DIRECT_SHA256},
        "structured_eta": {"path": str(ETA_CHECKPOINT), "sha256": ETA_SHA256},
        "local_head": local_head,
        "mode_head": mode_head,
        "normal_actions": ["SAFETY", "LOCAL", "ENTER_RECOVERY"],
        "recovery_semantics": "one eta prediction at entry; immutable eta; dense to terminal; no exit",
        "contains_periodic_timing": False,
    }
    tasks = [{
        "task_index": index, "source_id": row["source_id"], "root_source_id": row["source_id"],
        "split": split, "episode_index": int(row["episode_index"]),
        "rollout_id": int(row["rollout_id"]), "flow_root_seed": int(row["flow_root_seed"]),
        "initial_positions": row["initial_positions"],
    } for index, row in enumerate(rows)]
    implementations = {
        "mode_common": HERE / "mode_common.py",
        "full_loop_common": HERE / "stage2_full_loop_common.py",
        "full_loop_runner": HERE / "run_stage2_full_loop.py",
    }
    result = {
        "schema": "semantic_stage2_full_loop_manifest_v1",
        "status": "FROZEN_READY_FOR_EXECUTION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "controller": controller, "split": split, "policy": policy,
        "policy_hash": content_hash(policy),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_manifest_content_sha256": source["content_sha256"],
        "implementation_hashes": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in implementations.items()
        },
        "task_count": len(tasks), "tasks": tasks,
        "matched_flow_semantics": "same frozen root source and Flow stream for all controllers",
        "final_test_forbidden": True, "contains_periodic_timing": False,
        "budget": {"new_continuations": len(tasks), "maximum_physical_steps": 850*len(tasks)},
    }
    result["content_sha256"] = content_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), required=True)
    parser.add_argument("--split", choices=sorted(ALLOWED_SPLITS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode-checkpoint", type=Path)
    args = parser.parse_args()
    result = build(controller=args.controller, split=args.split, output=args.output,
                   mode_checkpoint=args.mode_checkpoint)
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
