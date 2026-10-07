#!/usr/bin/env python3
"""Development-only audit of preregistered late S-XL checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

from diagnostics.double_bottleneck_initial_state_coverage.tools.evaluate_scaling import (
    _teacher_batched,
)
from diagnostics.double_bottleneck_recovery_v2.tools.evaluate_v2 import _aggregate_rollouts
from double_bottleneck.environment import Config
from double_bottleneck.evaluate_flowbc_4a import _rollout
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    study = root / "diagnostics/double_bottleneck_sxl_baseline_maturation"
    output = study / "checkpoint_stability.json"
    if output.exists():
        raise FileExistsError(output)
    development = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "val", seed=71
    )
    summary = json.loads((study / "model/training_summary.json").read_text())
    checkpoints = {
        "primary_75pct": study / "model/ckpt_primary_75pct.pkl",
        "primary_87_5pct": study / "model/ckpt_primary_87_5pct.pkl",
        "primary_final": study / "model/ckpt_primary_final.pkl",
    }
    result = {
        "schema": "double_bottleneck_sxl_checkpoint_stability_v1",
        "scientific_set": "development_only",
        "untouched_test_loaded_or_used": False,
        "checkpoint_selection_changed_by_this_audit": False,
        "rollout_seeds": [0, 1, 2, 3],
        "checkpoints": {},
    }
    started = time.perf_counter()
    for label, checkpoint in checkpoints.items():
        policy, metadata = load_checkpoint(
            checkpoint, development.environment_fingerprint
        )
        rows = []
        rollout_id = 0
        for family in development.family_names:
            episode = development.by_family[family][0]
            for seed in result["rollout_seeds"]:
                row, _ = _rollout(policy, development, episode, seed, rollout_id)
                rows.append(row)
                rollout_id += 1
        result["checkpoints"][label] = {
            "step": int(metadata["checkpoint_step"]),
            "aggregate": _aggregate_rollouts(rows),
            "teacher_forced": _teacher_batched(
                policy, development, Config(**development.config)
            ),
        }
        print(json.dumps({"stage": "checkpoint_complete", "label": label}), flush=True)
    result["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
