#!/usr/bin/env python3
"""Freeze S-XL development and two untouched-test evaluations."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from diagnostics.double_bottleneck_initial_state_coverage.tools.evaluate_scaling import (
    _evaluate_set,
    _support_tree,
)
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_observations(study: Path):
    nominal = FlowBC4ADataset(study / "data/train_pool", "train", seed=0)
    with np.load(study / "data/recovery_train.npz", allow_pickle=False) as archive:
        recovery = archive["observations"].astype(np.float32)
    result = np.concatenate((nominal.observations, recovery), axis=0)
    if len(result) != 1840528:
        raise ValueError("S-XL support library count mismatch")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    study = root / "diagnostics/double_bottleneck_sxl_baseline_maturation"
    output = study / "evaluation"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    development = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "val", seed=44
    )
    existing = FlowBC4ADataset(
        root
        / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
        "val",
        seed=45,
    )
    fresh = FlowBC4ADataset(study / "data/fresh_test_pool", "val", seed=46)
    checkpoint = study / "model/ckpt_selected.pkl"
    policy, metadata = load_checkpoint(checkpoint, development.environment_fingerprint)
    if metadata.get("checkpoint_step") != 461000:
        raise ValueError("selected checkpoint is not the frozen training decision")
    d0_checkpoint = (
        root
        / "diagnostics/double_bottleneck_toy_transfer_audit/nominal_25k/ckpt_final.pkl"
    )
    canonical, _ = load_checkpoint(d0_checkpoint, development.environment_fingerprint)
    observations = _training_observations(study)
    tree, training_rows = _support_tree(observations, canonical.config)
    del observations
    gc.collect()
    result = {
        "schema": "double_bottleneck_sxl_frozen_evaluation_v1",
        "preregistration_sha256": _sha(study / "PREREGISTRATION.json"),
        "data_manifest_sha256": _sha(study / "data/manifest.json"),
        "training_summary_sha256": _sha(study / "model/training_summary.json"),
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": _sha(checkpoint),
        "model_selection_frozen_before_test": True,
        "individual_existing_test_outcomes_inspected_before_freeze": False,
        "individual_fresh_test_outcomes_inspected_before_freeze": False,
        "sets": {},
    }
    started = time.perf_counter()
    for name, dataset, seeds, scientific_set in (
        ("development", development, (0, 1, 2, 3), "development"),
        ("existing_untouched_test", existing, (101, 211, 307, 401), "untouched_test"),
        ("fresh_untouched_test", fresh, (503, 607, 701, 809), "fresh_untouched_test"),
    ):
        set_output = output / name
        set_output.mkdir(exist_ok=False)
        result["sets"][name] = _evaluate_set(
            "S-XL",
            policy,
            checkpoint,
            metadata,
            tree,
            training_rows,
            canonical.config,
            dataset,
            seeds,
            scientific_set,
            set_output,
        )
        print(json.dumps({"stage": "set_complete", "set": name}), flush=True)
    result["elapsed_seconds"] = time.perf_counter() - started
    comparison = output / "comparison.json"
    comparison.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    aggregate = {
        name: {
            "successes": value["aggregate"]["successes"],
            "rollouts": value["aggregate"]["rollouts"],
            "wall_collisions": value["aggregate"]["wall_collisions"],
            "agent_collisions": value["aggregate"]["agent_collisions"],
            "timeouts": value["aggregate"]["timeouts"],
            "x0_mean_distance": value["aggregate"]["support"]["timestep0_mean_distance"],
            "x0_ood": value["aggregate"]["support"]["timestep0_ood_fraction_unique_states"],
            "rollout_ood": value["aggregate"]["support"]["whole_rollout_ood_fraction"],
            "k100_collision": value["k_step"]["summary"]["100"]["collision_rate"],
            "teacher_rmse": value["teacher_forced"]["overall"]["joint_action_rmse"],
        }
        for name, value in result["sets"].items()
    }
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
