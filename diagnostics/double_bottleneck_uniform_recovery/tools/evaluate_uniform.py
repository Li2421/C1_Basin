#!/usr/bin/env python3
"""Freeze matched development and untouched-test results before failure inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from diagnostics.double_bottleneck_recovery_v2.tools.evaluate_v2 import (
    SUPPORT_THRESHOLD,
    _aggregate_rollouts,
    _k_step,
    _support_distances,
    _support_tree,
    _teacher,
    _trajectory_observations,
)
from double_bottleneck.environment import Config
from double_bottleneck.evaluate_flowbc_4a import _rollout
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset


DEVELOPMENT_SEEDS = (0, 1, 2, 3)
TEST_SEEDS = (101, 211, 307, 401)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path, category: str | None = None):
    with np.load(path, allow_pickle=False) as archive:
        observations = archive["observations"]
        actions = archive["actions"]
        if category is not None:
            mask = archive["category"] == category
            observations, actions = observations[mask], actions[mask]
    return observations.astype(np.float32), actions.astype(np.float32)


def _components(root: Path):
    nominal = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "train", seed=0
    )
    data = root / "diagnostics/double_bottleneck_uniform_recovery/data"
    return {
        "nominal": nominal.all_transitions(),
        "v1": _load_npz(
            root / "diagnostics/double_bottleneck_toy_transfer_audit/recovery_train.npz"
        ),
        "transition": _load_npz(
            root / "diagnostics/double_bottleneck_recovery_v2/data/targeted_train.npz",
            "transition",
        ),
        "u_low": _load_npz(data / "u_low_train.npz"),
        "u_mid": _load_npz(data / "u_mid_train.npz"),
        "u_high": _load_npz(data / "u_high_train.npz"),
    }


def _evaluate_one(
    variant: str,
    checkpoint: Path,
    component_names: tuple[str, ...],
    components,
    canonical_config,
    dataset: FlowBC4ADataset,
    seeds: tuple[int, ...],
    output: Path,
):
    config = Config(**dataset.config)
    policy, checkpoint_metadata = load_checkpoint(
        checkpoint, dataset.environment_fingerprint
    )
    tree, support_rows = _support_tree(components, component_names, canonical_config)
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    rollout_rows = []
    all_distances = []
    rollout_id = 0
    for family in dataset.family_names:
        episode = dataset.by_family[family][0]
        for seed in seeds:
            summary, arrays = _rollout(policy, dataset, episode, seed, rollout_id)
            observations = _trajectory_observations(
                arrays["positions"],
                arrays["actions"],
                episode.initial_velocities,
                dataset.goals,
            )
            distances = _support_distances(tree, observations, canonical_config)
            summary["outside_support_fraction"] = float(
                np.mean(distances > SUPPORT_THRESHOLD)
            )
            summary["mean_support_distance"] = float(np.mean(distances))
            summary["first_outside_support_step"] = (
                int(np.flatnonzero(distances > SUPPORT_THRESHOLD)[0])
                if np.any(distances > SUPPORT_THRESHOLD)
                else None
            )
            all_distances.append(distances)
            np.savez_compressed(
                trajectory_dir / f"{variant.lower()}_{rollout_id:03d}.npz",
                positions=arrays["positions"],
                observations=observations,
                actions=arrays["actions"],
                support_distances=distances,
            )
            rollout_rows.append(summary)
            rollout_id += 1
    aggregate = _aggregate_rollouts(rollout_rows)
    distances = np.concatenate(all_distances)
    aggregate["support"] = {
        "training_rows": support_rows,
        "common_coordinate_threshold": SUPPORT_THRESHOLD,
        "outside_fraction": float(np.mean(distances > SUPPORT_THRESHOLD)),
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
    }
    result = {
        "schema": "double_bottleneck_uniform_recovery_evaluation_v1",
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "training_components": list(component_names),
        "rollout_seeds": list(seeds),
        "rollout_protocol": f"{len(dataset.family_names)} families x {len(seeds)} fixed seeds; canonical raw rollout",
        "support_protocol": "canonical nominal-2k normalized 72D RMS NN; fixed q99 0.039569792891474595",
        "aggregate": aggregate,
        "rollouts": rollout_rows,
        "teacher_forced": _teacher(policy, dataset, config),
        "k_step": _k_step(policy, dataset, config),
    }
    (output / f"{variant.lower()}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_uniform_recovery/evaluation"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    development = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "val", seed=44
    )
    untouched = FlowBC4ADataset(
        root
        / "diagnostics/double_bottleneck_uniform_recovery/data/untouched_test_dataset",
        "val",
        seed=45,
    )
    variants = {
        "D0": (
            root / "diagnostics/double_bottleneck_toy_transfer_audit/nominal_25k/ckpt_final.pkl",
            ("nominal",),
        ),
        "D1": (
            root / "diagnostics/double_bottleneck_toy_transfer_audit/recovery_union_25k/ckpt_final.pkl",
            ("nominal", "v1"),
        ),
        "D2-T": (
            root / "diagnostics/double_bottleneck_recovery_v2/models/d2_t/ckpt_final.pkl",
            ("nominal", "v1", "transition"),
        ),
        "U-Low": (
            root / "diagnostics/double_bottleneck_uniform_recovery/models/u_low/ckpt_final.pkl",
            ("nominal", "u_low"),
        ),
        "U-Mid": (
            root / "diagnostics/double_bottleneck_uniform_recovery/models/u_mid/ckpt_final.pkl",
            ("nominal", "u_mid"),
        ),
        "U-High": (
            root / "diagnostics/double_bottleneck_uniform_recovery/models/u_high/ckpt_final.pkl",
            ("nominal", "u_high"),
        ),
    }
    canonical, _ = load_checkpoint(variants["D0"][0], development.environment_fingerprint)
    components = _components(root)
    result = {
        "schema": "double_bottleneck_uniform_recovery_comparison_v1",
        "preregistration_sha256": _sha(
            root / "diagnostics/double_bottleneck_uniform_recovery/PREREGISTRATION.json"
        ),
        "individual_model_outcomes_inspected_before_freeze": False,
        "sets": {},
    }
    started = time.perf_counter()
    for set_name, dataset, seeds in (
        ("development", development, DEVELOPMENT_SEEDS),
        ("untouched_test", untouched, TEST_SEEDS),
    ):
        set_output = output / set_name
        set_output.mkdir()
        result["sets"][set_name] = {}
        for variant, (checkpoint, names) in variants.items():
            result["sets"][set_name][variant] = _evaluate_one(
                variant,
                checkpoint,
                names,
                components,
                canonical.config,
                dataset,
                seeds,
                set_output,
            )
    result["elapsed_seconds"] = time.perf_counter() - started
    comparison = output / "comparison.json"
    comparison.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Aggregate output occurs only after the frozen file exists.
    print(
        json.dumps(
            {
                set_name: {
                    variant: {
                        key: values["aggregate"][key]
                        for key in (
                            "successes",
                            "wall_collisions",
                            "agent_collisions",
                            "timeouts",
                            "median_episode_steps",
                        )
                    }
                    for variant, values in variants_result.items()
                }
                for set_name, variants_result in result["sets"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
