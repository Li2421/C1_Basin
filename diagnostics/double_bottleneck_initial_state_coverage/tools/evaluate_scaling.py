#!/usr/bin/env python3
"""Freeze development and untouched-test initial-coverage evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import numpy as np
from scipy.spatial import cKDTree

from diagnostics.double_bottleneck_recovery_v2.tools.evaluate_v2 import (
    SUPPORT_THRESHOLD,
    _aggregate_rollouts,
    _k_step,
    _phase_arrays,
    _trajectory_observations,
    _wall_directed,
)
from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.evaluate_flowbc_4a import _radial_bound64, _rollout
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset


VARIANTS = ("D0", "D1", "U-High", "D2-T", "S-Small", "S-Medium", "S-Large")
DEVELOPMENT_SEEDS = (0, 1, 2, 3)
TEST_SEEDS = (101, 211, 307, 401)
TEACHER_BATCH = 4096


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


def _new_variant_arrays(root: Path, variant: str):
    study = root / "diagnostics/double_bottleneck_initial_state_coverage"
    selection = json.loads(
        (
            study
            / "data/variant_manifests"
            / f"{variant.lower().replace('-', '_')}.json"
        ).read_text()
    )
    selected = set(selection["family_ids"])
    dataset = FlowBC4ADataset(study / "data/train_pool", "train", seed=0)
    observations = np.concatenate(
        [
            episode.observations[:-1]
            for episode in dataset.episodes
            if episode.family_id in selected
        ],
        axis=0,
    ).astype(np.float32)
    actions = np.concatenate(
        [
            episode.actions
            for episode in dataset.episodes
            if episode.family_id in selected
        ],
        axis=0,
    ).astype(np.float32)
    with np.load(study / "data/recovery_train.npz", allow_pickle=False) as archive:
        mask = np.isin(archive["source_family_id"].astype(str), list(selected))
        recovery_obs = archive["observations"][mask].astype(np.float32)
        recovery_act = archive["actions"][mask].astype(np.float32)
    return (
        np.concatenate((observations, recovery_obs), axis=0),
        np.concatenate((actions, recovery_act), axis=0),
    )


def _training_arrays(root: Path, variant: str):
    old_nominal = root / "diagnostics/double_bottleneck_expert_dataset_8mode"
    if variant in ("S-Small", "S-Medium", "S-Large"):
        return _new_variant_arrays(root, variant)
    nominal = FlowBC4ADataset(old_nominal, "train", seed=0)
    nominal_arrays = (nominal.observations, nominal.actions)
    if variant == "D0":
        return nominal_arrays
    if variant == "D1":
        recovery = _load_npz(
            root / "diagnostics/double_bottleneck_toy_transfer_audit/recovery_train.npz"
        )
        return tuple(
            np.concatenate((nominal_arrays[index], recovery[index]), axis=0)
            for index in range(2)
        )
    if variant == "U-High":
        recovery = _load_npz(
            root / "diagnostics/double_bottleneck_uniform_recovery/data/u_high_train.npz"
        )
        return tuple(
            np.concatenate((nominal_arrays[index], recovery[index]), axis=0)
            for index in range(2)
        )
    if variant == "D2-T":
        v1 = _load_npz(
            root / "diagnostics/double_bottleneck_toy_transfer_audit/recovery_train.npz"
        )
        transition = _load_npz(
            root / "diagnostics/double_bottleneck_recovery_v2/data/targeted_train.npz",
            "transition",
        )
        return tuple(
            np.concatenate((nominal_arrays[index], v1[index], transition[index]), axis=0)
            for index in range(2)
        )
    raise KeyError(variant)


def _support_tree(observations: np.ndarray, canonical_config):
    flat = observations.reshape((len(observations), -1)).astype(np.float64)
    flat -= np.asarray(canonical_config["obs_mean"], dtype=np.float64)
    flat /= np.asarray(canonical_config["obs_scale"], dtype=np.float64)
    flat /= np.sqrt(flat.shape[1])
    return cKDTree(flat, copy_data=False), len(flat)


def _support_distances(tree, observations, canonical_config):
    flat = observations.reshape((len(observations), -1)).astype(np.float64)
    flat -= np.asarray(canonical_config["obs_mean"], dtype=np.float64)
    flat /= np.asarray(canonical_config["obs_scale"], dtype=np.float64)
    flat /= np.sqrt(flat.shape[1])
    values, _ = tree.query(flat, workers=1)
    return np.asarray(values, dtype=np.float64)


def _teacher_batched(policy, dataset, config, seed=8811):
    observations, expert_actions = dataset.all_transitions()
    positions = observations[:, :, :2].astype(np.float64)
    phases = _phase_arrays(dataset, config)
    walls = DoubleBottleneckEnv(config).walls
    expert_wall = _wall_directed(expert_actions.astype(np.float64), positions, walls)
    state_rmse = np.zeros(len(observations), dtype=np.float64)
    state_wall = np.zeros(len(observations), dtype=np.float64)
    root = jax.random.PRNGKey(seed)
    for begin in range(0, len(observations), TEACHER_BATCH):
        end = min(len(observations), begin + TEACHER_BATCH)
        batch = observations[begin:end]
        batch_positions = positions[begin:end]
        batch_expert = expert_actions[begin:end].astype(np.float64)
        rmse_samples = []
        wall_samples = []
        for sample_index in range(4):
            key = jax.random.fold_in(
                jax.random.fold_in(root, sample_index), begin // TEACHER_BATCH
            )
            raw = np.asarray(policy.sample_actions(batch, key), dtype=np.float64)
            predicted = _radial_bound64(raw, config.max_speed)
            rmse_samples.append(
                np.sqrt(np.mean((predicted - batch_expert) ** 2, axis=(1, 2)))
            )
            predicted_wall = _wall_directed(predicted, batch_positions, walls)
            wall_samples.append(
                np.mean(
                    np.abs(predicted_wall - expert_wall[begin:end]), axis=1
                )
            )
        state_rmse[begin:end] = np.mean(np.stack(rmse_samples, axis=1), axis=1)
        state_wall[begin:end] = np.mean(np.stack(wall_samples, axis=1), axis=1)

    def subset(mask):
        return {
            "states": int(np.sum(mask)),
            "joint_action_rmse": float(np.mean(state_rmse[mask])),
            "wall_directed_absolute_error": float(np.mean(state_wall[mask])),
        }

    result = {
        "states": len(observations),
        "samples_per_state": 4,
        "batch_size": TEACHER_BATCH,
        "overall": subset(np.ones(len(observations), dtype=bool)),
        "by_phase": {
            phase: subset(phases == phase) for phase in sorted(set(phases.tolist()))
        },
    }
    result["waiting_transition"] = subset(
        np.isin(phases, ("waiting_yielding", "coordination_mode_transition"))
    )
    result["goal_region"] = subset(
        np.isin(phases, ("final_goal_approach", "near_goal_termination"))
    )
    return result


def _k_subset(dataset, scientific_set: str):
    if scientific_set == "development":
        return dataset
    episodes = []
    for family in dataset.family_names:
        by_mode = dataset.by_family[family]
        if len(by_mode) != 8:
            raise ValueError("untouched test family must contain all 8 modes")
        episodes.extend((by_mode[0], by_mode[4]))
    return SimpleNamespace(episodes=tuple(episodes))


def _coverage_summary(rollout_rows, distances_by_rollout):
    all_distances = np.concatenate(distances_by_rollout)
    x0 = np.asarray([values[0] for values in distances_by_rollout])
    family_x0 = {}
    for row, value in zip(rollout_rows, x0, strict=True):
        family_x0.setdefault(row["family_id"], float(value))
    supported = x0 <= SUPPORT_THRESHOLD
    later_exit = []
    later_exit_steps = []
    for is_supported, values in zip(supported, distances_by_rollout, strict=True):
        if not is_supported:
            continue
        outside = np.flatnonzero(values[1:] > SUPPORT_THRESHOLD)
        later_exit.append(bool(len(outside)))
        if len(outside):
            later_exit_steps.append(int(outside[0] + 1))
    return {
        "training_rows": None,
        "common_coordinate_threshold": SUPPORT_THRESHOLD,
        "timestep0_ood_fraction_rollouts": float(np.mean(x0 > SUPPORT_THRESHOLD)),
        "timestep0_ood_fraction_unique_states": float(
            np.mean(np.asarray(list(family_x0.values())) > SUPPORT_THRESHOLD)
        ),
        "timestep0_mean_distance": float(np.mean(x0)),
        "whole_rollout_ood_fraction": float(
            np.mean(all_distances > SUPPORT_THRESHOLD)
        ),
        "whole_rollout_mean_distance": float(np.mean(all_distances)),
        "supported_initial_rollouts": int(np.sum(supported)),
        "later_exit_fraction_given_supported_x0": (
            float(np.mean(later_exit)) if later_exit else None
        ),
        "median_later_exit_step_given_exit": (
            float(np.median(later_exit_steps)) if later_exit_steps else None
        ),
    }


def _evaluate_set(
    variant,
    policy,
    checkpoint,
    checkpoint_metadata,
    tree,
    training_rows,
    canonical_config,
    dataset,
    seeds,
    scientific_set,
    output,
):
    config = Config(**dataset.config)
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    rollout_rows = []
    distances_by_rollout = []
    rollout_id = 0
    for family in dataset.family_names:
        episode = dataset.by_family[family][0]
        for seed in seeds:
            summary, arrays = _rollout(policy, dataset, episode, seed, rollout_id)
            observations = _trajectory_observations(
                arrays["positions"], arrays["actions"], episode.initial_velocities, dataset.goals
            )
            distances = _support_distances(tree, observations, canonical_config)
            summary["timestep0_support_distance"] = float(distances[0])
            summary["outside_support_fraction"] = float(
                np.mean(distances > SUPPORT_THRESHOLD)
            )
            summary["first_outside_support_step"] = (
                int(np.flatnonzero(distances > SUPPORT_THRESHOLD)[0])
                if np.any(distances > SUPPORT_THRESHOLD)
                else None
            )
            rollout_rows.append(summary)
            distances_by_rollout.append(distances)
            np.savez_compressed(
                trajectory_dir / f"{variant.lower()}_{rollout_id:03d}.npz",
                positions=arrays["positions"],
                observations=observations,
                actions=arrays["actions"],
                support_distances=distances,
            )
            rollout_id += 1
    aggregate = _aggregate_rollouts(rollout_rows)
    aggregate["support"] = _coverage_summary(rollout_rows, distances_by_rollout)
    aggregate["support"]["training_rows"] = training_rows
    k_dataset = _k_subset(dataset, scientific_set)
    result = {
        "schema": "double_bottleneck_initial_state_coverage_evaluation_v1",
        "scientific_set": scientific_set,
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "rollout_seeds": list(seeds),
        "rollout_protocol": f"{len(dataset.family_names)} independent initial states x {len(seeds)} fixed seeds",
        "support_protocol": "canonical D0 normalized 72D RMS nearest neighbor; fixed q99 0.039569792891474595",
        "aggregate": aggregate,
        "rollouts": rollout_rows,
        "teacher_forced": _teacher_batched(policy, dataset, config),
        "k_step": _k_step(policy, k_dataset, config),
    }
    (output / f"{variant.lower()}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_initial_state_coverage/evaluation"),
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
        / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
        "val",
        seed=45,
    )
    checkpoints = {
        "D0": root
        / "diagnostics/double_bottleneck_toy_transfer_audit/nominal_25k/ckpt_final.pkl",
        "D1": root
        / "diagnostics/double_bottleneck_toy_transfer_audit/recovery_union_25k/ckpt_final.pkl",
        "U-High": root
        / "diagnostics/double_bottleneck_uniform_recovery/models/u_high/ckpt_final.pkl",
        "D2-T": root
        / "diagnostics/double_bottleneck_recovery_v2/models/d2_t/ckpt_final.pkl",
        "S-Small": root
        / "diagnostics/double_bottleneck_initial_state_coverage/models/s_small/ckpt_final.pkl",
        "S-Medium": root
        / "diagnostics/double_bottleneck_initial_state_coverage/models/s_medium/ckpt_final.pkl",
        "S-Large": root
        / "diagnostics/double_bottleneck_initial_state_coverage/models/s_large/ckpt_final.pkl",
    }
    canonical, _ = load_checkpoint(checkpoints["D0"], development.environment_fingerprint)
    result = {
        "schema": "double_bottleneck_initial_state_coverage_comparison_v1",
        "preregistration_sha256": _sha(
            root
            / "diagnostics/double_bottleneck_initial_state_coverage/PREREGISTRATION.json"
        ),
        "individual_test_outcomes_inspected_before_freeze": False,
        "sets": {"development": {}, "untouched_test": {}},
    }
    started = time.perf_counter()
    for variant in VARIANTS:
        policy, metadata = load_checkpoint(
            checkpoints[variant], development.environment_fingerprint
        )
        observations, _ = _training_arrays(root, variant)
        tree, training_rows = _support_tree(observations, canonical.config)
        del observations
        gc.collect()
        print(
            json.dumps(
                {"stage": "evaluation_start", "variant": variant, "support_rows": training_rows},
                sort_keys=True,
            ),
            flush=True,
        )
        for scientific_set, dataset, seeds in (
            ("development", development, DEVELOPMENT_SEEDS),
            ("untouched_test", untouched, TEST_SEEDS),
        ):
            set_output = output / scientific_set
            set_output.mkdir(exist_ok=True)
            result["sets"][scientific_set][variant] = _evaluate_set(
                variant,
                policy,
                checkpoints[variant],
                metadata,
                tree,
                training_rows,
                canonical.config,
                dataset,
                seeds,
                scientific_set,
                set_output,
            )
        del tree, policy
        gc.collect()
        print(
            json.dumps({"stage": "evaluation_complete", "variant": variant}, sort_keys=True),
            flush=True,
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    comparison = output / "comparison.json"
    comparison.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # Only aggregate results are printed, and only after the frozen comparison exists.
    print(
        json.dumps(
            {
                set_name: {
                    variant: {
                        "successes": item["aggregate"]["successes"],
                        "wall_collisions": item["aggregate"]["wall_collisions"],
                        "agent_collisions": item["aggregate"]["agent_collisions"],
                        "timeouts": item["aggregate"]["timeouts"],
                        "x0_ood": item["aggregate"]["support"][
                            "timestep0_ood_fraction_unique_states"
                        ],
                        "rollout_ood": item["aggregate"]["support"][
                            "whole_rollout_ood_fraction"
                        ],
                    }
                    for variant, item in set_results.items()
                }
                for set_name, set_results in result["sets"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
