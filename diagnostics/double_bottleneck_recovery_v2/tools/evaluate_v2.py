#!/usr/bin/env python3
"""Matched closed-loop, support, teacher, and K-step Recovery-V2 evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import numpy as np
from scipy.spatial import cKDTree

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.evaluate_flowbc_4a import _rollout, _radial_bound64
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import observation_from_state, phase_labels


HORIZONS = (1, 5, 10, 25, 50, 100)
SUPPORT_THRESHOLD = 0.039569792891474595


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path, category: str | None = None):
    with np.load(path, allow_pickle=False) as data:
        obs = data["observations"]
        act = data["actions"]
        if category is not None:
            mask = data["category"] == category
            obs, act = obs[mask], act[mask]
    return obs.astype(np.float32), act.astype(np.float32)


def _components(root: Path):
    nominal_train = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "train"
    )
    result = {
        "nominal": nominal_train.all_transitions(),
        "v1": _load_npz(
            root / "diagnostics/double_bottleneck_toy_transfer_audit/recovery_train.npz"
        ),
    }
    target = root / "diagnostics/double_bottleneck_recovery_v2/data/targeted_train.npz"
    result["transition"] = _load_npz(target, "transition")
    result["goal"] = _load_npz(target, "goal")
    velocity = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_recovery_v2/data/velocity_trajectories",
        "train",
    )
    result["velocity"] = velocity.all_transitions()
    return result


def _support_tree(components, names, canonical_config):
    observations = np.concatenate([components[name][0] for name in names], axis=0)
    flat = observations.reshape((len(observations), -1)).astype(np.float64)
    mean = np.asarray(canonical_config["obs_mean"], dtype=np.float64)
    scale = np.asarray(canonical_config["obs_scale"], dtype=np.float64)
    normalized = (flat - mean) / scale
    return cKDTree(normalized / np.sqrt(normalized.shape[1])), len(observations)


def _trajectory_observations(positions, actions, initial_velocity, goals):
    velocity = np.empty_like(positions)
    velocity[0] = initial_velocity
    velocity[1:] = actions
    return np.stack(
        [observation_from_state(p, v, goals) for p, v in zip(positions, velocity, strict=True)]
    )


def _support_distances(tree, observations, canonical_config):
    flat = observations.reshape((len(observations), -1)).astype(np.float64)
    normalized = (flat - np.asarray(canonical_config["obs_mean"])) / np.asarray(
        canonical_config["obs_scale"]
    )
    distance, _ = tree.query(normalized / np.sqrt(normalized.shape[1]), workers=1)
    return np.asarray(distance, dtype=np.float64)


def _phase_arrays(dataset, config):
    return np.concatenate([phase_labels(episode, config)[:-1] for episode in dataset.episodes])


def _wall_directed(actions, positions, walls):
    # Unit vector from closest wall point toward agent; negative projection points at wall.
    p = positions[:, :, None, :]
    a = walls[None, None, :, 0, :]
    b = walls[None, None, :, 1, :]
    segment = b - a
    alpha = np.sum((p - a) * segment, axis=-1) / np.maximum(
        np.sum(segment * segment, axis=-1), 1e-30
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = a + alpha[..., None] * segment
    offset = p - closest
    distance = np.linalg.norm(offset, axis=-1)
    nearest = np.argmin(distance, axis=-1)
    normal = np.take_along_axis(offset, nearest[..., None, None], axis=2)[:, :, 0]
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-30)
    return np.maximum(-np.sum(actions * normal, axis=-1), 0.0)


def _teacher(policy, dataset, config, seed: int = 8811):
    observations, expert_actions = dataset.all_transitions()
    positions = observations[:, :, :2].astype(np.float64)
    phases = _phase_arrays(dataset, config)
    samples = []
    root = jax.random.PRNGKey(seed)
    for index in range(4):
        raw = np.asarray(
            policy.sample_actions(observations, jax.random.fold_in(root, index)),
            dtype=np.float64,
        )
        samples.append(_radial_bound64(raw, config.max_speed))
    predicted = np.stack(samples, axis=1)
    error = predicted - expert_actions[:, None]
    per_state_rmse = np.sqrt(np.mean(error**2, axis=(2, 3))).mean(axis=1)
    walls = DoubleBottleneckEnv(config).walls
    expert_wall = _wall_directed(expert_actions.astype(np.float64), positions, walls)
    predicted_wall = np.stack(
        [_wall_directed(predicted[:, index], positions, walls) for index in range(4)], axis=1
    )
    wall_error = np.abs(predicted_wall - expert_wall[:, None]).mean(axis=(1, 2))

    def subset(mask):
        return {
            "states": int(np.sum(mask)),
            "joint_action_rmse": float(np.mean(per_state_rmse[mask])),
            "wall_directed_absolute_error": float(np.mean(wall_error[mask])),
        }

    result = {
        "states": len(observations),
        "samples_per_state": 4,
        "overall": subset(np.ones(len(observations), dtype=bool)),
        "by_phase": {
            phase: subset(phases == phase) for phase in sorted(set(phases.tolist()))
        },
    }
    combined = np.isin(phases, ("waiting_yielding", "coordination_mode_transition"))
    goal = np.isin(phases, ("final_goal_approach", "near_goal_termination"))
    result["waiting_transition"] = subset(combined)
    result["goal_region"] = subset(goal)
    return result


def _anchors(episode, config):
    labels = phase_labels(episode, config)[:-1]
    groups = {
        "approach_first_bottleneck": np.isin(
            labels, ("initial_approach", "first_bottleneck_approach")
        ),
        "waiting_yielding": labels == "waiting_yielding",
        "first_bottleneck": labels == "bottleneck_traversal",
        "chamber": labels == "chamber_traversal",
        "second_bottleneck": labels == "second_bottleneck",
        "final_goal_approach": np.isin(
            labels, ("final_goal_approach", "near_goal_termination")
        ),
    }
    result = []
    for phase, mask in groups.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"episode {episode.rollout_id} lacks {phase}")
        result.append((phase, int(indices[len(indices) // 2])))
    return result


def _restore_intermediate(episode, step: int, config: Config):
    env = DoubleBottleneckEnv(config)
    env.reset(episode.positions[step], regime=episode.regime)
    state = env.augmented_state()
    state["last_applied_velocity"] = episode.observations[step, :, 2:4].astype(np.float64)
    env.restore_augmented_state(state)
    return env


def _k_step(policy, dataset, config):
    rows = []
    continuation_id = 0
    for episode in dataset.episodes:
        for phase, source_step in _anchors(episode, config):
            for horizon in HORIZONS:
                env = _restore_intermediate(episode, source_step, config)
                predicted_positions = [env.positions.copy()]
                predicted_actions = []
                collided = False
                first_divergence = None
                min_wall = float(env.distances()[0].min())
                min_pair = float(env.distances()[1].min())
                available = min(horizon, episode.length - source_step)
                episode_key = jax.random.fold_in(jax.random.PRNGKey(77123), continuation_id)
                for local_step in range(available):
                    step_key = jax.random.fold_in(episode_key, local_step)
                    raw = np.asarray(
                        policy.sample_actions(env.observation()[None], step_key)[0],
                        dtype=np.float64,
                    )
                    action = _radial_bound64(raw, config.max_speed)
                    _, _, done, info = env.step(action)
                    predicted_actions.append(action)
                    predicted_positions.append(env.positions.copy())
                    min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
                    min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
                    reference = episode.positions[source_step + local_step + 1]
                    deviation = float(np.sqrt(np.mean((env.positions - reference) ** 2)))
                    if first_divergence is None and deviation > 0.08:
                        first_divergence = local_step + 1
                    if done:
                        collided = bool(info["wall_collision"] or info["agent_collision"])
                        break
                executed = len(predicted_actions)
                predicted_positions = np.asarray(predicted_positions)
                predicted_actions = np.asarray(predicted_actions)
                reference_positions = episode.positions[
                    source_step : source_step + executed + 1
                ]
                reference_actions = episode.actions[source_step : source_step + executed]
                rows.append(
                    {
                        "phase": phase,
                        "horizon": horizon,
                        "executed_steps": executed,
                        "position_rmse": float(
                            np.sqrt(np.mean((predicted_positions - reference_positions) ** 2))
                        ),
                        "action_rmse": float(
                            np.sqrt(np.mean((predicted_actions - reference_actions) ** 2))
                        ),
                        "collision_within_horizon": collided,
                        "first_divergence_step": first_divergence,
                        "minimum_wall_clearance": min_wall,
                        "minimum_pair_clearance": min_pair,
                    }
                )
            continuation_id += 1
    summary = {}
    for horizon in HORIZONS:
        selected = [row for row in rows if row["horizon"] == horizon]
        summary[str(horizon)] = {
            "continuations": len(selected),
            "position_rmse": float(np.mean([row["position_rmse"] for row in selected])),
            "action_rmse": float(np.mean([row["action_rmse"] for row in selected])),
            "collision_rate": float(
                np.mean([row["collision_within_horizon"] for row in selected])
            ),
            "minimum_wall_clearance_mean": float(
                np.mean([row["minimum_wall_clearance"] for row in selected])
            ),
            "minimum_pair_clearance_mean": float(
                np.mean([row["minimum_pair_clearance"] for row in selected])
            ),
        }
    phase_k100 = {}
    for phase in sorted({row["phase"] for row in rows}):
        selected = [
            row for row in rows if row["horizon"] == 100 and row["phase"] == phase
        ]
        phase_k100[phase] = {
            "continuations": len(selected),
            "position_rmse": float(np.mean([row["position_rmse"] for row in selected])),
            "action_rmse": float(np.mean([row["action_rmse"] for row in selected])),
            "collision_rate": float(
                np.mean([row["collision_within_horizon"] for row in selected])
            ),
        }
    return {"summary": summary, "phase_k100": phase_k100, "rows": rows}


def _aggregate_rollouts(rows):
    count = len(rows)
    by_family = {}
    for family in sorted({row["family_id"] for row in rows}):
        selected = [row for row in rows if row["family_id"] == family]
        by_family[family] = {
            "rollouts": len(selected),
            "successes": sum(row["success"] for row in selected),
            "wall_collisions": sum(row["wall_collision"] for row in selected),
            "agent_collisions": sum(row["agent_collision"] for row in selected),
            "median_steps": float(np.median([row["episode_steps"] for row in selected])),
        }
    return {
        "rollouts": count,
        "successes": sum(row["success"] for row in rows),
        "wall_collisions": sum(row["wall_collision"] for row in rows),
        "agent_collisions": sum(row["agent_collision"] for row in rows),
        "deadlocks": sum(row["deadlock"] for row in rows),
        "timeouts": sum(row["timeout"] for row in rows),
        "median_episode_steps": float(np.median([row["episode_steps"] for row in rows])),
        "mean_episode_steps": float(np.mean([row["episode_steps"] for row in rows])),
        "minimum_wall_clearance": float(min(row["min_swept_wall_clearance"] for row in rows)),
        "minimum_pair_clearance": float(
            min(row["min_swept_pair_surface_distance"] for row in rows)
        ),
        "successful_mode_signatures": sorted(
            {
                row["coordination_mode"]["signature"]
                for row in rows
                if row["success"]
            }
        ),
        "by_family": by_family,
    }


def evaluate_one(
    variant,
    checkpoint,
    component_names,
    components,
    canonical_config,
    dataset,
    output,
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
        for seed in (0, 1, 2, 3):
            summary, arrays = _rollout(policy, dataset, episode, seed, rollout_id)
            observations = _trajectory_observations(
                arrays["positions"], arrays["actions"], episode.initial_velocities, dataset.goals
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
        "schema": "double_bottleneck_recovery_v2_matched_evaluation_v1",
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "training_components": list(component_names),
        "rollout_protocol": "3 heldout families x seeds 0,1,2,3; canonical raw rollout",
        "support_protocol": "canonical nominal-2k normalized 72D RMS NN; fixed q99 0.039569792891474595",
        "aggregate": aggregate,
        "rollouts": rollout_rows,
        "teacher_forced": _teacher(policy, dataset, config),
        "k_step": _k_step(policy, dataset, config),
    }
    (output / f"{variant.lower()}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "variant": variant,
                **{key: aggregate[key] for key in ("successes", "wall_collisions", "agent_collisions", "timeouts", "median_episode_steps")},
                "ood": aggregate["support"]["outside_fraction"],
                "k100_collision": result["k_step"]["summary"]["100"]["collision_rate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_recovery_v2/evaluation"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    dataset = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "val", seed=44
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
        "D2-G": (
            root / "diagnostics/double_bottleneck_recovery_v2/models/d2_g/ckpt_final.pkl",
            ("nominal", "v1", "goal"),
        ),
        "D2-V": (
            root / "diagnostics/double_bottleneck_recovery_v2/models/d2_v/ckpt_final.pkl",
            ("nominal", "v1", "velocity"),
        ),
        "D2": (
            root / "diagnostics/double_bottleneck_recovery_v2/models/d2/ckpt_final.pkl",
            ("nominal", "v1", "transition", "goal", "velocity"),
        ),
    }
    canonical, _ = load_checkpoint(variants["D0"][0], dataset.environment_fingerprint)
    components = _components(root)
    comparison = {}
    started = time.perf_counter()
    for variant, (checkpoint, names) in variants.items():
        comparison[variant] = evaluate_one(
            variant,
            checkpoint,
            names,
            components,
            canonical.config,
            dataset,
            output,
        )
    result = {
        "schema": "double_bottleneck_recovery_v2_evaluation_comparison_v1",
        "elapsed_seconds": time.perf_counter() - started,
        "variants": comparison,
    }
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
