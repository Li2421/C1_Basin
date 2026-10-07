#!/usr/bin/env python3
"""Replay residual D2/D2-T failures against support and recovery experts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.spatial import cKDTree

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.expert import CentralizedExpert
from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import phase_labels, query_reference_recovery


THRESHOLD = 0.039569792891474595


def _load(path, category=None):
    with np.load(path, allow_pickle=False) as data:
        obs = data["observations"]
        if category is not None:
            obs = obs[data["category"] == category]
    return obs.astype(np.float32)


def _support(root, canonical_config):
    nominal = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "train"
    ).observations
    v1 = _load(root / "diagnostics/double_bottleneck_toy_transfer_audit/recovery_train.npz")
    target = root / "diagnostics/double_bottleneck_recovery_v2/data/targeted_train.npz"
    transition = _load(target, "transition")
    goal = _load(target, "goal")
    velocity = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_recovery_v2/data/velocity_trajectories",
        "train",
    ).observations
    components = {
        "nominal": nominal,
        "v1": v1,
        "transition": transition,
        "goal": goal,
        "velocity": velocity,
    }
    mean = np.asarray(canonical_config["obs_mean"])
    scale = np.asarray(canonical_config["obs_scale"])

    def make(names):
        arrays = []
        labels = []
        for name in names:
            arrays.append(components[name])
            labels.extend([name] * len(components[name]))
        flat = np.concatenate(arrays).reshape((-1, 72)).astype(np.float64)
        points = ((flat - mean) / scale) / np.sqrt(72)
        return cKDTree(points), np.asarray(labels)

    return make(("nominal", "v1", "transition", "goal", "velocity")), make(
        ("v1", "transition", "goal", "velocity")
    )


def _query_support(tree_and_labels, observation, canonical_config):
    tree, labels = tree_and_labels
    value = observation.reshape(72).astype(np.float64)
    point = (
        (value - np.asarray(canonical_config["obs_mean"]))
        / np.asarray(canonical_config["obs_scale"])
        / np.sqrt(72)
    )
    distance, index = tree.query(point)
    return {"distance": float(distance), "component": str(labels[int(index)])}


def _expert_manifold(family_episodes, config):
    points = []
    mapping = []
    phase_cache = {}
    for episode_index, episode in enumerate(family_episodes):
        labels = phase_labels(episode, config)
        phase_cache[episode_index] = labels
        points.append(episode.positions.reshape((len(episode.positions), 8)))
        mapping.extend((episode_index, step) for step in range(len(episode.positions)))
    values = np.concatenate(points) / np.sqrt(8)
    return cKDTree(values), mapping, phase_cache


def _nearest_candidates(tree, mapping, position, count=128):
    _, indices = tree.query(position.reshape(8) / np.sqrt(8), k=count)
    result = []
    seen_episodes = set()
    for raw in np.atleast_1d(indices):
        pair = mapping[int(raw)]
        # Enumerate nearest states from distinct normal expert hypotheses.
        if pair[0] not in seen_episodes:
            seen_episodes.add(pair[0])
            result.append(pair)
        if len(result) == 8:
            break
    return result


def _fresh_replan(config, regime, position, velocity):
    try:
        env = DoubleBottleneckEnv(config)
        env.reset(position, regime=regime)
        state = env.augmented_state()
        state["last_applied_velocity"] = velocity.copy()
        env.restore_augmented_state(state)
        plan = CentralizedExpert().plan(env)
        return {
            "success": bool(plan.success),
            "terminal_reason": plan.terminal_reason,
            "candidate_count": plan.candidate_count,
        }
    except (ValueError, RuntimeError) as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


def _local_recoveries(family_episodes, candidates, position, velocity):
    attempts = []
    for episode_index, source_step in candidates:
        episode = family_episodes[episode_index]
        if source_step >= episode.length:
            source_step = episode.length - 1
        try:
            result = query_reference_recovery(
                episode, source_step, position, velocity
            )
            attempts.append(
                {
                    "source_rollout_id": episode.rollout_id,
                    "source_step": source_step,
                    "success": result.success,
                    "terminal_reason": result.terminal_reason,
                    "mode_signature_match": result.mode_signature_match,
                    "min_wall_clearance": result.min_wall_clearance,
                    "min_pair_clearance": result.min_pair_clearance,
                }
            )
            if result.success:
                break
        except ValueError as error:
            attempts.append(
                {
                    "source_rollout_id": episode.rollout_id,
                    "source_step": source_step,
                    "success": False,
                    "error": f"ValueError: {error}",
                }
            )
    return {"success": any(row["success"] for row in attempts), "attempts": attempts}


def analyze_variant(
    root,
    variant,
    evaluation,
    dataset,
    config,
    canonical_config,
    full_support,
    recovery_support,
):
    results = []
    for row in evaluation["rollouts"]:
        if row["success"]:
            continue
        rollout_id = int(row["rollout_id"])
        path = (
            root
            / "diagnostics/double_bottleneck_recovery_v2/evaluation/trajectories"
            / f"{variant.lower()}_{rollout_id:03d}.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            positions = data["positions"].astype(np.float64)
            observations = data["observations"].astype(np.float32)
        family_episodes = dataset.by_family[row["family_id"]]
        expert_tree, mapping, phase_cache = _expert_manifold(family_episodes, config)
        manifold_distance, nearest_index = expert_tree.query(
            positions.reshape((-1, 8)) / np.sqrt(8)
        )
        clear = np.flatnonzero(manifold_distance > 0.08)
        divergence_step = int(clear[0]) if len(clear) else None
        probe_step = divergence_step if divergence_step is not None else max(0, len(positions) - 2)
        collision_probe_step = max(0, len(positions) - 2)

        def analyze_state(step):
            nearest_pair = mapping[int(nearest_index[step])]
            nearest_episode, nearest_step = nearest_pair
            phase = str(phase_cache[nearest_episode][nearest_step])
            full = _query_support(full_support, observations[step], canonical_config)
            recovery = _query_support(
                recovery_support, observations[step], canonical_config
            )
            candidates = _nearest_candidates(expert_tree, mapping, positions[step])
            velocity = observations[step, :, 2:4].astype(np.float64)
            return {
                "step": step,
                "expert_manifold_position_rms": float(manifold_distance[step]),
                "nearest_expert_rollout": family_episodes[nearest_episode].rollout_id,
                "nearest_expert_step": nearest_step,
                "phase": phase,
                "nearest_D2_support": full,
                "nearest_recovery_support": recovery,
                "inside_fixed_support": full["distance"] <= THRESHOLD,
                "nearby_recovery_example": recovery["distance"] <= THRESHOLD,
                "fresh_centralized_replan": _fresh_replan(
                    config, family_episodes[0].regime, positions[step], velocity
                ),
                "local_hypothesis_recovery": _local_recoveries(
                    family_episodes, candidates, positions[step], velocity
                ),
            }

        results.append(
            {
                "rollout_id": rollout_id,
                "family_id": row["family_id"],
                "regime": row["regime"],
                "seed": row["seed"],
                "termination": row["termination"],
                "episode_steps": row["episode_steps"],
                "wall_collision": row["wall_collision"],
                "agent_collision": row["agent_collision"],
                "first_clear_divergence_step": divergence_step,
                "first_clear_divergence": (
                    analyze_state(divergence_step) if divergence_step is not None else None
                ),
                "terminal_probe_when_no_clear_divergence": (
                    analyze_state(probe_step) if divergence_step is None else None
                ),
                "last_safe_pre_collision": analyze_state(collision_probe_step),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostics/double_bottleneck_recovery_v2/failure_replay.json"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    dataset = FlowBC4ADataset(
        root / "diagnostics/double_bottleneck_expert_dataset_8mode", "val"
    )
    config = Config(**dataset.config)
    canonical, _ = load_checkpoint(
        root / "diagnostics/double_bottleneck_toy_transfer_audit/nominal_25k/ckpt_final.pkl",
        dataset.environment_fingerprint,
    )
    full_support, recovery_support = _support(root, canonical.config)
    result = {
        "schema": "double_bottleneck_recovery_v2_failure_replay_v1",
        "position_divergence_threshold_m": 0.08,
        "support_threshold": THRESHOLD,
        "expert_recovery_semantics": {
            "fresh": "existing CentralizedExpert enumerates all 8 normal hypotheses from the failed state",
            "local": "nearest normal expert hypotheses use scenario-local reference recovery; no policy mode input",
        },
        "variants": {},
    }
    for variant in ("D2-T", "D2"):
        evaluation = json.loads(
            (
                root
                / "diagnostics/double_bottleneck_recovery_v2/evaluation"
                / f"{variant.lower()}.json"
            ).read_text()
        )
        result["variants"][variant] = analyze_variant(
            root,
            variant,
            evaluation,
            dataset,
            config,
            canonical.config,
            full_support,
            recovery_support,
        )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {}
    for variant, rows in result["variants"].items():
        summary[variant] = {
            "failures": len(rows),
            "failures_with_clear_divergence": sum(
                r["first_clear_divergence"] is not None for r in rows
            ),
            "divergence_phases": dict(
                sorted(
                    Counter(
                        r["first_clear_divergence"]["phase"]
                        for r in rows
                        if r["first_clear_divergence"] is not None
                    ).items()
                )
            ),
            "divergence_inside_support": sum(
                r["first_clear_divergence"]["inside_fixed_support"]
                for r in rows
                if r["first_clear_divergence"] is not None
            ),
            "divergence_near_recovery": sum(
                r["first_clear_divergence"]["nearby_recovery_example"]
                for r in rows
                if r["first_clear_divergence"] is not None
            ),
            "local_recoverable_at_divergence": sum(
                r["first_clear_divergence"]["local_hypothesis_recovery"]["success"]
                for r in rows
                if r["first_clear_divergence"] is not None
            ),
            "local_recoverable_pre_collision": sum(
                r["last_safe_pre_collision"]["local_hypothesis_recovery"]["success"]
                for r in rows
            ),
            "fresh_replan_at_divergence": sum(
                r["first_clear_divergence"]["fresh_centralized_replan"]["success"]
                for r in rows
                if r["first_clear_divergence"] is not None
            ),
        }
    result["summary"] = summary
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
