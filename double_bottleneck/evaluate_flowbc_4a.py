"""Evaluate raw MACFlow-official joint 4-agent Flow-BC without eta or CBF."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from .environment import Config, DoubleBottleneckEnv
from .expert_dataset import environment_fingerprint, infer_coordination_mode
from .flowbc_4a_agent import load_checkpoint
from .flowbc_4a_dataset import Episode, FlowBC4ADataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = REPO_ROOT / "diagnostics" / "double_bottleneck_macflow_pilot_v2"
DEFAULT_DATASET = REPO_ROOT / "diagnostics" / "double_bottleneck_expert_dataset"


def _initialize_env(config: Config, episode: Episode) -> DoubleBottleneckEnv:
    env = DoubleBottleneckEnv(config)
    env.reset(episode.initial_positions, regime=episode.regime)
    state = env.augmented_state()
    state["last_applied_velocity"] = episode.initial_velocities.copy()
    env.restore_augmented_state(state)
    return env


def _radial_bound64(actions: np.ndarray, max_speed: float) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    norms = np.linalg.norm(actions, axis=-1, keepdims=True)
    return actions * np.minimum(1.0, float(max_speed) / np.maximum(norms, 1e-30))


def _phase_resample(positions: np.ndarray, samples: int = 101) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    old = np.linspace(0.0, 1.0, len(positions))
    new = np.linspace(0.0, 1.0, samples)
    result = np.empty((samples, 4, 2), dtype=np.float64)
    for agent in range(4):
        for coordinate in range(2):
            result[:, agent, coordinate] = np.interp(
                new, old, positions[:, agent, coordinate]
            )
    return result


def _nearest_expert(positions: np.ndarray, experts: tuple[Episode, ...]) -> dict:
    candidate = _phase_resample(positions)
    rows = []
    for expert in experts:
        reference = _phase_resample(expert.positions)
        rmse = float(np.sqrt(np.mean(np.square(candidate - reference))))
        rows.append((rmse, expert))
    rmse, expert = min(rows, key=lambda item: item[0])
    return {
        "phase_aligned_position_rmse": rmse,
        "rollout_id": expert.rollout_id,
        "mode_signature": expert.mode_signature,
        "expert_steps": expert.length,
        "expert_path_length": float(expert.metadata["path_length"]),
    }


def _rollout(policy, dataset, episode: Episode, seed: int, rollout_id: int):
    config = Config(**dataset.config)
    env = _initialize_env(config, episode)
    episode_key = jax.random.fold_in(jax.random.PRNGKey(seed), rollout_id)
    positions = [env.positions.copy()]
    raw_actions, actions, speeds = [], [], []
    goal_errors = [np.linalg.norm(env.goals - env.positions, axis=-1)]
    min_pair = float(env.distances()[1].min())
    min_wall = float(env.distances()[0].min())
    terminal = "running"
    final_info = None
    started = time.perf_counter()
    for step in range(config.max_steps):
        observation = env.observation()[None]
        step_key = jax.random.fold_in(episode_key, step)
        raw = np.asarray(policy.sample_actions(observation, step_key)[0], dtype=np.float64)
        action = _radial_bound64(raw, config.max_speed)
        _, _, done, info = env.step(action)
        raw_actions.append(raw)
        actions.append(action)
        speeds.append(np.linalg.norm(action, axis=-1))
        positions.append(env.positions.copy())
        goal_errors.append(info["goal_errors"].copy())
        min_pair = min(min_pair, float(info["min_swept_agent_distance"]))
        min_wall = min(min_wall, float(info["min_swept_wall_distance"]))
        terminal = str(info["termination"])
        final_info = info
        if done:
            break
    elapsed = time.perf_counter() - started
    positions = np.asarray(positions)
    raw_actions = np.asarray(raw_actions)
    actions = np.asarray(actions)
    speeds = np.asarray(speeds)
    goal_errors = np.asarray(goal_errors)
    mode = infer_coordination_mode(positions, env.goals, config)
    nearest = _nearest_expert(positions, dataset.by_family[episode.family_id])
    path_length = float(np.linalg.norm(actions, axis=-1).sum() * config.dt)
    total_variation = float(
        0.0 if len(actions) < 2 else np.linalg.norm(np.diff(actions, axis=0), axis=-1).mean()
    )
    backtracking = float(np.maximum(goal_errors[1:] - goal_errors[:-1], 0.0).sum())
    summary = {
        "family_id": episode.family_id,
        "regime": episode.regime,
        "seed": int(seed),
        "rollout_id": int(rollout_id),
        "termination": terminal,
        "success": terminal == "success",
        "collision": terminal == "collision",
        "wall_collision": bool(final_info["wall_collision"]),
        "agent_collision": bool(final_info["agent_collision"]),
        "deadlock": terminal == "deadlock",
        "timeout": terminal == "timeout",
        "episode_steps": len(actions),
        "completion_time": len(actions) * config.dt if terminal == "success" else None,
        "min_swept_pair_surface_distance": min_pair,
        "min_swept_wall_clearance": min_wall,
        "path_length": path_length,
        "path_length_minus_nearest_expert": path_length - nearest["expert_path_length"],
        "mean_action_total_variation": total_variation,
        "per_agent_waiting_seconds": (np.mean(speeds < 0.025, axis=0) * len(actions) * config.dt).tolist(),
        "total_goal_error_backtracking": backtracking,
        "max_raw_speed": float(np.linalg.norm(raw_actions, axis=-1).max()),
        "max_executed_speed": float(speeds.max()),
        "terminal_positions": positions[-1].tolist(),
        "net_displacements": (positions[-1] - positions[0]).tolist(),
        "coordination_mode": mode,
        "nearest_expert": nearest,
        "rollout_wall_seconds": elapsed,
        "rng_protocol": "fold_in(fold_in(PRNGKey(seed), rollout_id), absolute_step)",
        "controller": "raw_joint_flowbc_then_per_agent_radial_speed_bound",
        "hard_safety_projection": False,
        "eta": False,
        "g_phi": False,
    }
    arrays = {
        "positions": positions,
        "raw_actions": raw_actions,
        "actions": actions,
        "goal_errors": goal_errors,
        "speeds": speeds,
    }
    return summary, arrays


def _teacher_forced(policy, dataset, seed: int, state_count: int = 512, samples: int = 8):
    observations, expert_actions = dataset.all_transitions()
    indices = np.linspace(0, len(observations) - 1, min(state_count, len(observations)), dtype=int)
    observations = observations[indices]
    expert = expert_actions[indices].reshape((len(indices), -1)).astype(np.float64)
    errors, bounded_samples = [], []
    root = jax.random.PRNGKey(seed)
    for index in range(samples):
        raw = np.asarray(
            policy.sample_actions(observations, jax.random.fold_in(root, index)), dtype=np.float64
        )
        bounded = _radial_bound64(raw, float(policy.config["max_speed"])).reshape(
            (len(indices), -1)
        )
        bounded_samples.append(bounded)
        errors.append(np.sqrt(np.mean(np.square(bounded - expert), axis=-1)))
    errors = np.stack(errors, axis=1)
    samples_array = np.stack(bounded_samples, axis=1)
    # Off-diagonal covariance magnitude is a simple numeric joint-output check;
    # it is not by itself evidence of useful coordination.
    flattened = samples_array.reshape((-1, 8))
    covariance = np.cov(flattened, rowvar=False)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    return {
        "states": len(indices),
        "samples_per_state": samples,
        "mean_sample_action_rmse": float(errors.mean()),
        "best_of_k_action_rmse": float(errors.min(axis=1).mean()),
        "mean_abs_off_diagonal_action_covariance": float(np.mean(np.abs(off_diagonal))),
        "sample_shape": [len(indices), samples, 4, 2],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PILOT / "best.pkl")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_PILOT / "evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if jax.default_backend() != "cpu":
        raise RuntimeError("pilot evaluation must run on CPU")
    dataset = FlowBC4ADataset(args.dataset, "val", seed=44)
    policy, checkpoint_metadata = load_checkpoint(
        args.checkpoint, dataset.environment_fingerprint
    )
    config = Config(**dataset.config)
    if environment_fingerprint(DoubleBottleneckEnv(config)) != dataset.environment_fingerprint:
        raise ValueError("runtime plant does not match the fixed checkpoint geometry")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"evaluation directory must be empty: {output}")
    rollouts_dir = output / "rollouts"
    rollouts_dir.mkdir()

    representatives = tuple(dataset.by_family[name][0] for name in dataset.family_names)
    summaries = []
    rollout_id = 0
    started = time.perf_counter()
    for episode in representatives:
        for seed in args.seeds:
            summary, arrays = _rollout(policy, dataset, episode, seed, rollout_id)
            np.savez_compressed(rollouts_dir / f"rollout_{rollout_id:03d}.npz", **arrays)
            summaries.append(summary)
            print(
                f"rollout={rollout_id} regime={episode.regime} seed={seed} "
                f"terminal={summary['termination']} steps={summary['episode_steps']} "
                f"nearest_rmse={summary['nearest_expert']['phase_aligned_position_rmse']:.4f}",
                flush=True,
            )
            rollout_id += 1

    teacher = _teacher_forced(policy, dataset, seed=915)
    count = len(summaries)
    terminations = {name: sum(row[name] for row in summaries) for name in ("success", "collision", "deadlock", "timeout")}
    expert_variations = []
    expert_paths = []
    for episode in dataset.episodes:
        expert_paths.append(float(episode.metadata["path_length"]))
        expert_variations.append(
            float(
                0.0
                if episode.length < 2
                else np.linalg.norm(np.diff(episode.actions, axis=0), axis=-1).mean()
            )
        )
    successful_signatures = sorted(
        {row["coordination_mode"]["signature"] for row in summaries if row["success"]}
    )
    aggregate = {
        "schema": "double_bottleneck_macflow_stage_i_joint_4a_raw_evaluation_v1",
        "scientific_scope": (
            "raw speed-bounded MACFlow-official Stage-I joint Flow-BC; "
            "no hard safety, eta, or G_phi"
        ),
        "backend": jax.default_backend(),
        "environment_fingerprint": dataset.environment_fingerprint,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_metadata": checkpoint_metadata,
        "rollouts": count,
        "regimes": sorted({row["regime"] for row in summaries}),
        "noise_seeds": list(args.seeds),
        "success_rate": terminations["success"] / count,
        "collision_rate": terminations["collision"] / count,
        "wall_collision_rate": sum(row["wall_collision"] for row in summaries) / count,
        "agent_collision_rate": sum(row["agent_collision"] for row in summaries) / count,
        "deadlock_rate": terminations["deadlock"] / count,
        "timeout_rate": terminations["timeout"] / count,
        "termination_counts": terminations,
        "mean_episode_steps": float(np.mean([row["episode_steps"] for row in summaries])),
        "mean_nearest_expert_phase_rmse": float(
            np.mean([row["nearest_expert"]["phase_aligned_position_rmse"] for row in summaries])
        ),
        "min_pair_surface_distance": float(
            min(row["min_swept_pair_surface_distance"] for row in summaries)
        ),
        "min_wall_clearance": float(min(row["min_swept_wall_clearance"] for row in summaries)),
        "mean_action_total_variation": float(
            np.mean([row["mean_action_total_variation"] for row in summaries])
        ),
        "expert_reference": {
            "mean_episode_steps": float(np.mean([episode.length for episode in dataset.episodes])),
            "mean_path_length": float(np.mean(expert_paths)),
            "mean_action_total_variation": float(np.mean(expert_variations)),
            "successful_mode_signatures": sorted({episode.mode_signature for episode in dataset.episodes}),
        },
        "successful_coordination_signatures": successful_signatures,
        "successful_signature_count": len(successful_signatures),
        "teacher_forced": teacher,
        "evaluation_wall_seconds": time.perf_counter() - started,
    }
    result = {"aggregate": aggregate, "rollouts": summaries}
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
