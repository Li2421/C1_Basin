"""Render Ring Stage-I diagnostic artefacts with an explicit frozen-test gate."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from new_benchmark_common.dev_closed_loop import load_dev_nominal_cases
from new_benchmark_common.final_diagnostics import load_nominal
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions

from .environment import LocalFrameConfig, RingExchangeEnv
from .expert import circulation_signature
from .local_frame import local_actions_to_world, local_observation
from .visualization import save_failure_points_svg, save_trajectory_svg


def render(dataset: str | Path, checkpoint: str | Path, output: str | Path, *, seed: int = 3127,
           split: str = "dev", frozen_test: bool = False) -> dict:
    dataset, checkpoint, output = (Path(value).resolve() for value in (dataset, checkpoint, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if split == "test" and not frozen_test:
        raise PermissionError("test visualization requires explicit frozen_test=True")
    manifest, trajectories, selection = load_nominal(dataset, split, frozen_test=frozen_test)
    dev_audit = None
    if split == "dev":
        # A second, deliberately restrictive loader documents the invariant
        # that exactly the selected development nominal archives were read.
        _, cases, dev_audit = load_dev_nominal_cases(dataset, expected_count=len(trajectories))
        if [item.rollout_id for item in trajectories] != [item.rollout_id for item in cases]:
            raise AssertionError("dev nominal selection changed between loaders")
    if split == "dev" and selection["opened_test_archives"]:
        raise AssertionError("development visualization opened a test archive")
    if split == "test" and selection["opened_test_archives"] != len(trajectories):
        raise AssertionError("frozen-test selection audit is incomplete")
    cfg = LocalFrameConfig(**manifest["scenario_config"])
    agent, _ = load_checkpoint(checkpoint, expected_environment_fingerprint=manifest["environment_fingerprint"])
    output.mkdir(parents=True, exist_ok=True)
    policy_rows, terminal_points = [], []
    key = jax.random.PRNGKey(seed)
    for index, trajectory in enumerate(trajectories):
        env = RingExchangeEnv(cfg)
        state = trajectory.initial_state
        env.reset(np.asarray(state["positions"]), velocities=np.asarray(state["velocities"]), goals=np.asarray(state["goals"]))
        positions = [env.positions.copy()]
        episode_key = jax.random.fold_in(key, index)
        terminal = "timeout"
        for timestep in range(cfg.max_steps):
            observation = local_observation(env.positions, env.velocities, env.goals, cfg)
            action = np.asarray(sample_bounded_actions(agent, observation[None], jax.random.fold_in(episode_key, timestep))[0], dtype=np.float64)
            norms = np.linalg.norm(action, axis=-1, keepdims=True)
            action *= np.minimum(1.0, (cfg.max_speed - 1e-8) / np.maximum(norms, 1e-12))
            _, _, done, info = env.step(local_actions_to_world(action, env.positions))
            positions.append(env.positions.copy())
            terminal = str(info["termination"])
            if done:
                break
        path = np.asarray(positions)
        row = {"rollout_id": trajectory.rollout_id, "termination": terminal,
               "steps": len(path) - 1, "mode_signature": circulation_signature(path),
               "final_goal_errors": np.linalg.norm(env.goals - env.positions, axis=-1).tolist()}
        policy_rows.append(row)
        if terminal == "timeout":
            terminal_points.extend(path[-1].tolist())
        # Render two successes and two non-successes, with their matching
        # centralized expert nominal trajectory, for geometry/mode inspection.
        successes = sum(item["termination"] == "success" for item in policy_rows)
        failures = sum(item["termination"] != "success" for item in policy_rows)
        if (terminal == "success" and successes <= 2) or (terminal != "success" and failures <= 2):
            label = trajectory.rollout_id
            save_trajectory_svg(env, path, output / f"policy_{label}.svg",
                                title=f"Ring policy {split} {label}", terminal_reason=terminal)
            expert_env = RingExchangeEnv(cfg)
            expert_env.reset(np.asarray(state["positions"]), velocities=np.asarray(state["velocities"]), goals=np.asarray(state["goals"]))
            save_trajectory_svg(expert_env, trajectory.states, output / f"expert_{label}.svg",
                                title=f"Ring centralized expert {split} {label}", terminal_reason=trajectory.metadata.get("terminal_reason", "success"))
    geometry_env = RingExchangeEnv(cfg)
    geometry_env.reset(np.asarray(trajectories[0].initial_state["positions"]),
                       velocities=np.asarray(trajectories[0].initial_state["velocities"]),
                       goals=np.asarray(trajectories[0].initial_state["goals"]))
    if terminal_points:
        save_failure_points_svg(geometry_env, np.asarray(terminal_points), output / "timeout_final_positions.svg",
                                title=f"Ring v10 {split} timeout final positions")
    scope = ("development nominal only; test archive not opened" if split == "dev"
             else "master-authorized frozen test visualization only; no post-test adaptation")
    report = {"scope": scope, "selection_audit": selection,
              "dev_loader_audit": dev_audit, "policy_rollouts": policy_rows,
              "policy_mode_counts": dict(sorted(Counter(row["mode_signature"] for row in policy_rows).items())),
              "expert_mode_counts": dict(sorted(Counter(str(item.metadata.get("circulation_mode", "unknown")) for item in trajectories).items())),
              "timeout_final_point_count": len(terminal_points)}
    (output / "mode_and_visualization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3127)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--frozen-test", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(render(args.dataset, args.checkpoint, args.output, seed=args.seed,
                           split=args.split, frozen_test=args.frozen_test), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
