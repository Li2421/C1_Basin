"""Development-only closed-loop evaluation for Ring's local physical chart.

The policy observes/acts in instantaneous radial--tangential coordinates;
the plant still receives the deterministically decoded world velocity.  The
loader is deliberately restricted to ``dev`` nominal initial states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from new_benchmark_common.dev_closed_loop import evaluate_dev_nominal, load_dev_nominal_cases
from new_benchmark_common.macflow import load_checkpoint

from .environment import LocalFrameConfig, RingExchangeEnv
from .local_frame import local_actions_to_world, local_observation


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(dataset: str | Path, checkpoint: str | Path, output: str | Path, *,
        count: int = 30, seed: int = 3125) -> dict:
    dataset, checkpoint, output = (Path(value).resolve() for value in (dataset, checkpoint, output))
    diagnostics_root = (REPO_ROOT / "diagnostics").resolve()
    if not output.is_relative_to(diagnostics_root):
        raise ValueError("development evaluation output must remain under diagnostics/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    manifest, cases, audit = load_dev_nominal_cases(dataset, expected_count=count)
    config = LocalFrameConfig(**manifest["scenario_config"])
    if manifest["environment_fingerprint"] != config.fingerprint:
        raise ValueError("dataset manifest/config fingerprint mismatch")
    agent, checkpoint_metadata = load_checkpoint(
        checkpoint, expected_environment_fingerprint=manifest["environment_fingerprint"])
    if float(agent.config["max_speed"]) != config.max_speed:
        raise ValueError("checkpoint speed bound does not match Ring environment")

    def reset(env, initial_state):
        env.reset(np.asarray(initial_state["positions"], dtype=np.float64),
                  velocities=np.asarray(initial_state["velocities"], dtype=np.float64),
                  goals=np.asarray(initial_state["goals"], dtype=np.float64))

    def step(env, local_action):
        # Decode at the *current* position; it is a coordinate conversion,
        # not a mode switch or a hand-written controller.
        _, _, done, info = env.step(local_actions_to_world(local_action, env.positions))
        return done, info

    result = evaluate_dev_nominal(
        agent, cases, make_env=lambda: RingExchangeEnv(config), reset=reset,
        observation=lambda env: local_observation(env.positions, env.velocities, env.goals, config),
        step=step, summary=lambda env: env.summary(), max_steps=config.max_steps,
        max_speed=config.max_speed, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    for row in result["rollouts"]:
        (output / f"{row['rollout_id']}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": "ring_exchange_stage1_local_dev_closed_loop_v1",
        "scope": "development nominal only; no test rollout archive opened",
        "representation": "radial_tangential_local_action_deterministically_decoded_to_world",
        "dataset": str(dataset), "dataset_manifest_sha256": audit["manifest_sha256"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_metadata": checkpoint_metadata, "seed": seed, "dev_selection_audit": audit,
        **result,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3125)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.dataset, args.checkpoint, args.output, count=args.count, seed=args.seed)["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
