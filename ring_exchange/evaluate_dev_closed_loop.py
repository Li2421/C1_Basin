"""Frozen development-only Ring Exchange closed-loop Stage-I evaluation.

It accepts only dev nominal initial states.  Test rollout archives are neither
loaded nor used to choose states, collect data, or change the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from new_benchmark_common.dev_closed_loop import evaluate_dev_nominal, load_dev_nominal_cases
from new_benchmark_common.macflow import load_checkpoint

from .environment import Config, RingExchangeEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "diagnostics/ring_exchange_stage1/base_u_plus_w_v5_dataset"
DEFAULT_CHECKPOINT = REPO_ROOT / "diagnostics/ring_exchange_stage1/base_u_plus_w_v5_macflow/best.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "diagnostics/ring_exchange_stage1/base_u_plus_w_v5_macflow/dev_closed_loop_25"


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run(dataset: str | Path = DEFAULT_DATASET, checkpoint: str | Path = DEFAULT_CHECKPOINT,
        output: str | Path = DEFAULT_OUTPUT, *, count: int = 25, seed: int = 911) -> dict:
    dataset, checkpoint, output = Path(dataset).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    diagnostics_root = (REPO_ROOT / "diagnostics").resolve()
    if not output.is_relative_to(diagnostics_root):
        raise ValueError("development evaluation output must remain under diagnostics/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    manifest, cases, audit = load_dev_nominal_cases(dataset, expected_count=count)
    config = Config(**manifest["scenario_config"])
    if manifest["environment_fingerprint"] != config.fingerprint:
        raise ValueError("dataset manifest/config fingerprint mismatch")
    agent, checkpoint_metadata = load_checkpoint(
        checkpoint, expected_environment_fingerprint=manifest["environment_fingerprint"]
    )
    if float(agent.config["max_speed"]) != config.max_speed:
        raise ValueError("checkpoint speed bound does not match Ring environment")

    def reset(env, initial_state):
        env.reset(np.asarray(initial_state["positions"], dtype=np.float64),
                  velocities=np.asarray(initial_state["velocities"], dtype=np.float64),
                  goals=np.asarray(initial_state["goals"], dtype=np.float64))

    def step(env, action):
        _, _, done, info = env.step(action)
        return done, info

    result = evaluate_dev_nominal(
        agent, cases, make_env=lambda: RingExchangeEnv(config), reset=reset,
        observation=lambda env: env.observation(), step=step, summary=lambda env: env.summary(),
        max_steps=config.max_steps, max_speed=config.max_speed, seed=seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    for row in result["rollouts"]:
        (output / f"{row['rollout_id']}.json").write_text(json.dumps(row, indent=2, sort_keys=True, default=_jsonable) + "\n")
    report = {
        "schema": "ring_exchange_stage1_dev_closed_loop_v1",
        "scope": "development nominal only; no test rollout archive opened",
        "dataset": str(dataset), "dataset_manifest_sha256": audit["manifest_sha256"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_metadata": checkpoint_metadata, "seed": seed, "dev_selection_audit": audit,
        **result,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=_jsonable) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=911)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.checkpoint, args.output, count=args.count, seed=args.seed)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
