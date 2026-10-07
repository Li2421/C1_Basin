"""Frozen development-only Four-Way Stage-I closed-loop evaluation.

Only manifest entries tagged ``dev`` and ``nominal`` are opened.  In
particular, test archive paths are neither loaded nor used for adaptation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from new_benchmark_common.dev_closed_loop import evaluate_dev_nominal, load_dev_nominal_cases
from new_benchmark_common.macflow import load_checkpoint

from .environment import Config, FourWayIntersectionEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "diagnostics/four_way_intersection_stage1/base_u_plus_w_v5_outer_buffer_dense_dataset"
DEFAULT_CHECKPOINT = REPO_ROOT / "diagnostics/four_way_intersection_stage1/base_u_plus_w_v5_outer_buffer_dense_macflow/best.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "diagnostics/four_way_intersection_stage1/base_u_plus_w_v5_outer_buffer_dense_macflow/dev_closed_loop_20"


def run(dataset: str | Path = DEFAULT_DATASET, checkpoint: str | Path = DEFAULT_CHECKPOINT,
        output: str | Path = DEFAULT_OUTPUT, *, count: int = 20, seed: int = 912) -> dict:
    dataset, checkpoint, output = Path(dataset).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    diagnostics_root = (REPO_ROOT / "diagnostics").resolve()
    if not output.is_relative_to(diagnostics_root):
        raise ValueError("development evaluation output must remain under diagnostics/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    manifest, cases, audit = load_dev_nominal_cases(dataset, expected_count=count)
    # This explicit construction is part of the frozen dense-v5 evaluation
    # contract, rather than accepting a potentially accidental default world.
    config = Config(world_half_extent=4.75)
    if manifest["scenario_config"] != config.to_dict() or manifest["environment_fingerprint"] != config.fingerprint:
        raise ValueError("dataset is not the frozen Four-Way dense-v5 world_half_extent=4.75 configuration")
    agent, checkpoint_metadata = load_checkpoint(
        checkpoint, expected_environment_fingerprint=manifest["environment_fingerprint"]
    )
    if float(agent.config["max_speed"]) != config.max_speed:
        raise ValueError("checkpoint speed bound does not match Four-Way environment")

    def reset(env, initial_state):
        env.reset(np.asarray(initial_state["positions"], dtype=np.float64),
                  np.asarray(initial_state["velocities"], dtype=np.float64))

    def step(env, action):
        _, _, done, info = env.step(action)
        return done, info

    result = evaluate_dev_nominal(
        agent, cases, make_env=lambda: FourWayIntersectionEnv(config), reset=reset,
        observation=lambda env: env.observation(), step=step, summary=lambda env: env.summary(),
        max_steps=config.max_steps, max_speed=config.max_speed, seed=seed,
    )
    # Four-Way has outer walls but no central obstacle. Keep a zero-valued
    # field in the common schema and state this explicitly for comparisons.
    result["aggregate"]["obstacle_collision"] = 0.0
    output.mkdir(parents=True, exist_ok=True)
    for row in result["rollouts"]:
        (output / f"{row['rollout_id']}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": "four_way_intersection_stage1_dev_closed_loop_v1",
        "scope": "development nominal only; no test rollout archive opened",
        "frozen_config": config.to_dict(), "dataset": str(dataset),
        "dataset_manifest_sha256": audit["manifest_sha256"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_metadata": checkpoint_metadata, "seed": seed, "dev_selection_audit": audit,
        **result,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=912)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.checkpoint, args.output, count=args.count, seed=args.seed)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
