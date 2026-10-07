"""Run one Double-Bottleneck rollout with the frozen two-agent Flow checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .controller import load_frozen_pair_policy
from .environment import Config, DoubleBottleneckEnv
from .rollout import run_rollout
from .visualization import save_trajectory_svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eta", nargs=3, type=float, required=True)
    parser.add_argument(
        "--regime",
        choices=("clearly_asymmetric", "weakly_asymmetric", "near_symmetric"),
        default="weakly_asymmetric",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="engineering-smoke horizon override; omit for the frozen 850-step horizon",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("output directory must be empty")
    policy, provenance = load_frozen_pair_policy(args.checkpoint)
    result = run_rollout(
        policy,
        eta=args.eta,
        regime=args.regime,
        seed=args.seed,
        max_steps=args.max_steps,
        rollout_id=args.rollout_id,
    )
    records = result.records
    arrays = {
        "positions": result.positions,
        "eta": np.asarray(result.eta),
        "u_flow": np.stack([record.control.u_flow for record in records]),
        "u_safe": np.stack([record.control.u_safe for record in records]),
        "g_raw": np.stack([record.control.g_raw for record in records]),
        "u_exec": np.stack([record.control.u_exec for record in records]),
        "flow_key_data": np.stack([record.control.flow_key_data for record in records]),
    }
    np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
    scene = DoubleBottleneckEnv(Config(initial_regime=args.regime))
    save_trajectory_svg(
        scene,
        result.positions,
        result.eta,
        result.summary["termination"],
        args.output_dir / "trajectory.svg",
        title="Double-Bottleneck frozen Flow-BC rollout",
    )
    report = {
        "summary": result.summary,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "flow_method": provenance.get("method") if provenance else None,
        "smoke_horizon_override": args.max_steps,
        "scientific_basin_result": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
