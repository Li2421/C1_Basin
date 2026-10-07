"""Checkpoint-free Double-Bottleneck controller/rollout plumbing smoke test.

The deterministic pair policy below exists only to exercise tensor, projection,
event, logging, and visualization interfaces.  It is not a replacement for the
frozen Flow-BC checkpoint and no outcome from it is a scientific basin result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .rollout import run_rollout
from .visualization import save_trajectory_svg


class PlumbingGoalPairPolicy:
    """Minimal two-agent ``sample_actions`` interface for engineering tests."""

    def sample_actions(self, observations, seed=None):
        del seed
        values = np.asarray(observations, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (2, 10):
            raise ValueError("expected legacy pair observations [B,2,10]")
        goal_delta = values[..., 4:6]
        norm = np.linalg.norm(goal_delta, axis=-1, keepdims=True)
        return 0.35 * goal_delta / np.maximum(norm, 1e-30)


def run_smoke(
    eta=(0.0, 0.0, 0.0),
    regime: str = "weakly_asymmetric",
    seed: int = 0,
    steps: int = 12,
    output: str | Path | None = None,
) -> dict:
    """Run a tiny fixed-eta episode and optionally emit a static SVG."""
    result = run_rollout(
        PlumbingGoalPairPolicy(),
        eta=eta,
        regime=regime,
        seed=seed,
        max_steps=steps,
    )
    if output is not None:
        from .environment import Config, DoubleBottleneckEnv

        scene = DoubleBottleneckEnv(Config(initial_regime=regime))
        save_trajectory_svg(
            scene,
            result.positions,
            result.eta,
            result.summary["termination"],
            output,
            title="Double-Bottleneck engineering smoke (plumbing-only pair policy)",
        )
    max_speed = max(
        float(np.linalg.norm(record.control.u_exec, axis=-1).max()) for record in result.records
    )
    max_violation = max(
        float(record.control.second_projection["max_constraint_violation"])
        for record in result.records
    )
    return {
        "scientific_policy": False,
        "policy_label": "plumbing_only_deterministic_pair_policy",
        "eta": list(result.eta),
        "regime": result.regime,
        "seed": result.seed,
        "executed_steps": len(result.records),
        "positions_shape": list(result.positions.shape),
        "action_shape": list(result.records[0].control.u_exec.shape),
        "termination": result.summary["termination"],
        "max_executed_speed": max_speed,
        "max_second_projection_constraint_violation": max_violation,
        "both_projections_observed": all(
            record.control.first_projection["accepted"]
            and record.control.second_projection["accepted"]
            for record in result.records
        ),
        "svg": None if output is None else str(Path(output)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--regime", default="weakly_asymmetric")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(args.eta, args.regime, args.seed, args.steps, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ("PlumbingGoalPairPolicy", "run_smoke")
