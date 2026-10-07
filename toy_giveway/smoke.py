"""Small, checkpoint-free Toy Give-Way integration smoke test."""

from __future__ import annotations

import argparse
import json

import numpy as np

from toy_giveway.environment import GiveWayEnv, bounded_nominal
from toy_giveway.expert import Expert
from toy_giveway.safety import CBFSafetyFilter


def run_smoke(steps: int = 16) -> dict:
    """Run a short expert + hard-projection rollout without writing files."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    env = GiveWayEnv()
    expert = Expert(env, mode=0)
    safety = CBFSafetyFilter()
    statuses = []
    info = None
    for _ in range(steps):
        nominal = bounded_nominal(expert.action(), env.config.max_speed)
        filtered = safety(env.snapshot(), nominal)
        statuses.append(filtered.status)
        _, _, done, info = env.step(filtered.velocity)
        if done:
            break
    assert info is not None
    return {
        "requested_steps": steps,
        "executed_steps": env.step_count,
        "observation_shape": list(env.observation().shape),
        "action_shape": list(env.velocities.shape),
        "termination": info["termination"],
        "wall_collision": bool(info["wall_collision"]),
        "agent_collision": bool(info["agent_collision"]),
        "max_speed": float(np.linalg.norm(env.velocities, axis=-1).max()),
        "max_integration_residual": float(info["integration_residual"]),
        "projection_statuses": sorted(set(statuses)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
