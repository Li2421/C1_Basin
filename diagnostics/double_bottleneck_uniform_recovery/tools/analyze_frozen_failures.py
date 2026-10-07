#!/usr/bin/env python3
"""Post-hoc analysis of already frozen uniform-recovery test failures."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from double_bottleneck.environment import Config
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import phase_labels


SUPPORT_THRESHOLD = 0.039569792891474595
POSITION_DIVERGENCE_M = 0.08
VARIANTS = ("U-Low", "U-Mid", "U-High")


def _expert_manifold(dataset: FlowBC4ADataset, family: str):
    points, episode_index, step_index, phases = [], [], [], []
    config = Config(**dataset.config)
    episodes = dataset.by_family[family]
    for ep_index, episode in enumerate(episodes):
        labels = phase_labels(episode, config)
        points.append(episode.positions.reshape((len(episode.positions), -1)))
        episode_index.extend([ep_index] * len(episode.positions))
        step_index.extend(range(len(episode.positions)))
        phases.extend(labels.tolist())
    points = np.concatenate(points, axis=0)
    # Euclidean distance after division by sqrt(8) is joint-position RMS.
    return (
        cKDTree(points / np.sqrt(points.shape[1])),
        np.asarray(episode_index, dtype=np.int16),
        np.asarray(step_index, dtype=np.int16),
        np.asarray(phases),
    )


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    evaluation = root / "diagnostics/double_bottleneck_uniform_recovery/evaluation"
    if not (evaluation / "comparison.json").is_file():
        raise FileNotFoundError("frozen comparison must exist before failure inspection")
    dataset = FlowBC4ADataset(
        root
        / "diagnostics/double_bottleneck_uniform_recovery/data/untouched_test_dataset",
        "val",
        seed=0,
    )
    manifolds = {
        family: _expert_manifold(dataset, family) for family in dataset.family_names
    }
    result = {
        "schema": "double_bottleneck_uniform_recovery_posthoc_failures_v1",
        "analysis_started_after_frozen_comparison": True,
        "new_recovery_queries_or_data": 0,
        "position_divergence_threshold_m": POSITION_DIVERGENCE_M,
        "support_threshold": SUPPORT_THRESHOLD,
        "variants": {},
    }
    for variant in VARIANTS:
        model_result = json.loads(
            (evaluation / "untouched_test" / f"{variant.lower()}.json").read_text()
        )
        rows = []
        for rollout_id, rollout in enumerate(model_result["rollouts"]):
            if rollout["success"]:
                continue
            trajectory_path = (
                evaluation
                / "untouched_test"
                / "trajectories"
                / f"{variant.lower()}_{rollout_id:03d}.npz"
            )
            with np.load(trajectory_path, allow_pickle=False) as archive:
                positions = archive["positions"].astype(np.float64)
                support = archive["support_distances"].astype(np.float64)
            tree, expert_ep, expert_step, phases = manifolds[rollout["family_id"]]
            distance, nearest = tree.query(
                positions.reshape((len(positions), -1)) / np.sqrt(8.0), workers=1
            )
            clear = np.flatnonzero(distance > POSITION_DIVERGENCE_M)
            first_clear = int(clear[0]) if len(clear) else None
            terminal_index = len(positions) - 1
            terminal_nearest = int(nearest[terminal_index])
            row = {
                "rollout_id": rollout_id,
                "family_id": rollout["family_id"],
                "regime": rollout["regime"],
                "seed": rollout["seed"],
                "termination": rollout["termination"],
                "episode_steps": rollout["episode_steps"],
                "wall_collision": rollout["wall_collision"],
                "agent_collision": rollout["agent_collision"],
                "timeout": rollout["timeout"],
                "first_outside_support_step": rollout["first_outside_support_step"],
                "first_clear_divergence_step": first_clear,
                "terminal_expert_manifold_position_rms": float(distance[terminal_index]),
                "terminal_nearest_phase": str(phases[terminal_nearest]),
            }
            if first_clear is not None:
                nearest_index = int(nearest[first_clear])
                row["divergence"] = {
                    "position_rms_m": float(distance[first_clear]),
                    "nearest_expert_episode_index": int(expert_ep[nearest_index]),
                    "nearest_expert_step": int(expert_step[nearest_index]),
                    "nearest_expert_phase": str(phases[nearest_index]),
                    "training_support_distance": float(support[first_clear]),
                    "inside_fixed_support": bool(
                        support[first_clear] <= SUPPORT_THRESHOLD
                    ),
                }
            rows.append(row)
        phases = Counter(
            row["divergence"]["nearest_expert_phase"]
            for row in rows
            if "divergence" in row
        )
        result["variants"][variant] = {
            "failures": len(rows),
            "termination_counts": dict(
                sorted(Counter(row["termination"] for row in rows).items())
            ),
            "failures_by_regime": dict(
                sorted(Counter(row["regime"] for row in rows).items())
            ),
            "failures_with_clear_divergence": sum(
                "divergence" in row for row in rows
            ),
            "clear_divergence_inside_support": sum(
                row.get("divergence", {}).get("inside_fixed_support", False)
                for row in rows
            ),
            "clear_divergence_phase_counts": dict(sorted(phases.items())),
            "rows": rows,
        }
    output = root / "diagnostics/double_bottleneck_uniform_recovery/failure_analysis.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                variant: {
                    key: value[key]
                    for key in (
                        "failures",
                        "failures_with_clear_divergence",
                        "clear_divergence_inside_support",
                        "clear_divergence_phase_counts",
                    )
                }
                for variant, value in result["variants"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
