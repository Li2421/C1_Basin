#!/usr/bin/env python3
"""Post-hoc S-XL failure analysis after both test results are frozen."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.spatial import cKDTree

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import phase_labels, query_reference_recovery


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
SUPPORT_THRESHOLD = 0.039569792891474595
DIV_THRESHOLD_M = 0.08
DIV_PERSISTENCE = 5


def _first_persistent(mask: np.ndarray, length: int):
    if len(mask) < length:
        return None
    hits = np.convolve(mask.astype(np.int8), np.ones(length, dtype=np.int8), mode="valid")
    found = np.flatnonzero(hits == length)
    return int(found[0]) if len(found) else None


def _family_manifold(dataset, family, config):
    points, mapping, phases = [], [], []
    for episode_index, episode in enumerate(dataset.by_family[family]):
        labels = phase_labels(episode, config)
        positions = np.asarray(episode.positions, dtype=np.float64)
        points.append(positions.reshape((len(positions), 8)))
        mapping.extend((episode_index, step) for step in range(len(positions)))
        phases.extend(str(label) for label in labels)
    return cKDTree(np.concatenate(points) / np.sqrt(8.0)), mapping, np.asarray(phases)


def _valid_probe(config, regime, positions, observations, preferred):
    for step in range(min(preferred, len(positions) - 1), -1, -1):
        try:
            env = DoubleBottleneckEnv(config)
            env.reset(positions[step], regime=regime)
            state = env.augmented_state()
            state["last_applied_velocity"] = observations[step, :, 2:4].astype(np.float64)
            env.restore_augmented_state(state)
            return step, env
        except ValueError:
            continue
    raise RuntimeError("no valid probe state in failed rollout")


def _expert_recovery(episode, source_step, positions, velocity):
    try:
        result = query_reference_recovery(
            episode, source_step, positions, velocity
        )
        return {
            "success": bool(result.success),
            "terminal_reason": str(result.terminal_reason),
            "episode_steps": int(result.recovery_steps),
            "mode_signature_match": bool(result.mode_signature_match),
            "min_wall_clearance": float(result.min_wall_clearance),
            "min_pair_clearance": float(result.min_pair_clearance),
        }
    except (ValueError, RuntimeError) as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


def _category(row, phase, clear_step):
    if row["wall_collision"]:
        return "wall_collision"
    if row["agent_collision"]:
        return "agent_collision"
    if phase in ("waiting_yielding", "coordination_mode_transition"):
        return "waiting_or_coordination_failure"
    if phase in ("final_goal_approach", "near_goal_termination"):
        return "terminal_or_goal_failure"
    if clear_step is not None:
        return "long_horizon_drift"
    if row["timeout"]:
        return "timeout"
    return "other"


def _analyze_set(name, dataset_path):
    dataset = FlowBC4ADataset(dataset_path, "val", seed=91)
    config = Config(**dataset.config)
    frozen_path = STUDY / "evaluation" / name / "s-xl.json"
    frozen = json.loads(frozen_path.read_text())
    tree_cache = {
        family: _family_manifold(dataset, family, config)
        for family in dataset.family_names
    }
    rows = []
    for summary in frozen["rollouts"]:
        if summary["success"]:
            continue
        rollout_id = int(summary["rollout_id"])
        trajectory_path = STUDY / "evaluation" / name / "trajectories" / f"s-xl_{rollout_id:03d}.npz"
        with np.load(trajectory_path, allow_pickle=False) as archive:
            positions = archive["positions"].astype(np.float64)
            observations = archive["observations"].astype(np.float32)
            support = archive["support_distances"].astype(np.float64)
        tree, mapping, phases = tree_cache[summary["family_id"]]
        distances, nearest = tree.query(
            positions.reshape((len(positions), 8)) / np.sqrt(8.0), workers=1
        )
        clear_step = _first_persistent(distances > DIV_THRESHOLD_M, DIV_PERSISTENCE)
        preferred = clear_step if clear_step is not None else len(positions) - 2
        probe_step, _ = _valid_probe(
            config, summary["regime"], positions, observations, preferred
        )
        nearest_episode, nearest_step = mapping[int(nearest[probe_step])]
        phase = str(phases[int(nearest[probe_step])])
        reference_episode = dataset.by_family[summary["family_id"]][nearest_episode]
        recovery = _expert_recovery(
            reference_episode,
            nearest_step,
            positions[probe_step],
            observations[probe_step, :, 2:4].astype(np.float64),
        )
        rows.append(
            {
                "rollout_id": rollout_id,
                "family_id": summary["family_id"],
                "regime": summary["regime"],
                "seed": int(summary["seed"]),
                "termination": summary["termination"],
                "wall_collision": bool(summary["wall_collision"]),
                "agent_collision": bool(summary["agent_collision"]),
                "timeout": bool(summary["timeout"]),
                "episode_steps": int(summary["episode_steps"]),
                "first_clear_divergence_step": clear_step,
                "first_clear_divergence_phase": (
                    str(phases[int(nearest[clear_step])]) if clear_step is not None else None
                ),
                "support_distance_at_divergence": (
                    float(support[clear_step]) if clear_step is not None else None
                ),
                "inside_support_at_divergence": (
                    bool(support[clear_step] <= SUPPORT_THRESHOLD)
                    if clear_step is not None else None
                ),
                "probe_step": int(probe_step),
                "probe_phase": phase,
                "probe_position_rms_to_same_family_expert_m": float(distances[probe_step]),
                "probe_support_distance": float(support[probe_step]),
                "probe_inside_strict_support": bool(support[probe_step] <= SUPPORT_THRESHOLD),
                "nearest_expert_rollout_id": dataset.by_family[summary["family_id"]][nearest_episode].rollout_id,
                "nearest_expert_step": int(nearest_step),
                "expert_recovery": recovery,
                "primary_category": _category(summary, phase, clear_step),
            }
        )
    categories = Counter(row["primary_category"] for row in rows)
    div_rows = [row for row in rows if row["first_clear_divergence_step"] is not None]
    summary = {
        "failures": len(rows),
        "primary_category_counts": dict(sorted(categories.items())),
        "termination_counts": dict(sorted(Counter(row["termination"] for row in rows).items())),
        "clear_divergences": len(div_rows),
        "median_first_clear_divergence_step": (
            float(np.median([row["first_clear_divergence_step"] for row in div_rows]))
            if div_rows else None
        ),
        "clear_divergence_phase_counts": dict(sorted(Counter(row["first_clear_divergence_phase"] for row in div_rows).items())),
        "clear_divergence_inside_support": sum(bool(row["inside_support_at_divergence"]) for row in div_rows),
        "probe_inside_strict_support": sum(row["probe_inside_strict_support"] for row in rows),
        "expert_recoverable_probes": sum(bool(row["expert_recovery"]["success"]) for row in rows),
        "expert_recovery_queries": len(rows),
        "data_rows_generated": 0,
        "training_or_checkpoint_changed": False,
    }
    return {"summary": summary, "rows": rows}


def main():
    output = STUDY / "posthoc_failure_analysis.json"
    if output.exists():
        raise FileExistsError(output)
    result = {
        "schema": "double_bottleneck_sxl_posthoc_failure_analysis_v1",
        "frozen_results_read_before_analysis": True,
        "support_threshold": SUPPORT_THRESHOLD,
        "divergence_definition": {
            "distance": "joint-position RMS to any state in all 8 same-initial-state expert trajectories",
            "threshold_m": DIV_THRESHOLD_M,
            "persistence_steps": DIV_PERSISTENCE,
        },
        "expert_probe": "one deterministic valid state per failed rollout; scenario-local recovery tracks the nearest same-family normal expert continuation; diagnostic only",
        "sets": {
            "existing_untouched_test": _analyze_set(
                "existing_untouched_test",
                ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool",
            ),
            "fresh_untouched_test": _analyze_set(
                "fresh_untouched_test", STUDY / "data/fresh_test_pool"
            ),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["summary"] for key, value in result["sets"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
