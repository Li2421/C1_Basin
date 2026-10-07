#!/usr/bin/env python3
"""Generate targeted Recovery-V2 rows and nearby initial-velocity trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.expert import CentralizedExpert, all_coordination_hypotheses
from double_bottleneck.expert_dataset import (
    InitialConditionSpec,
    initialize_environment,
    record_from_plan,
    save_dataset,
)
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import (
    goal_anchors,
    observation_from_state,
    query_reference_recovery,
    radial_bound,
    transition_anchors,
)
from double_bottleneck.scenario import initial_positions


SEEDS = {"target_train": 20260931, "target_val": 20260932}
POSITION_SCALE = {
    "transition": {"longitudinal": 0.025, "lateral": 0.015, "spacing": 0.020},
    "goal": {"longitudinal": 0.040, "lateral": 0.025, "spacing": 0.015},
}
VELOCITY_SCALE = {"transition": 0.080, "goal": 0.100}


def _safe(config: Config, positions: np.ndarray) -> bool:
    try:
        DoubleBottleneckEnv(config).reset(positions)
    except ValueError:
        return False
    return True


def _perturb(
    rng: np.random.Generator,
    episode,
    step: int,
    category: str,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_obs = np.asarray(episode.observations[step], dtype=np.float64)
    base_position = source_obs[:, :2]
    base_velocity = source_obs[:, 2:4]
    scale = POSITION_SCALE[category]
    delta_position = np.empty((4, 2), dtype=np.float64)
    delta_position[:, 0] = rng.uniform(-scale["longitudinal"], scale["longitudinal"], 4)
    delta_position[:, 1] = rng.uniform(-scale["lateral"], scale["lateral"], 4)
    # Explicitly change within-convoy relative spacing without translating its centre.
    travel = np.asarray((1.0, 1.0, -1.0, -1.0))
    for first, second in ((0, 1), (2, 3)):
        spacing = rng.uniform(-scale["spacing"], scale["spacing"])
        delta_position[first, 0] += 0.5 * travel[first] * spacing
        delta_position[second, 0] -= 0.5 * travel[second] * spacing
    delta_velocity = rng.uniform(-VELOCITY_SCALE[category], VELOCITY_SCALE[category], (4, 2))
    if category == "goal":
        # Independent residual travel components create asymmetric arrival timing.
        delta_velocity[:, 0] += travel * rng.uniform(-0.04, 0.04, 4)
    position = base_position + delta_position
    velocity = radial_bound(base_velocity + delta_velocity, config.max_speed)
    return position, velocity, position - base_position, velocity - base_velocity


def _generate_target_split(dataset_root: Path, split: str, output: Path) -> dict:
    data = FlowBC4ADataset(dataset_root, split, seed=0)
    config = Config(**data.config)
    rng = np.random.default_rng(SEEDS[f"target_{split}"])
    stored: dict[str, list] = {
        name: []
        for name in (
            "observations",
            "actions",
            "source_episode_index",
            "source_step",
            "source_rollout_id",
            "category",
            "source_phase",
            "event_tags",
            "position_delta",
            "velocity_delta",
            "resample_attempt",
            "recovery_steps",
            "min_wall_clearance",
            "min_pair_clearance",
            "mode_signature_match",
        )
    }
    attempted = Counter()
    rejected = Counter()
    started = time.perf_counter()
    for episode_index, episode in enumerate(data.episodes):
        selections = {
            "transition": transition_anchors(episode, config),
            "goal": goal_anchors(episode, config),
        }
        for category, anchors in selections.items():
            for anchor in anchors:
                for replicate in range(2):
                    accepted = False
                    for attempt in range(1, 51):
                        attempted[category] += 1
                        position, velocity, delta_p, delta_v = _perturb(
                            rng, episode, int(anchor["step"]), category, config
                        )
                        if not _safe(config, position):
                            rejected[f"{category}:invalid_initial_state"] += 1
                            continue
                        try:
                            recovery = query_reference_recovery(
                                episode, int(anchor["step"]), position, velocity
                            )
                        except (ValueError, FloatingPointError):
                            rejected[f"{category}:query_exception"] += 1
                            continue
                        if not recovery.success:
                            rejected[f"{category}:{recovery.terminal_reason}"] += 1
                            continue
                        if not recovery.mode_signature_match:
                            rejected[f"{category}:mode_signature_change"] += 1
                            continue
                        stored["observations"].append(
                            observation_from_state(position, velocity, data.goals)
                        )
                        stored["actions"].append(recovery.actions[0].astype(np.float32))
                        stored["source_episode_index"].append(episode_index)
                        stored["source_step"].append(int(anchor["step"]))
                        stored["source_rollout_id"].append(episode.rollout_id)
                        stored["category"].append(category)
                        stored["source_phase"].append(str(anchor["phase"]))
                        stored["event_tags"].append("|".join(anchor["events"]))
                        stored["position_delta"].append(delta_p)
                        stored["velocity_delta"].append(delta_v)
                        stored["resample_attempt"].append(attempt)
                        stored["recovery_steps"].append(recovery.recovery_steps)
                        stored["min_wall_clearance"].append(recovery.min_wall_clearance)
                        stored["min_pair_clearance"].append(recovery.min_pair_clearance)
                        stored["mode_signature_match"].append(True)
                        accepted = True
                        break
                    if not accepted:
                        raise RuntimeError(
                            f"could not recover {split}/{episode.rollout_id}/"
                            f"{category}/{anchor['step']}/{replicate}"
                        )
    arrays = {
        "observations": np.asarray(stored["observations"], dtype=np.float32),
        "actions": np.asarray(stored["actions"], dtype=np.float32),
        "source_episode_index": np.asarray(stored["source_episode_index"], dtype=np.int16),
        "source_step": np.asarray(stored["source_step"], dtype=np.int16),
        "source_rollout_id": np.asarray(stored["source_rollout_id"]),
        "category": np.asarray(stored["category"]),
        "source_phase": np.asarray(stored["source_phase"]),
        "event_tags": np.asarray(stored["event_tags"]),
        "position_delta": np.asarray(stored["position_delta"], dtype=np.float32),
        "velocity_delta": np.asarray(stored["velocity_delta"], dtype=np.float32),
        "resample_attempt": np.asarray(stored["resample_attempt"], dtype=np.int16),
        "recovery_steps": np.asarray(stored["recovery_steps"], dtype=np.int16),
        "min_wall_clearance": np.asarray(stored["min_wall_clearance"], dtype=np.float32),
        "min_pair_clearance": np.asarray(stored["min_pair_clearance"], dtype=np.float32),
        "mode_signature_match": np.asarray(stored["mode_signature_match"], dtype=bool),
        "split": np.asarray(split),
        "seed": np.asarray(SEEDS[f"target_{split}"], dtype=np.int64),
    }
    np.savez_compressed(output, **arrays)
    categories = Counter(stored["category"])
    phases = Counter(stored["source_phase"])
    directions = Counter(
        data.episodes[int(index)].metadata["hypothesis"]["first_direction"]
        for index in stored["source_episode_index"]
    )
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(stored["actions"]),
        "category_counts": dict(sorted(categories.items())),
        "source_phase_counts": dict(sorted(phases.items())),
        "first_direction_counts": dict(sorted(directions.items())),
        "attempted": dict(sorted(attempted.items())),
        "rejected": dict(sorted(rejected.items())),
        "maximum_resample_attempt": int(max(stored["resample_attempt"])),
        "minimum_recovery_wall_clearance": float(np.min(stored["min_wall_clearance"])),
        "minimum_recovery_pair_clearance": float(np.min(stored["min_pair_clearance"])),
        "maximum_recovery_steps": int(max(stored["recovery_steps"])),
        "mode_signature_matches": int(sum(stored["mode_signature_match"])),
        "generation_seconds": time.perf_counter() - started,
    }


def _velocity_specs(config: Config) -> tuple[InitialConditionSpec, ...]:
    regime = "weakly_asymmetric"
    base = initial_positions(config, regime)
    symmetry = np.asarray(
        ((0.018, 0.012), (-0.014, -0.010), (0.018, -0.012), (-0.014, 0.010))
    )
    pattern = -np.asarray(
        ((0.035, 0.010), (0.020, -0.012), (-0.030, -0.008), (-0.018, 0.010))
    )
    offsets = (
        np.asarray(((0.012, -0.006), (-0.010, 0.005), (0.011, 0.006), (-0.009, -0.005))),
        np.asarray(((-0.015, 0.007), (0.012, -0.006), (-0.013, -0.007), (0.010, 0.006))),
    )
    specs = []
    for index, (scale, offset) in enumerate(zip((0.75, 1.25), offsets, strict=True)):
        velocity = pattern * scale
        spec = InitialConditionSpec(
            condition_id=f"weakly_asymmetric__recovery_v2_velocity_negative_{scale:.2f}",
            family_id=f"weakly_asymmetric__recovery_v2_velocity_negative_{scale:.2f}",
            regime=regime,
            positions=base + symmetry + offset,
            initial_velocities=velocity,
            perturbation={
                "family": "recovery_v2_initial_velocity",
                "heldout_neighbor_not_duplicate": True,
                "velocity_scale_relative_to_heldout": scale,
                "delta_positions_from_heldout": offset.tolist(),
                "initial_velocities": velocity.tolist(),
            },
        )
        initialize_environment(config, spec)
        specs.append(spec)
    return tuple(specs)


def _generate_velocity_trajectories(output: Path) -> dict:
    config = Config()
    expert = CentralizedExpert()
    records = []
    started = time.perf_counter()
    for spec in _velocity_specs(config):
        for mode_index, hypothesis in enumerate(all_coordination_hypotheses()):
            env = initialize_environment(config, spec)
            plan = expert.plan_hypothesis(env, hypothesis)
            record = record_from_plan(
                plan,
                env,
                spec,
                f"recovery_v2_velocity__{spec.condition_id}__mode_{mode_index}",
            )
            if not record.training_eligible:
                raise RuntimeError(f"velocity expert failed: {record.rollout_id}")
            records.append(replace(record, split="train"))
    manifest = save_dataset(
        output,
        records,
        generation_metadata={
            "schema": "double_bottleneck_recovery_v2_velocity_trajectories_v1",
            "purpose": "nearby negative-sign initial-velocity families without heldout-state duplication",
            "families": len(_velocity_specs(config)),
            "hypotheses_per_family": 8,
            "generation_seconds": time.perf_counter() - started,
        },
    )
    return {
        "path": str(output.resolve()),
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "rollouts": len(records),
        "transitions": int(sum(record.episode_steps for record in records)),
        "families": sorted({record.family_id for record in records}),
        "first_direction_counts": dict(
            sorted(Counter(record.hypothesis["first_direction"] for record in records).items())
        ),
        "successes": int(sum(record.success for record in records)),
        "wall_collisions": int(sum(record.wall_collision for record in records)),
        "agent_collisions": int(sum(record.agent_collision for record in records)),
        "timeouts": int(sum(record.timeout for record in records)),
        "minimum_wall_clearance": float(
            min(record.min_swept_wall_clearance for record in records)
        ),
        "minimum_pair_clearance": float(
            min(record.min_swept_pair_surface_distance for record in records)
        ),
        "mode_analysis": manifest["mode_analysis"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("diagnostics/double_bottleneck_expert_dataset_8mode"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_recovery_v2/data"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summary = {
        "schema": "double_bottleneck_recovery_v2_targeted_dataset_v1",
        "source_dataset": str(args.dataset.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "calibration": {
            "position": "V1 K25/K50 RMSE 0.01355/0.02732 m and observed pre-collision drift",
            "velocity": "V1 K25/K50 action RMSE 0.03175/0.03868 m/s",
            "policy": "local bounded perturbations; rejection of unsafe starts and unsuccessful/mode-changing recoveries",
        },
        "perturbation": {
            "position_uniform_half_widths_m": POSITION_SCALE,
            "velocity_uniform_half_width_mps": VELOCITY_SCALE,
            "replicates_per_anchor": 2,
            "tracking_gain_per_second": 3.0,
            "seeds": SEEDS,
        },
    }
    for split in ("train", "val"):
        summary[f"targeted_{split}"] = _generate_target_split(
            args.dataset, split, output / f"targeted_{split}.npz"
        )
    summary["velocity_train"] = _generate_velocity_trajectories(
        output / "velocity_trajectories"
    )
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
