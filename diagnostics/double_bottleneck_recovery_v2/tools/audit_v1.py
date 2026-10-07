#!/usr/bin/env python3
"""Audit exact Recovery-V1 state coverage before any V2 generation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset


PHASES = (
    "initial_approach",
    "first_bottleneck_approach",
    "waiting_yielding",
    "coordination_mode_transition",
    "bottleneck_traversal",
    "chamber_traversal",
    "second_bottleneck",
    "final_goal_approach",
    "near_goal_termination",
)


def _speed(actions: np.ndarray, length: int) -> np.ndarray:
    speed = np.linalg.norm(actions, axis=-1)
    if len(speed) < length:
        speed = np.concatenate((speed, speed[-1:]), axis=0)
    return speed[:length]


def transition_mask(episode, radius: int = 3) -> np.ndarray:
    """Activity/direction-change neighborhoods, independent of planner labels."""
    actions = np.asarray(episode.actions, dtype=np.float64)
    speed = _speed(actions, len(episode.positions))
    active = speed > 0.025
    event = np.zeros(len(episode.positions), dtype=bool)
    event[1:] |= np.any(active[1:] != active[:-1], axis=1)
    if len(actions) > 1:
        delta = np.linalg.norm(actions[1:] - actions[:-1], axis=-1)
        event[1 : len(actions)] |= np.any(delta > 0.08, axis=1)
    indices = np.flatnonzero(event)
    expanded = np.zeros_like(event)
    for index in indices:
        expanded[max(0, index - radius) : min(len(event), index + radius + 1)] = True
    return expanded


def phase_labels(episode, config: Config) -> np.ndarray:
    positions = np.asarray(episode.positions, dtype=np.float64)
    actions = np.asarray(episode.actions, dtype=np.float64)
    goals = np.asarray(episode.metadata["environment"]["goals"], dtype=np.float64)
    length = len(positions)
    speed = _speed(actions, length)
    active = speed > 0.025
    goal_error = np.linalg.norm(goals[None] - positions, axis=-1)
    transition = transition_mask(episode)
    outer_gate = config.chamber_half_length + config.bottleneck_length
    inner = config.chamber_half_length
    first_direction = str(episode.metadata["hypothesis"]["first_direction"])
    travel_sign = np.asarray((1.0, 1.0, -1.0, -1.0))
    labels = []
    for t, point in enumerate(positions):
        # Goal buckets take priority so the terminal field is separately auditable.
        unfinished = goal_error[t] > config.goal_tolerance
        if t >= length - 21 or np.max(goal_error[t]) <= 0.20:
            labels.append("near_goal_termination")
            continue
        goal_side_outer = travel_sign * point[:, 0] >= outer_gate
        if np.any(unfinished & goal_side_outer & (goal_error[t] <= 0.75)):
            labels.append("final_goal_approach")
            continue
        if transition[t]:
            labels.append("coordination_mode_transition")
            continue
        waiting = (~active[t]) & unfinished & (goal_error[t] > 0.20)
        if np.any(waiting) and np.any(active[t]):
            labels.append("waiting_yielding")
            continue
        in_left_gate = (point[:, 0] >= -outer_gate) & (point[:, 0] <= -inner)
        in_right_gate = (point[:, 0] >= inner) & (point[:, 0] <= outer_gate)
        destination_gate = np.where(travel_sign > 0, in_right_gate, in_left_gate)
        origin_gate = np.where(travel_sign > 0, in_left_gate, in_right_gate)
        if np.any(destination_gate):
            labels.append("second_bottleneck")
            continue
        if np.any(origin_gate):
            labels.append("bottleneck_traversal")
            continue
        in_chamber = np.abs(point[:, 0]) < inner
        if np.any(in_chamber & active[t]):
            labels.append("chamber_traversal")
            continue
        toward_origin_gate = (
            (travel_sign * point[:, 0] < -inner)
            & (travel_sign * point[:, 0] >= -outer_gate - 0.60)
            & active[t]
        )
        if np.any(toward_origin_gate):
            labels.append("first_bottleneck_approach")
            continue
        labels.append("initial_approach")
    result = np.asarray(labels)
    if set(result) - set(PHASES):
        raise AssertionError("unknown phase")
    return result


def _percentiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def audit_split(dataset_root: Path, recovery_path: Path, split: str):
    data = FlowBC4ADataset(dataset_root, split, seed=0)
    recovery = np.load(recovery_path, allow_pickle=False)
    if len(recovery["observations"]) != len(data):
        raise ValueError("V1 recovery/source row count mismatch")
    config = Config(**data.config)
    env = DoubleBottleneckEnv(config)
    cached_labels = [phase_labels(ep, config) for ep in data.episodes]
    phase_count = Counter()
    family_count = Counter()
    direction_count = Counter()
    per_phase = defaultdict(lambda: defaultdict(list))
    all_stats = defaultdict(list)
    rows = []
    for row_index, (observation, episode_index, source_step) in enumerate(
        zip(
            recovery["observations"],
            recovery["source_episode_index"],
            recovery["source_step"],
            strict=True,
        )
    ):
        episode = data.episodes[int(episode_index)]
        step = int(source_step)
        phase = str(cached_labels[int(episode_index)][step])
        perturbed_position = np.asarray(observation[:, :2], dtype=np.float64)
        perturbed_velocity = np.asarray(observation[:, 2:4], dtype=np.float64)
        source_observation = np.asarray(episode.observations[step], dtype=np.float64)
        position_delta = perturbed_position - source_observation[:, :2]
        velocity_delta = perturbed_velocity - source_observation[:, 2:4]
        wall, pair = env.distances(perturbed_position)
        pos_rms = float(np.sqrt(np.mean(position_delta**2)))
        pos_max = float(np.max(np.abs(position_delta)))
        vel_rms = float(np.sqrt(np.mean(velocity_delta**2)))
        vel_max = float(np.max(np.abs(velocity_delta)))
        wall_min = float(wall.min())
        pair_min = float(pair.min())
        phase_count[phase] += 1
        family = episode.family_id.split("__", 1)[1]
        family_count[family] += 1
        direction = str(episode.metadata["hypothesis"]["first_direction"])
        direction_count[direction] += 1
        for name, value in (
            ("position_rms", pos_rms),
            ("position_max_abs", pos_max),
            ("velocity_rms", vel_rms),
            ("velocity_max_abs", vel_max),
            ("wall_clearance", wall_min),
            ("pair_clearance", pair_min),
        ):
            per_phase[phase][name].append(value)
            all_stats[name].append(value)
        rows.append(
            {
                "split": split,
                "row": row_index,
                "source_rollout_id": episode.rollout_id,
                "source_step": step,
                "phase": phase,
                "family_id": episode.family_id,
                "regime": episode.regime,
                "first_direction": direction,
                "position_rms": pos_rms,
                "position_max_abs": pos_max,
                "velocity_rms": vel_rms,
                "velocity_max_abs": vel_max,
                "wall_clearance": wall_min,
                "pair_clearance": pair_min,
            }
        )
    return {
        "split": split,
        "rows": len(rows),
        "phase_counts": {phase: int(phase_count[phase]) for phase in PHASES},
        "phase_fractions": {
            phase: float(phase_count[phase] / len(rows)) for phase in PHASES
        },
        "initial_velocity_family_counts": dict(sorted(family_count.items())),
        "first_direction_counts": dict(sorted(direction_count.items())),
        "overall": {name: _percentiles(value) for name, value in all_stats.items()},
        "by_phase": {
            phase: {name: _percentiles(value) for name, value in metrics.items()}
            for phase, metrics in per_phase.items()
        },
    }, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("diagnostics/double_bottleneck_expert_dataset_8mode"),
    )
    parser.add_argument(
        "--v1-root",
        type=Path,
        default=Path("diagnostics/double_bottleneck_toy_transfer_audit"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_recovery_v2/v1_coverage_audit.json"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    summaries = {}
    rows = []
    for split in ("train", "val"):
        summary, split_rows = audit_split(
            args.dataset, args.v1_root / f"recovery_{split}.npz", split
        )
        summaries[split] = summary
        rows.extend(split_rows)
    v1_summary = json.loads((args.v1_root / "dataset_summary.json").read_text())
    result = {
        "schema": "double_bottleneck_recovery_v1_coverage_audit_v1",
        "phase_protocol": {
            "exclusive_precedence": list(reversed(PHASES)),
            "transition": "within +/-3 steps of any 0.025 m/s activity change or >0.08 m/s joint action-vector change",
            "near_goal": "last 20 source states or maximum agent goal error <=0.20 m",
            "goal_approach": "unfinished agent on destination-side outer room within 0.75 m of goal",
            "waiting": "at least one unfinished stationary agent and one moving agent",
            "geometry": "origin gate=first bottleneck, destination gate=second bottleneck",
        },
        "source_manifest_sha256": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "recovery_files": {
            split: {
                "path": str((args.v1_root / f"recovery_{split}.npz").resolve()),
                "sha256": hashlib.sha256(
                    (args.v1_root / f"recovery_{split}.npz").read_bytes()
                ).hexdigest(),
            }
            for split in ("train", "val")
        },
        "recovery_validation": {
            split: v1_summary[split]["recovery_validation"] for split in ("train", "val")
        },
        "splits": summaries,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({s: summaries[s]["phase_counts"] for s in summaries}, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
