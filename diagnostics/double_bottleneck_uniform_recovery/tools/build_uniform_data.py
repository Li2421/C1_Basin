#!/usr/bin/env python3
"""Materialize preregistered uniform recovery subsets and untouched test data."""

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
from double_bottleneck.recovery_v2 import phase_labels
from double_bottleneck.scenario import INITIAL_REGIMES, initial_positions


K_BY_VARIANT = {"U-Low": 64, "U-Mid": 256, "U-High": 640}
ANCHOR_SEED_LABEL = "uniform_recovery_v1|20261001"
TEST_SEED = 20261002


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(split: str, rollout_id: str) -> int:
    digest = hashlib.sha256(
        f"{ANCHOR_SEED_LABEL}|{split}|{rollout_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _row_lookup(cache: dict[str, np.ndarray], dataset: FlowBC4ADataset) -> np.ndarray:
    lengths = [episode.length for episode in dataset.episodes]
    width = max(lengths)
    lookup = np.full((len(lengths), width), -1, dtype=np.int64)
    for row, (episode_index, step) in enumerate(
        zip(cache["source_episode_index"], cache["source_step"], strict=True)
    ):
        episode_index, step = int(episode_index), int(step)
        if lookup[episode_index, step] != -1:
            raise ValueError("cached recovery contains duplicate source transition")
        lookup[episode_index, step] = row
    for episode_index, length in enumerate(lengths):
        if np.any(lookup[episode_index, :length] < 0):
            raise ValueError("cached recovery does not cover every source transition")
        if np.any(lookup[episode_index, length:] >= 0):
            raise ValueError("cached recovery references a step beyond episode length")
    return lookup


def _variant_stats(
    dataset: FlowBC4ADataset,
    selected_rows: np.ndarray,
    selected_episodes: np.ndarray,
    selected_steps: np.ndarray,
    observations: np.ndarray,
) -> dict:
    config = Config(**dataset.config)
    source_observations = np.stack(
        [
            dataset.episodes[int(ep)].observations[int(step)]
            for ep, step in zip(selected_episodes, selected_steps, strict=True)
        ]
    )
    position_delta = observations[:, :, :2] - source_observations[:, :, :2]
    velocity_delta = observations[:, :, 2:4] - source_observations[:, :, 2:4]
    position_rms = np.sqrt(np.mean(position_delta.astype(np.float64) ** 2, axis=(1, 2)))
    velocity_rms = np.sqrt(np.mean(velocity_delta.astype(np.float64) ** 2, axis=(1, 2)))
    phases = []
    phase_cache: dict[int, np.ndarray] = {}
    wall = []
    pair = []
    regime = Counter()
    direction = Counter()
    family = Counter()
    for ep_index, step, obs in zip(
        selected_episodes, selected_steps, observations, strict=True
    ):
        ep_index = int(ep_index)
        episode = dataset.episodes[ep_index]
        if ep_index not in phase_cache:
            phase_cache[ep_index] = phase_labels(episode, config)
        phases.append(str(phase_cache[ep_index][int(step)]))
        distances = DoubleBottleneckEnv(config).distances(obs[:, :2].astype(np.float64))
        wall.append(float(np.min(distances[0])))
        pair.append(float(np.min(distances[1])))
        regime[episode.regime] += 1
        direction[str(episode.metadata["hypothesis"]["first_direction"])] += 1
        family[str(episode.metadata["perturbation"]["family"])] += 1
    return {
        "rows": int(len(selected_rows)),
        "source_phase_counts": dict(sorted(Counter(phases).items())),
        "regime_counts": dict(sorted(regime.items())),
        "first_direction_counts": dict(sorted(direction.items())),
        "initial_condition_family_counts": dict(sorted(family.items())),
        "position_delta_rms_m": _quantiles(position_rms),
        "velocity_delta_rms_mps": _quantiles(velocity_rms),
        "instantaneous_wall_clearance_m": _quantiles(np.asarray(wall)),
        "instantaneous_pair_clearance_m": _quantiles(np.asarray(pair)),
    }


def _build_split(root: Path, output: Path, split: str) -> dict:
    dataset_root = root / "diagnostics/double_bottleneck_expert_dataset_8mode"
    cache_path = (
        root
        / "diagnostics/double_bottleneck_toy_transfer_audit"
        / f"recovery_{split}.npz"
    )
    dataset = FlowBC4ADataset(dataset_root, split, seed=0)
    cache = _load_cache(cache_path)
    if bool(cache["independent_transitions"].item()) is not True:
        raise ValueError("recovery cache is not independent-transition data")
    if int(np.max(cache["resample_attempt"])) != 1:
        raise ValueError("cache contains an adaptively resampled recovery row")
    lookup = _row_lookup(cache, dataset)
    permutations = []
    for episode in dataset.episodes:
        permutations.append(
            np.random.default_rng(_seed(split, episode.rollout_id)).permutation(episode.length)
        )
    summaries = {}
    for variant, count in K_BY_VARIANT.items():
        rows, episodes, steps, ranks = [], [], [], []
        for episode_index, permutation in enumerate(permutations):
            chosen = permutation[:count]
            for rank, step in enumerate(chosen):
                rows.append(int(lookup[episode_index, int(step)]))
                episodes.append(episode_index)
                steps.append(int(step))
                ranks.append(rank)
        rows = np.asarray(rows, dtype=np.int64)
        episodes = np.asarray(episodes, dtype=np.int16)
        steps = np.asarray(steps, dtype=np.int16)
        ranks = np.asarray(ranks, dtype=np.int16)
        observations = cache["observations"][rows].astype(np.float32)
        actions = cache["actions"][rows].astype(np.float32)
        path = output / f"{variant.lower().replace('-', '_')}_{split}.npz"
        np.savez_compressed(
            path,
            observations=observations,
            actions=actions,
            source_episode_index=episodes,
            source_step=steps,
            selection_rank=ranks,
            split=np.asarray(split),
            variant=np.asarray(variant),
            anchors_per_trajectory=np.asarray(count, dtype=np.int16),
            selection_seed_label=np.asarray(ANCHOR_SEED_LABEL),
            cached_requery_sha256=np.asarray(_sha(cache_path)),
            position_noise=np.asarray(float(cache["position_noise"])),
            velocity_noise=np.asarray(float(cache["velocity_noise"])),
            tracking_gain=np.asarray(float(cache["tracking_gain"])),
            recovery_horizon=np.asarray(1, dtype=np.int16),
        )
        stats = _variant_stats(dataset, rows, episodes, steps, observations)
        stats.update(
            {
                "path": str(path.resolve()),
                "sha256": _sha(path),
                "anchors_per_trajectory": count,
                "episodes": len(dataset.episodes),
                "accepted_requeries": int(len(rows)),
                "rejected_requeries": 0,
                "resampled_requeries": 0,
                "successful_full_continuations": int(len(rows)),
            }
        )
        summaries[variant] = stats
    return summaries


def _test_specs(config: Config) -> tuple[InitialConditionSpec, ...]:
    rng = np.random.default_rng(TEST_SEED)
    specs = []
    for regime in INITIAL_REGIMES:
        for family_index in range(2):
            delta_position = np.empty((4, 2), dtype=np.float64)
            delta_position[:, 0] = rng.uniform(-0.06, 0.06, 4)
            delta_position[:, 1] = rng.uniform(-0.02, 0.02, 4)
            velocity = rng.uniform(-0.04, 0.04, (4, 2))
            condition_id = f"uniform_test__{regime}__family_{family_index}"
            spec = InitialConditionSpec(
                condition_id=condition_id,
                family_id=condition_id,
                regime=regime,
                positions=initial_positions(config, regime) + delta_position,
                initial_velocities=velocity,
                perturbation={
                    "family": "untouched_uniform_test",
                    "generation_seed": TEST_SEED,
                    "draw_index_within_regime": family_index,
                    "delta_positions": delta_position.tolist(),
                    "delta_initial_velocities": velocity.tolist(),
                },
            )
            # No retry/replacement: any invalid draw aborts construction.
            initialize_environment(config, spec)
            specs.append(spec)
    return tuple(specs)


def _build_test(output: Path, preregistration_sha256: str) -> dict:
    config = Config()
    expert = CentralizedExpert()
    records = []
    started = time.perf_counter()
    for spec in _test_specs(config):
        for mode_index, hypothesis in enumerate(all_coordination_hypotheses()):
            env = initialize_environment(config, spec)
            plan = expert.plan_hypothesis(env, hypothesis)
            record = record_from_plan(
                plan,
                env,
                spec,
                f"uniform_test_{len(records):03d}__mode_{mode_index}",
            )
            if not record.training_eligible:
                raise RuntimeError(
                    f"untouched test expert construction failed without replacement: "
                    f"{spec.condition_id}/{hypothesis.label}/{record.terminal_reason}"
                )
            # The shared trajectory format accepts only train/val labels.  This
            # is an isolated dataset, so "val" is a storage label only; the
            # generation manifest below records its scientific role as test.
            records.append(replace(record, split="val"))
    manifest = save_dataset(
        output,
        records,
        generation_metadata={
            "schema": "double_bottleneck_uniform_untouched_test_v1",
            "preregistration_sha256": preregistration_sha256,
            "generation_seed": TEST_SEED,
            "families_per_regime": 2,
            "hypotheses_per_family": 8,
            "adaptive_redraw": False,
            "model_outcomes_inspected_during_generation": False,
            "scientific_split": "untouched_test",
            "storage_split_label": "val",
            "generation_seconds": time.perf_counter() - started,
        },
    )
    return {
        "path": str(output.resolve()),
        "manifest_sha256": _sha(output / "manifest.json"),
        "families": 6,
        "rollouts": 48,
        "transitions": int(sum(record.length for record in records)),
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
        "initial_states": [
            {
                "condition_id": spec.condition_id,
                "regime": spec.regime,
                "positions": spec.positions.tolist(),
                "initial_velocities": spec.initial_velocities.tolist(),
            }
            for spec in _test_specs(config)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_uniform_recovery/data"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    preregistration = root / "diagnostics/double_bottleneck_uniform_recovery/PREREGISTRATION.json"
    preregistration_sha256 = _sha(preregistration)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "double_bottleneck_uniform_recovery_dataset_manifest_v1",
        "preregistration": str(preregistration.resolve()),
        "preregistration_sha256": preregistration_sha256,
        "selection_rule": ANCHOR_SEED_LABEL,
        "densities": K_BY_VARIANT,
        "splits": {},
    }
    for split in ("train", "val"):
        summary["splits"][split] = _build_split(root, output, split)
    summary["untouched_test"] = _build_test(
        output / "untouched_test_dataset", preregistration_sha256
    )
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
