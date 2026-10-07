#!/usr/bin/env python3
"""Build the preregistered nested, globally uniform S-XL-128 recovery data."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from diagnostics.double_bottleneck_initial_state_coverage.tools.generate_pools import (
    ROOT_LABEL,
    _quantiles,
    _seed,
)
from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import (
    observation_from_state,
    phase_labels,
    query_reference_recovery,
    radial_bound,
)


REFERENCE = Path("diagnostics/double_bottleneck_sxl_baseline_maturation")
ANCHORS_64 = 64
ANCHORS_128 = 128


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extra_episode(payload):
    episode, existing_steps = payload
    env_config = episode.metadata["environment"]["config"]
    config = Config(**env_config)
    available = np.setdiff1d(
        np.arange(episode.length, dtype=np.int64),
        np.asarray(sorted(existing_steps), dtype=np.int64),
        assume_unique=True,
    )
    if len(existing_steps) != ANCHORS_64 or len(available) < ANCHORS_64:
        raise ValueError(f"invalid nested anchors for {episode.rollout_id}")
    anchor_rng = np.random.default_rng(
        _seed("recovery_extra128", "train", episode.rollout_id)
    )
    selected = anchor_rng.permutation(available)[:ANCHORS_64]
    perturb_rng = np.random.default_rng(
        _seed("perturb_extra128", "train", episode.rollout_id)
    )
    labels = phase_labels(episode, config)
    accepted, rejected = [], []
    for local_rank, step_value in enumerate(selected):
        step = int(step_value)
        source = np.asarray(episode.observations[step], dtype=np.float64)
        delta_position = perturb_rng.uniform(-0.02, 0.02, (4, 2))
        delta_velocity = perturb_rng.uniform(-0.2, 0.2, (4, 2))
        positions = source[:, :2] + delta_position
        velocities = radial_bound(source[:, 2:4] + delta_velocity, config.max_speed)
        try:
            env = DoubleBottleneckEnv(config)
            env.reset(positions, regime=episode.regime)
        except ValueError as error:
            rejected.append({"step": step, "rank": ANCHORS_64 + local_rank, "reason": f"invalid_initial:{error}"})
            continue
        try:
            recovery = query_reference_recovery(episode, step, positions, velocities)
        except (ValueError, FloatingPointError) as error:
            rejected.append({"step": step, "rank": ANCHORS_64 + local_rank, "reason": f"query_exception:{error}"})
            continue
        if not recovery.success:
            rejected.append({"step": step, "rank": ANCHORS_64 + local_rank, "reason": f"recovery_{recovery.terminal_reason}"})
            continue
        wall, pair = env.distances()
        accepted.append(
            {
                "observation": observation_from_state(positions, velocities, env.goals),
                "action": recovery.actions[0].astype(np.float32),
                "source_step": step,
                "selection_rank": ANCHORS_64 + local_rank,
                "phase": str(labels[step]),
                "position_delta": delta_position.astype(np.float32),
                "velocity_delta": (velocities - source[:, 2:4]).astype(np.float32),
                "initial_wall_clearance": float(np.min(wall)),
                "initial_pair_clearance": float(np.min(pair)),
                "recovery_steps": recovery.recovery_steps,
                "recovery_min_wall": recovery.min_wall_clearance,
                "recovery_min_pair": recovery.min_pair_clearance,
                "mode_signature_match": recovery.mode_signature_match,
            }
        )
    return {
        "rollout_id": episode.rollout_id,
        "family_id": episode.family_id,
        "regime": episode.regime,
        "first_direction": episode.metadata["hypothesis"]["first_direction"],
        "accepted": accepted,
        "rejected": rejected,
    }


def _arrays(results):
    keys = (
        "observations", "actions", "source_rollout_id", "source_family_id",
        "source_episode_index", "source_step", "selection_rank", "source_phase",
        "regime", "first_direction", "position_delta", "velocity_delta",
        "initial_wall_clearance", "initial_pair_clearance", "recovery_steps",
        "recovery_min_wall", "recovery_min_pair", "mode_signature_match",
    )
    stored = {key: [] for key in keys}
    rejected = []
    for episode_index, result in enumerate(results):
        for row in result["accepted"]:
            stored["observations"].append(row["observation"])
            stored["actions"].append(row["action"])
            stored["source_rollout_id"].append(result["rollout_id"])
            stored["source_family_id"].append(result["family_id"])
            stored["source_episode_index"].append(episode_index)
            stored["source_step"].append(row["source_step"])
            stored["selection_rank"].append(row["selection_rank"])
            stored["source_phase"].append(row["phase"])
            stored["regime"].append(result["regime"])
            stored["first_direction"].append(result["first_direction"])
            for key in (
                "position_delta", "velocity_delta", "initial_wall_clearance",
                "initial_pair_clearance", "recovery_steps", "recovery_min_wall",
                "recovery_min_pair", "mode_signature_match",
            ):
                stored[key].append(row[key])
        rejected.extend({"rollout_id": result["rollout_id"], "family_id": result["family_id"], **row} for row in result["rejected"])
    result = {
        "observations": np.asarray(stored["observations"], dtype=np.float32),
        "actions": np.asarray(stored["actions"], dtype=np.float32),
        "source_rollout_id": np.asarray(stored["source_rollout_id"]),
        "source_family_id": np.asarray(stored["source_family_id"]),
        "source_episode_index": np.asarray(stored["source_episode_index"], dtype=np.int16),
        "source_step": np.asarray(stored["source_step"], dtype=np.int16),
        "selection_rank": np.asarray(stored["selection_rank"], dtype=np.int16),
        "source_phase": np.asarray(stored["source_phase"]),
        "regime": np.asarray(stored["regime"]),
        "first_direction": np.asarray(stored["first_direction"]),
        "position_delta": np.asarray(stored["position_delta"], dtype=np.float32),
        "velocity_delta": np.asarray(stored["velocity_delta"], dtype=np.float32),
        "initial_wall_clearance": np.asarray(stored["initial_wall_clearance"], dtype=np.float32),
        "initial_pair_clearance": np.asarray(stored["initial_pair_clearance"], dtype=np.float32),
        "recovery_steps": np.asarray(stored["recovery_steps"], dtype=np.int16),
        "recovery_min_wall": np.asarray(stored["recovery_min_wall"], dtype=np.float32),
        "recovery_min_pair": np.asarray(stored["recovery_min_pair"], dtype=np.float32),
        "mode_signature_match": np.asarray(stored["mode_signature_match"], dtype=bool),
        "split": np.asarray("train"),
        "anchors_per_trajectory": np.asarray(ANCHORS_64, dtype=np.int16),
        "root_seed_label": np.asarray(ROOT_LABEL),
    }
    return result, rejected


def _summary(arrays, rejection_rows, path, elapsed):
    position = arrays["position_delta"].astype(np.float64)
    velocity = arrays["velocity_delta"].astype(np.float64)
    return {
        "path": str(path.resolve()), "sha256": _sha(path),
        "source_trajectories": len(set(arrays["source_rollout_id"].astype(str))),
        "attempted": 2304 * ANCHORS_64, "accepted": len(arrays["actions"]),
        "rejected": len(rejection_rows),
        "phase_counts": dict(sorted(Counter(arrays["source_phase"].astype(str)).items())),
        "regime_counts": dict(sorted(Counter(arrays["regime"].astype(str)).items())),
        "first_direction_counts": dict(sorted(Counter(arrays["first_direction"].astype(str)).items())),
        "position_delta_rms": _quantiles(np.sqrt(np.mean(position ** 2, axis=(1, 2)))),
        "velocity_delta_rms": _quantiles(np.sqrt(np.mean(velocity ** 2, axis=(1, 2)))),
        "selection_rank_min": int(np.min(arrays["selection_rank"])),
        "selection_rank_max": int(np.max(arrays["selection_rank"])),
        "elapsed_seconds": elapsed,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("diagnostics/double_bottleneck_recovery_density_final/data"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = root / REFERENCE
    prereg = root / "diagnostics/double_bottleneck_recovery_density_final/PREREGISTRATION.json"
    old_path = reference / "data/recovery_train.npz"
    dataset = FlowBC4ADataset(reference / "data/train_pool", "train", seed=0)
    if len(dataset.episodes) != 2304:
        raise ValueError("frozen S-XL pool is not 2,304 trajectories")
    with np.load(old_path, allow_pickle=False) as archive:
        old = {key: archive[key].copy() for key in archive.files}
    groups = {}
    for rollout_id, step in zip(old["source_rollout_id"].astype(str), old["source_step"], strict=True):
        groups.setdefault(rollout_id, set()).add(int(step))
    if len(groups) != 2304 or set(map(len, groups.values())) != {ANCHORS_64}:
        raise ValueError("S-XL-64 recovery is not exactly 64 distinct anchors per trajectory")
    payloads = [(episode, groups[episode.rollout_id]) for episode in dataset.episodes]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = []
        for index, value in enumerate(executor.map(_extra_episode, payloads, chunksize=1), 1):
            results.append(value)
            if index % 96 == 0 or index == len(payloads):
                print(json.dumps({"stage": "uniform_extra64", "episodes_complete": index, "episodes_total": len(payloads)}, sort_keys=True), flush=True)
    extra, extra_rejections = _arrays(results)
    extra_path = output / "recovery_extra_64.npz"
    np.savez_compressed(extra_path, **extra)
    extra_rejection_path = output / "recovery_extra_64_rejections.json"
    extra_rejection_path.write_text(json.dumps(extra_rejections, indent=2, sort_keys=True) + "\n")
    if len(extra["actions"]) != 2304 * ANCHORS_64:
        raise ValueError("extra recovery acceptance differs from frozen direct counterpart")
    combined = {}
    for key, value in old.items():
        if value.ndim == 0:
            continue
        combined[key] = np.concatenate((value, extra[key]), axis=0)
    combined["split"] = np.asarray("train")
    combined["anchors_per_trajectory"] = np.asarray(ANCHORS_128, dtype=np.int16)
    combined["root_seed_label"] = np.asarray(ROOT_LABEL)
    combined["anchor_protocol"] = np.asarray("nested_existing64_plus_uniform_remaining64_v1")
    combined_path = output / "recovery_128_train.npz"
    np.savez_compressed(combined_path, **combined)
    if len(combined["actions"]) != 2304 * ANCHORS_128:
        raise ValueError("combined recovery count mismatch")
    overlap = 0
    for rollout_id in groups:
        old_steps = groups[rollout_id]
        extra_steps = set(extra["source_step"][extra["source_rollout_id"].astype(str) == rollout_id].tolist())
        overlap += len(old_steps & extra_steps)
    if overlap:
        raise ValueError("nested 128 anchors contain a duplicated step")
    reference_manifest = json.loads((reference / "data/manifest.json").read_text())
    manifest = {
        "schema": "double_bottleneck_recovery_density_final_data_v1",
        "preregistration_sha256": _sha(prereg),
        "reference_data_manifest_sha256": _sha(reference / "data/manifest.json"),
        "reference_train_pool_sha256": _sha(reference / "data/train_pool/manifest.json"),
        "train_pool": str((reference / "data/train_pool").resolve()),
        "nominal_transitions": reference_manifest["variant"]["nominal_transitions"],
        "expert_trajectories": 2304,
        "initial_states": 288,
        "reference_recovery_64": {"path": str(old_path.resolve()), "sha256": _sha(old_path), "rows": len(old["actions"])},
        "extra_recovery_64": _summary(extra, extra_rejections, extra_path, time.perf_counter() - started),
        "combined_recovery_128": {"path": str(combined_path.resolve()), "sha256": _sha(combined_path), "rows": len(combined["actions"]), "anchors_per_trajectory": ANCHORS_128, "overlapping_source_steps": overlap},
        "total_transitions": int(reference_manifest["variant"]["nominal_transitions"] + len(combined["actions"])),
        "effective_transition_increase": int(len(combined["actions"]) - len(old["actions"])),
        "same_nominal_trajectories": True,
        "same_8_mode_balance": True,
        "failure_or_phase_dependent_sampling": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"nominal": manifest["nominal_transitions"], "recovery_64": len(old["actions"]), "recovery_128": len(combined["actions"]), "total": manifest["total_transitions"], "extra_rejected": len(extra_rejections), "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
