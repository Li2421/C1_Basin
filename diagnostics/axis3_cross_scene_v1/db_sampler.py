#!/usr/bin/env python3
"""Generate preregistered broad initial-state pools and uniform recoveries."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from double_bottleneck.environment import Config, DoubleBottleneckEnv
from double_bottleneck.expert import CentralizedExpert, all_coordination_hypotheses
from double_bottleneck.expert_dataset import (
    InitialConditionSpec,
    initialize_environment,
    record_from_plan,
    save_dataset,
)
from double_bottleneck.flowbc_4a_dataset import Episode, FlowBC4ADataset
from double_bottleneck.recovery_v2 import (
    observation_from_state,
    phase_labels,
    query_reference_recovery,
    radial_bound,
)
from double_bottleneck.scenario import INITIAL_REGIMES, initial_positions


ROOT_LABEL = "double_bottleneck_initial_coverage_v1|20260924"
TRAIN_COUNTS = {"S-Small": 8, "S-Medium": 24, "S-Large": 48}
OTHER_COUNT = 8
RECOVERY_K = 64
MAX_DRAWS = 4096


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(*parts: str) -> int:
    value = "|".join((ROOT_LABEL, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little", signed=False)


def _sample_spec(split: str, regime: str, candidate_index: int, rng) -> InitialConditionSpec:
    config = Config()
    travel = np.asarray((1.0, 1.0, -1.0, -1.0), dtype=np.float64)
    side_progress = rng.uniform(-0.10, 0.10, 2)
    progress_spacing = rng.uniform(-0.05, 0.05, 2)
    progress = np.asarray(
        (
            side_progress[0] + 0.5 * progress_spacing[0],
            side_progress[0] - 0.5 * progress_spacing[0],
            side_progress[1] + 0.5 * progress_spacing[1],
            side_progress[1] - 0.5 * progress_spacing[1],
        )
    )
    side_lateral = rng.uniform(-0.015, 0.015, 2)
    lateral_spacing = rng.uniform(-0.010, 0.010, 2)
    lateral = np.asarray(
        (
            side_lateral[0] + 0.5 * lateral_spacing[0],
            side_lateral[0] - 0.5 * lateral_spacing[0],
            side_lateral[1] + 0.5 * lateral_spacing[1],
            side_lateral[1] - 0.5 * lateral_spacing[1],
        )
    )
    delta_position = np.stack((travel * progress, lateral), axis=-1)
    parallel_velocity = rng.uniform(-0.05, 0.05, 4)
    lateral_velocity = rng.uniform(-0.02, 0.02, 4)
    velocity = np.stack((travel * parallel_velocity, lateral_velocity), axis=-1)
    condition_id = f"coverage_{split}__{regime}__candidate_{candidate_index:04d}"
    return InitialConditionSpec(
        condition_id=condition_id,
        family_id=condition_id,
        regime=regime,
        positions=initial_positions(config, regime) + delta_position,
        initial_velocities=velocity,
        perturbation={
            "family": "preregistered_global_initial_state_v1",
            "split": split,
            "candidate_index": candidate_index,
            "side_progress_common": side_progress.tolist(),
            "within_side_progress_spacing": progress_spacing.tolist(),
            "side_lateral_common": side_lateral.tolist(),
            "within_side_lateral_spacing": lateral_spacing.tolist(),
            "parallel_velocity": parallel_velocity.tolist(),
            "lateral_velocity": lateral_velocity.tolist(),
            "delta_positions": delta_position.tolist(),
            "initial_velocities": velocity.tolist(),
        },
    )


def _solve_spec(payload):
    split, storage_split, spec = payload
    config = Config()
    try:
        initialize_environment(config, spec)
    except ValueError as error:
        return {"accepted": False, "reason": f"invalid_initial:{error}", "spec": spec}
    expert = CentralizedExpert()
    records = []
    for mode_index, hypothesis in enumerate(all_coordination_hypotheses()):
        env = initialize_environment(config, spec)
        plan = expert.plan_hypothesis(env, hypothesis)
        record = record_from_plan(
            plan,
            env,
            spec,
            f"{spec.condition_id}__mode_{mode_index}",
        )
        if not record.training_eligible:
            return {
                "accepted": False,
                "reason": f"expert_{mode_index}:{record.terminal_reason}",
                "spec": spec,
            }
        records.append(replace(record, split=storage_split))
    if len({record.coordination_mode["signature"] for record in records}) != 8:
        return {"accepted": False, "reason": "fewer_than_8_actual_modes", "spec": spec}
    return {"accepted": True, "records": records, "spec": spec, "split": split}


def _spec_json(spec: InitialConditionSpec) -> dict:
    return {
        "condition_id": spec.condition_id,
        "family_id": spec.family_id,
        "regime": spec.regime,
        "positions": spec.positions.tolist(),
        "initial_velocities": spec.initial_velocities.tolist(),
        "perturbation": dict(spec.perturbation),
    }


def _generate_split(split: str, count_per_regime: int, workers: int):
    storage_split = "train" if split == "train" else "val"
    accepted_specs = []
    records = []
    rejections = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for regime in INITIAL_REGIMES:
            rng = np.random.default_rng(_seed("initial", split, regime))
            candidate_index = 0
            accepted_regime = []
            # Submit exactly the required initial iid candidates first.  Any
            # rejected candidate is replaced only by the next global iid draw.
            pending_specs = []
            for _ in range(count_per_regime):
                pending_specs.append(_sample_spec(split, regime, candidate_index, rng))
                candidate_index += 1
            results = list(
                executor.map(
                    _solve_spec,
                    [(split, storage_split, spec) for spec in pending_specs],
                )
            )
            while results:
                result = results.pop(0)
                if result["accepted"]:
                    accepted_regime.append(result)
                else:
                    rejections.append(
                        {"spec": _spec_json(result["spec"]), "reason": result["reason"]}
                    )
                if len(accepted_regime) < count_per_regime and not results:
                    if candidate_index >= MAX_DRAWS:
                        raise RuntimeError(f"{split}/{regime} exhausted fixed draw budget")
                    spec = _sample_spec(split, regime, candidate_index, rng)
                    candidate_index += 1
                    results.append(_solve_spec((split, storage_split, spec)))
            accepted_specs.extend(result["spec"] for result in accepted_regime)
            for result in accepted_regime:
                records.extend(result["records"])
            print(
                json.dumps(
                    {
                        "stage": "expert_pool",
                        "split": split,
                        "regime": regime,
                        "accepted": len(accepted_regime),
                        "candidate_draws": candidate_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return accepted_specs, records, rejections


def _recovery_episode(payload):
    split, episode = payload
    config = Config(**episode.metadata["environment"]["config"])
    anchor_rng = np.random.default_rng(_seed("recovery", split, episode.rollout_id))
    perturb_rng = np.random.default_rng(_seed("perturb", split, episode.rollout_id))
    selected = anchor_rng.permutation(episode.length)[:RECOVERY_K]
    labels = phase_labels(episode, config)
    accepted = []
    rejected = []
    for rank, step_value in enumerate(selected):
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
            rejected.append(
                {"step": step, "rank": rank, "reason": f"invalid_initial:{error}"}
            )
            continue
        try:
            recovery = query_reference_recovery(episode, step, positions, velocities)
        except (ValueError, FloatingPointError) as error:
            rejected.append(
                {"step": step, "rank": rank, "reason": f"query_exception:{error}"}
            )
            continue
        if not recovery.success:
            rejected.append(
                {
                    "step": step,
                    "rank": rank,
                    "reason": f"recovery_{recovery.terminal_reason}",
                }
            )
            continue
        wall, pair = env.distances()
        accepted.append(
            {
                "observation": observation_from_state(positions, velocities, env.goals),
                "action": recovery.actions[0].astype(np.float32),
                "source_step": step,
                "selection_rank": rank,
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


def _quantiles(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ("min", "p05", "median", "mean", "p95", "max")}
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _generate_recovery(pool: Path, split: str, output: Path, workers: int) -> dict:
    storage_split = "train" if split == "train" else "val"
    dataset = FlowBC4ADataset(pool, storage_split, seed=0)
    stored = {
        key: []
        for key in (
            "observations",
            "actions",
            "source_rollout_id",
            "source_family_id",
            "source_episode_index",
            "source_step",
            "selection_rank",
            "source_phase",
            "regime",
            "first_direction",
            "position_delta",
            "velocity_delta",
            "initial_wall_clearance",
            "initial_pair_clearance",
            "recovery_steps",
            "recovery_min_wall",
            "recovery_min_pair",
            "mode_signature_match",
        )
    }
    rejection_rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            _recovery_episode, [(split, episode) for episode in dataset.episodes], chunksize=1
        )
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
                    "position_delta",
                    "velocity_delta",
                    "initial_wall_clearance",
                    "initial_pair_clearance",
                    "recovery_steps",
                    "recovery_min_wall",
                    "recovery_min_pair",
                    "mode_signature_match",
                ):
                    stored[key].append(row[key])
            for rejection in result["rejected"]:
                rejection_rows.append(
                    {
                        "rollout_id": result["rollout_id"],
                        "family_id": result["family_id"],
                        **rejection,
                    }
                )
            if (episode_index + 1) % 96 == 0 or episode_index + 1 == len(dataset.episodes):
                print(
                    json.dumps(
                        {
                            "stage": "uniform_recovery",
                            "split": split,
                            "episodes_complete": episode_index + 1,
                            "episodes_total": len(dataset.episodes),
                            "accepted_rows": len(stored["actions"]),
                            "rejected_rows": len(rejection_rows),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    arrays = {
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
        "split": np.asarray(split),
        "anchors_per_trajectory": np.asarray(RECOVERY_K, dtype=np.int16),
        "root_seed_label": np.asarray(ROOT_LABEL),
    }
    np.savez_compressed(output, **arrays)
    rejection_path = output.with_name(output.stem + "_rejections.json")
    rejection_path.write_text(json.dumps(rejection_rows, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(output.resolve()),
        "sha256": _sha(output),
        "rejection_path": str(rejection_path.resolve()),
        "rejection_sha256": _sha(rejection_path),
        "source_trajectories": len(dataset.episodes),
        "attempted": len(dataset.episodes) * RECOVERY_K,
        "accepted": len(stored["actions"]),
        "rejected": len(rejection_rows),
        "phase_counts": dict(sorted(Counter(stored["source_phase"]).items())),
        "regime_counts": dict(sorted(Counter(stored["regime"]).items())),
        "first_direction_counts": dict(
            sorted(Counter(stored["first_direction"]).items())
        ),
        "position_delta_rms": _quantiles(
            np.sqrt(np.mean(arrays["position_delta"].astype(np.float64) ** 2, axis=(1, 2)))
        ),
        "velocity_delta_rms": _quantiles(
            np.sqrt(np.mean(arrays["velocity_delta"].astype(np.float64) ** 2, axis=(1, 2)))
        ),
        "initial_wall_clearance": _quantiles(arrays["initial_wall_clearance"]),
        "initial_pair_clearance": _quantiles(arrays["initial_pair_clearance"]),
        "recovery_min_wall": _quantiles(arrays["recovery_min_wall"]),
        "recovery_min_pair": _quantiles(arrays["recovery_min_pair"]),
        "mode_signature_matches": int(np.sum(arrays["mode_signature_match"])),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _pool_summary(path: Path, specs, records, rejections, manifest) -> dict:
    lengths = np.asarray([record.length for record in records], dtype=np.int64)
    return {
        "path": str(path.resolve()),
        "manifest_sha256": _sha(path / "manifest.json"),
        "initial_states": len(specs),
        "expert_trajectories": len(records),
        "nominal_transitions": int(np.sum(lengths)),
        "rejections": len(rejections),
        "rejection_reasons": dict(sorted(Counter(row["reason"] for row in rejections).items())),
        "episode_length": _quantiles(lengths),
        "minimum_wall_clearance": float(
            min(record.min_swept_wall_clearance for record in records)
        ),
        "minimum_pair_clearance": float(
            min(record.min_swept_pair_surface_distance for record in records)
        ),
        "actual_mode_counts": manifest["mode_analysis"]["mode_counts"],
        "initial_state_records": [_spec_json(spec) for spec in specs],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_initial_state_coverage/data"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    prereg = root / "diagnostics/double_bottleneck_initial_state_coverage/PREREGISTRATION.json"
    prereg_sha = _sha(prereg)
    started = time.perf_counter()
    summary = {
        "schema": "double_bottleneck_initial_state_coverage_data_v1",
        "preregistration_sha256": prereg_sha,
        "root_seed_label": ROOT_LABEL,
        "workers": args.workers,
        "pools": {},
        "variants": {},
    }
    split_results = {}
    for split, count in (("train", 48), ("validation", OTHER_COUNT), ("untouched_test", OTHER_COUNT)):
        specs, records, rejections = _generate_split(split, count, args.workers)
        pool_path = output / f"{split}_pool"
        manifest = save_dataset(
            pool_path,
            records,
            generation_metadata={
                "schema": "double_bottleneck_initial_state_pool_v1",
                "scientific_split": split,
                "storage_split_label": "train" if split == "train" else "val",
                "preregistration_sha256": prereg_sha,
                "root_seed_label": ROOT_LABEL,
                "accepted_states_per_regime": count,
                "modes_per_state": 8,
                "adaptive_sampling": False,
            },
        )
        rejection_path = pool_path / "initial_state_rejections.json"
        rejection_path.write_text(json.dumps(rejections, indent=2, sort_keys=True) + "\n")
        summary["pools"][split] = _pool_summary(
            pool_path, specs, records, rejections, manifest
        )
        summary["pools"][split]["rejection_manifest_sha256"] = _sha(rejection_path)
        split_results[split] = (specs, records, pool_path)
    train_specs, train_records, train_path = split_results["train"]
    validation_specs, validation_records, validation_path = split_results["validation"]
    recovery_train = _generate_recovery(
        train_path, "train", output / "recovery_train.npz", args.workers
    )
    recovery_validation = _generate_recovery(
        validation_path,
        "validation",
        output / "recovery_validation.npz",
        args.workers,
    )
    summary["recovery"] = {
        "train": recovery_train,
        "validation": recovery_validation,
    }
    by_family_records = {}
    for record in train_records:
        by_family_records.setdefault(record.family_id, []).append(record)
    accepted_by_regime = {
        regime: [spec for spec in train_specs if spec.regime == regime]
        for regime in INITIAL_REGIMES
    }
    with np.load(output / "recovery_train.npz", allow_pickle=False) as archive:
        recovery_family = archive["source_family_id"].astype(str)
    manifests = output / "variant_manifests"
    manifests.mkdir()
    for variant, per_regime in TRAIN_COUNTS.items():
        selected_specs = tuple(
            spec
            for regime in INITIAL_REGIMES
            for spec in accepted_by_regime[regime][:per_regime]
        )
        selected_families = {spec.family_id for spec in selected_specs}
        selected_records = [
            record for family in selected_families for record in by_family_records[family]
        ]
        recovery_count = int(np.sum(np.isin(recovery_family, list(selected_families))))
        variant_manifest = {
            "schema": "double_bottleneck_initial_state_coverage_variant_v1",
            "variant": variant,
            "preregistration_sha256": prereg_sha,
            "initial_states": len(selected_specs),
            "states_per_regime": per_regime,
            "family_ids": sorted(selected_families),
            "expert_trajectory_ids": sorted(record.rollout_id for record in selected_records),
            "expert_trajectories": len(selected_records),
            "nominal_transitions": int(sum(record.length for record in selected_records)),
            "recovery_transitions": recovery_count,
            "total_transitions": int(sum(record.length for record in selected_records))
            + recovery_count,
            "anchors_attempted_per_trajectory": RECOVERY_K,
            "train_pool_manifest_sha256": _sha(train_path / "manifest.json"),
            "recovery_train_sha256": recovery_train["sha256"],
            "validation_pool_manifest_sha256": _sha(validation_path / "manifest.json"),
            "recovery_validation_sha256": recovery_validation["sha256"],
        }
        path = manifests / f"{variant.lower().replace('-', '_')}.json"
        path.write_text(json.dumps(variant_manifest, indent=2, sort_keys=True) + "\n")
        summary["variants"][variant] = {
            **variant_manifest,
            "manifest": str(path.resolve()),
            "manifest_sha256": _sha(path),
        }
    summary["elapsed_seconds"] = time.perf_counter() - started
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "pools": {
                    key: {
                        name: value[name]
                        for name in (
                            "initial_states",
                            "expert_trajectories",
                            "nominal_transitions",
                            "rejections",
                        )
                    }
                    for key, value in summary["pools"].items()
                },
                "recovery": {
                    key: {
                        name: value[name]
                        for name in ("attempted", "accepted", "rejected")
                    }
                    for key, value in summary["recovery"].items()
                },
                "variants": {
                    key: {
                        name: value[name]
                        for name in (
                            "initial_states",
                            "expert_trajectories",
                            "nominal_transitions",
                            "recovery_transitions",
                            "total_transitions",
                        )
                    }
                    for key, value in summary["variants"].items()
                },
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
