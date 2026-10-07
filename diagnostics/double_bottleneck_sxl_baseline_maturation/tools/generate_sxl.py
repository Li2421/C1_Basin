#!/usr/bin/env python3
"""Generate the preregistered nested S-XL pool and fresh untouched test."""

from __future__ import annotations

import argparse
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
    MAX_DRAWS,
    RECOVERY_K,
    ROOT_LABEL,
    _generate_recovery,
    _generate_split,
    _pool_summary,
    _quantiles,
    _sample_spec,
    _seed,
    _solve_spec,
    _spec_json,
)
from double_bottleneck.expert_dataset import (
    InitialConditionSpec,
    load_record,
    save_dataset,
)
from double_bottleneck.scenario import INITIAL_REGIMES


PARENT = Path("diagnostics/double_bottleneck_initial_state_coverage")
OLD_PER_REGIME = 48
SXL_PER_REGIME = 96
FRESH_PER_REGIME = 8


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_specs(root: Path):
    manifest = json.loads((root / PARENT / "data/manifest.json").read_text())
    result = []
    for row in manifest["pools"]["train"]["initial_state_records"]:
        result.append(
            InitialConditionSpec(
                condition_id=row["condition_id"],
                family_id=row["family_id"],
                regime=row["regime"],
                positions=np.asarray(row["positions"], dtype=np.float64),
                initial_velocities=np.asarray(row["initial_velocities"], dtype=np.float64),
                perturbation=row["perturbation"],
            )
        )
    return result


def _old_records(root: Path):
    pool = root / PARENT / "data/train_pool"
    manifest = json.loads((pool / "manifest.json").read_text())
    return [load_record(pool / entry["file"]) for entry in manifest["files"]]


def _verify_prefix(old_specs, generated_prefix):
    old_by_id = {spec.condition_id: spec for spec in old_specs}
    for spec in generated_prefix:
        old = old_by_id.get(spec.condition_id)
        if old is None:
            raise ValueError(f"missing parent prefix state {spec.condition_id}")
        if not np.array_equal(spec.positions, old.positions):
            raise ValueError(f"parent position prefix mismatch: {spec.condition_id}")
        if not np.array_equal(spec.initial_velocities, old.initial_velocities):
            raise ValueError(f"parent velocity prefix mismatch: {spec.condition_id}")
        if dict(spec.perturbation) != dict(old.perturbation):
            raise ValueError(f"parent metadata prefix mismatch: {spec.condition_id}")


def _generate_train_extension(old_specs, workers):
    new_specs = []
    new_records = []
    rejections = []
    prefix_verified = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for regime in INITIAL_REGIMES:
            rng = np.random.default_rng(_seed("initial", "train", regime))
            stream = [
                _sample_spec("train", regime, candidate_index, rng)
                for candidate_index in range(SXL_PER_REGIME)
            ]
            prefix = stream[:OLD_PER_REGIME]
            _verify_prefix(old_specs, prefix)
            prefix_verified.extend(spec.condition_id for spec in prefix)
            candidate_index = SXL_PER_REGIME
            pending = stream[OLD_PER_REGIME:]
            results = list(
                executor.map(
                    _solve_spec,
                    [("train", "train", spec) for spec in pending],
                )
            )
            accepted = []
            queue = list(results)
            while queue:
                result = queue.pop(0)
                if result["accepted"]:
                    accepted.append(result)
                else:
                    rejections.append(
                        {"spec": _spec_json(result["spec"]), "reason": result["reason"]}
                    )
                if len(accepted) < SXL_PER_REGIME - OLD_PER_REGIME and not queue:
                    if candidate_index >= MAX_DRAWS:
                        raise RuntimeError(f"train/{regime} exhausted fixed draw budget")
                    spec = _sample_spec("train", regime, candidate_index, rng)
                    candidate_index += 1
                    queue.append(_solve_spec(("train", "train", spec)))
            new_specs.extend(result["spec"] for result in accepted)
            for result in accepted:
                new_records.extend(result["records"])
            print(
                json.dumps(
                    {
                        "stage": "sxl_train_extension",
                        "regime": regime,
                        "parent_prefix_verified": len(prefix),
                        "new_states_accepted": len(accepted),
                        "candidate_draws_total": candidate_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return new_specs, new_records, rejections, prefix_verified


def _combine_recovery(old_path: Path, new_path: Path, output: Path, old_episodes: int):
    with np.load(old_path, allow_pickle=False) as archive:
        old = {key: archive[key].copy() for key in archive.files}
    with np.load(new_path, allow_pickle=False) as archive:
        new = {key: archive[key].copy() for key in archive.files}
    new["source_episode_index"] = new["source_episode_index"] + old_episodes
    combined = {}
    for key in old:
        if old[key].ndim == 0:
            if str(old[key].item()) != str(new[key].item()):
                raise ValueError(f"recovery scalar mismatch for {key}")
            combined[key] = old[key]
        else:
            combined[key] = np.concatenate((old[key], new[key]), axis=0)
    np.savez_compressed(output, **combined)
    return combined


def _recovery_summary(arrays, path, rejection_path, elapsed):
    position = arrays["position_delta"].astype(np.float64)
    velocity = arrays["velocity_delta"].astype(np.float64)
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "rejection_path": str(rejection_path.resolve()),
        "rejection_sha256": _sha(rejection_path),
        "source_trajectories": len(set(arrays["source_rollout_id"].astype(str).tolist())),
        "attempted": int(len(arrays["actions"])),
        "accepted": int(len(arrays["actions"])),
        "rejected": len(json.loads(rejection_path.read_text())),
        "phase_counts": dict(sorted(Counter(arrays["source_phase"].astype(str)).items())),
        "regime_counts": dict(sorted(Counter(arrays["regime"].astype(str)).items())),
        "first_direction_counts": dict(
            sorted(Counter(arrays["first_direction"].astype(str)).items())
        ),
        "position_delta_rms": _quantiles(
            np.sqrt(np.mean(position**2, axis=(1, 2)))
        ),
        "velocity_delta_rms": _quantiles(
            np.sqrt(np.mean(velocity**2, axis=(1, 2)))
        ),
        "initial_wall_clearance": _quantiles(arrays["initial_wall_clearance"]),
        "initial_pair_clearance": _quantiles(arrays["initial_pair_clearance"]),
        "recovery_min_wall": _quantiles(arrays["recovery_min_wall"]),
        "recovery_min_pair": _quantiles(arrays["recovery_min_pair"]),
        "mode_signature_matches": int(np.sum(arrays["mode_signature_match"])),
        "elapsed_seconds_new_half_only": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/double_bottleneck_sxl_baseline_maturation/data"),
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
    prereg = root / "diagnostics/double_bottleneck_sxl_baseline_maturation/PREREGISTRATION.json"
    prereg_sha = _sha(prereg)
    started = time.perf_counter()

    old_specs = _old_specs(root)
    old_records = _old_records(root)
    if len(old_specs) != 144 or len(old_records) != 1152:
        raise ValueError("parent S-Large pool does not have the frozen 144x8 shape")
    new_specs, new_records, new_rejections, prefix_verified = _generate_train_extension(
        old_specs, args.workers
    )
    all_specs = old_specs + new_specs
    all_records = old_records + new_records
    if len(all_specs) != 288 or len(all_records) != 2304:
        raise AssertionError("S-XL did not produce exactly 288 states x 8 modes")

    train_path = output / "train_pool"
    train_manifest = save_dataset(
        train_path,
        all_records,
        generation_metadata={
            "schema": "double_bottleneck_sxl_train_pool_v1",
            "scientific_split": "train",
            "preregistration_sha256": prereg_sha,
            "root_seed_label": ROOT_LABEL,
            "states_per_regime": SXL_PER_REGIME,
            "modes_per_state": 8,
            "parent_prefix_states": 144,
            "parent_prefix_verified_exactly": True,
            "adaptive_sampling": False,
        },
    )
    (train_path / "initial_state_rejections.json").write_text(
        json.dumps(new_rejections, indent=2, sort_keys=True) + "\n"
    )

    new_path = output / "new_train_half_pool"
    save_dataset(
        new_path,
        new_records,
        generation_metadata={
            "schema": "double_bottleneck_sxl_new_half_v1",
            "scientific_split": "train",
            "preregistration_sha256": prereg_sha,
            "root_seed_label": ROOT_LABEL,
            "states_per_regime": SXL_PER_REGIME - OLD_PER_REGIME,
            "adaptive_sampling": False,
        },
    )

    fresh_specs, fresh_records, fresh_rejections = _generate_split(
        "fresh_test", FRESH_PER_REGIME, args.workers
    )
    fresh_path = output / "fresh_test_pool"
    fresh_manifest = save_dataset(
        fresh_path,
        fresh_records,
        generation_metadata={
            "schema": "double_bottleneck_sxl_fresh_test_pool_v1",
            "scientific_split": "fresh_test",
            "storage_split_label": "val",
            "preregistration_sha256": prereg_sha,
            "root_seed_label": ROOT_LABEL,
            "accepted_states_per_regime": FRESH_PER_REGIME,
            "modes_per_state": 8,
            "adaptive_sampling": False,
            "outcomes_inspected_before_training": False,
        },
    )
    (fresh_path / "initial_state_rejections.json").write_text(
        json.dumps(fresh_rejections, indent=2, sort_keys=True) + "\n"
    )

    new_recovery_path = output / "recovery_new_train_half.npz"
    new_recovery_started = time.perf_counter()
    new_recovery_summary = _generate_recovery(
        new_path, "train", new_recovery_path, args.workers
    )
    old_recovery_path = root / PARENT / "data/recovery_train.npz"
    combined_path = output / "recovery_train.npz"
    combined = _combine_recovery(
        old_recovery_path, new_recovery_path, combined_path, len(old_records)
    )
    old_rejections = json.loads(
        (root / PARENT / "data/recovery_train_rejections.json").read_text()
    )
    new_recovery_rejections = json.loads(
        new_recovery_path.with_name("recovery_new_train_half_rejections.json").read_text()
    )
    combined_rejection_path = output / "recovery_train_rejections.json"
    combined_rejection_path.write_text(
        json.dumps(old_rejections + new_recovery_rejections, indent=2, sort_keys=True) + "\n"
    )
    recovery_summary = _recovery_summary(
        combined,
        combined_path,
        combined_rejection_path,
        time.perf_counter() - new_recovery_started,
    )
    expected_recovery = len(all_records) * RECOVERY_K
    if len(combined["actions"]) != expected_recovery:
        raise ValueError("combined S-XL recovery count does not equal 64 per trajectory")

    train_summary = _pool_summary(
        train_path, all_specs, all_records, new_rejections, train_manifest
    )
    train_summary["parent_prefix_verified_state_ids"] = sorted(prefix_verified)
    fresh_summary = _pool_summary(
        fresh_path, fresh_specs, fresh_records, fresh_rejections, fresh_manifest
    )
    nominal_transitions = int(sum(record.length for record in all_records))
    variant = {
        "schema": "double_bottleneck_sxl_variant_v1",
        "variant": "S-XL",
        "preregistration_sha256": prereg_sha,
        "initial_states": len(all_specs),
        "states_per_regime": SXL_PER_REGIME,
        "family_ids": sorted(spec.family_id for spec in all_specs),
        "expert_trajectory_ids": sorted(record.rollout_id for record in all_records),
        "expert_trajectories": len(all_records),
        "nominal_transitions": nominal_transitions,
        "recovery_transitions": len(combined["actions"]),
        "total_transitions": nominal_transitions + len(combined["actions"]),
        "anchors_attempted_per_trajectory": RECOVERY_K,
        "train_pool_manifest_sha256": _sha(train_path / "manifest.json"),
        "recovery_train_sha256": _sha(combined_path),
        "parent_s_large_is_exact_prefix": True,
    }
    variant_path = output / "s_xl_manifest.json"
    variant_path.write_text(json.dumps(variant, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "double_bottleneck_sxl_data_v1",
        "preregistration_sha256": prereg_sha,
        "root_seed_label": ROOT_LABEL,
        "workers": args.workers,
        "train_pool": train_summary,
        "fresh_test_pool": fresh_summary,
        "recovery": recovery_summary,
        "new_half_recovery": new_recovery_summary,
        "variant": {
            **variant,
            "manifest": str(variant_path.resolve()),
            "manifest_sha256": _sha(variant_path),
        },
        "fresh_test_generated_before_training": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "train_states": train_summary["initial_states"],
                "train_trajectories": train_summary["expert_trajectories"],
                "nominal_transitions": nominal_transitions,
                "recovery_transitions": len(combined["actions"]),
                "fresh_test_states": fresh_summary["initial_states"],
                "expert_rejections": len(new_rejections) + len(fresh_rejections),
                "recovery_rejections": recovery_summary["rejected"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
