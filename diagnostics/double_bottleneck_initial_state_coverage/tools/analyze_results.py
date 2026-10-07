#!/usr/bin/env python3
"""Post-hoc analysis for the frozen initial-state coverage experiment.

This script reads the already frozen comparison.  It never calls the expert,
generates data, trains a policy, or changes a checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from double_bottleneck.environment import Config
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import phase_labels


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
THRESHOLD = 0.039569792891474595
SCALES = ("S-Small", "S-Medium", "S-Large")
REGIMES = ("clearly_asymmetric", "weakly_asymmetric", "near_symmetric")
DIV_THRESHOLD_M = 0.08
DIV_PERSISTENCE = 5


def _read(path: Path):
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(root_label: str, *parts: str) -> int:
    value = "|".join((root_label, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little", signed=False)


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _feature_row(episode):
    perturb = episode.metadata["perturbation"]
    side_progress = np.asarray(perturb["side_progress_common"], dtype=np.float64)
    progress_spacing = np.asarray(
        perturb["within_side_progress_spacing"], dtype=np.float64
    )
    side_lateral = np.asarray(perturb["side_lateral_common"], dtype=np.float64)
    lateral_spacing = np.asarray(
        perturb["within_side_lateral_spacing"], dtype=np.float64
    )
    velocities = np.asarray(perturb["initial_velocities"], dtype=np.float64)
    delta_positions = np.asarray(perturb["delta_positions"], dtype=np.float64)
    return {
        "family_id": episode.family_id,
        "regime": episode.regime,
        "candidate_index": int(perturb["candidate_index"]),
        "mean_initial_speed": float(np.mean(np.linalg.norm(velocities, axis=1))),
        "max_initial_speed": float(np.max(np.linalg.norm(velocities, axis=1))),
        "mean_abs_longitudinal_offset": float(np.mean(np.abs(delta_positions[:, 0]))),
        "mean_abs_lateral_offset": float(np.mean(np.abs(delta_positions[:, 1]))),
        "left_right_progress_imbalance": float(abs(side_progress[0] - side_progress[1])),
        "mean_abs_within_side_progress_spacing": float(np.mean(np.abs(progress_spacing))),
        "mean_abs_side_lateral_offset": float(np.mean(np.abs(side_lateral))),
        "mean_abs_within_side_lateral_spacing": float(np.mean(np.abs(lateral_spacing))),
    }


def _unique_feature_rows(dataset):
    return [_feature_row(dataset.by_family[family][0]) for family in dataset.family_names]


def _initial_distribution_summary(rows):
    names = [name for name in rows[0] if name not in ("family_id", "regime", "candidate_index")]
    return {
        name: _quantiles([row[name] for row in rows])
        for name in names
    }


def _terminal_counts(result):
    return dict(sorted(Counter(row["termination"] for row in result["rollouts"]).items()))


def _mode_counts(result):
    directions = Counter()
    signatures = Counter()
    for row in result["rollouts"]:
        if not row["success"]:
            continue
        mode = row["coordination_mode"]
        directions[mode["first_direction"]] += 1
        signatures[mode["signature"]] += 1
    return {
        "first_direction": dict(sorted(directions.items())),
        "signature": dict(sorted(signatures.items())),
    }


def _regime_counts(result):
    counts = {regime: Counter() for regime in REGIMES}
    for row in result["rollouts"]:
        item = counts[row["regime"]]
        item["rollouts"] += 1
        item["success"] += int(row["success"])
        item["wall_collision"] += int(row["wall_collision"])
        item["agent_collision"] += int(row["agent_collision"])
        item["timeout"] += int(row["timeout"])
    return {regime: dict(counts[regime]) for regime in REGIMES}


def _family_success(result):
    values = defaultdict(list)
    for row in result["rollouts"]:
        values[row["family_id"]].append(float(row["success"]))
    return {family: float(np.mean(success)) for family, success in values.items()}


def _correlations(test_features, comparison):
    output = {}
    feature_names = [
        key
        for key in test_features[0]
        if key not in ("family_id", "regime", "candidate_index")
    ]
    for scale in SCALES:
        result = comparison["sets"]["untouched_test"][scale]
        family_success = _family_success(result)
        target = np.asarray([family_success[row["family_id"]] for row in test_features])
        output[scale] = {}
        for name in feature_names:
            statistic = spearmanr(
                np.asarray([row[name] for row in test_features]), target
            )
            output[scale][name] = {
                "spearman_rho": (
                    float(statistic.statistic)
                    if np.isfinite(statistic.statistic)
                    else None
                ),
                "two_sided_p": (
                    float(statistic.pvalue) if np.isfinite(statistic.pvalue) else None
                ),
            }
    return output


def _first_persistent(mask: np.ndarray, length: int):
    if len(mask) < length:
        return None
    hits = np.convolve(mask.astype(np.int8), np.ones(length, dtype=np.int8), mode="valid")
    found = np.flatnonzero(hits == length)
    return int(found[0]) if len(found) else None


def _failure_divergence(test_dataset, frozen_result):
    phase_cache = {}
    tree_cache = {}
    config = Config(**test_dataset.config)
    for family in test_dataset.family_names:
        positions = []
        phases = []
        for episode in test_dataset.by_family[family]:
            episode_positions = np.asarray(episode.positions, dtype=np.float64)
            labels = phase_labels(episode, config)
            positions.append(episode_positions.reshape((len(episode_positions), -1)))
            phases.extend(str(value) for value in labels)
        flat = np.concatenate(positions, axis=0) / np.sqrt(8.0)
        tree_cache[family] = cKDTree(flat)
        phase_cache[family] = np.asarray(phases)

    rows = []
    phase_counts = Counter()
    terminal_phase_counts = Counter()
    supported_at_divergence = []
    for summary in frozen_result["rollouts"]:
        if summary["success"]:
            continue
        path = (
            STUDY
            / "evaluation/untouched_test/trajectories"
            / f"s-large_{summary['rollout_id']:03d}.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["positions"], dtype=np.float64)
            support = np.asarray(archive["support_distances"], dtype=np.float64)
        family = summary["family_id"]
        distances, indices = tree_cache[family].query(
            positions.reshape((len(positions), -1)) / np.sqrt(8.0), workers=1
        )
        first = _first_persistent(distances > DIV_THRESHOLD_M, DIV_PERSISTENCE)
        if first is None:
            divergence_phase = None
            divergence_support = None
            divergence_distance = None
        else:
            divergence_phase = str(phase_cache[family][indices[first]])
            divergence_support = float(support[first])
            divergence_distance = float(distances[first])
            phase_counts[divergence_phase] += 1
            supported_at_divergence.append(divergence_support <= THRESHOLD)
        terminal_phase = str(phase_cache[family][indices[-1]])
        terminal_phase_counts[terminal_phase] += 1
        rows.append(
            {
                "rollout_id": int(summary["rollout_id"]),
                "family_id": family,
                "regime": summary["regime"],
                "seed": int(summary["seed"]),
                "termination": summary["termination"],
                "episode_steps": int(summary["episode_steps"]),
                "first_clear_divergence_step": first,
                "first_clear_divergence_phase": divergence_phase,
                "expert_position_rms_at_divergence_m": divergence_distance,
                "support_distance_at_divergence": divergence_support,
                "inside_strict_support_at_divergence": (
                    divergence_support <= THRESHOLD
                    if divergence_support is not None
                    else None
                ),
                "terminal_nearest_expert_phase": terminal_phase,
                "terminal_expert_position_rms_m": float(distances[-1]),
                "timestep0_support_distance": float(summary["timestep0_support_distance"]),
                "whole_rollout_outside_support_fraction": float(
                    summary["outside_support_fraction"]
                ),
            }
        )
    divergent_steps = [
        row["first_clear_divergence_step"]
        for row in rows
        if row["first_clear_divergence_step"] is not None
    ]
    return {
        "definition": {
            "distance": "joint-position RMS to nearest state among all 8 expert trajectories for the same untouched initial state",
            "threshold_m": DIV_THRESHOLD_M,
            "persistence_steps": DIV_PERSISTENCE,
            "phase": "phase label of that nearest same-family expert state",
        },
        "failed_rollouts": len(rows),
        "failures_with_clear_divergence": len(divergent_steps),
        "median_first_clear_divergence_step": (
            float(np.median(divergent_steps)) if divergent_steps else None
        ),
        "first_clear_divergence_phase_counts": dict(sorted(phase_counts.items())),
        "terminal_nearest_expert_phase_counts": dict(sorted(terminal_phase_counts.items())),
        "fraction_inside_strict_support_at_divergence": (
            float(np.mean(supported_at_divergence)) if supported_at_divergence else None
        ),
        "rows": rows,
        "expert_requery_performed": False,
        "data_generated_from_failures": False,
    }


def _recovery_stats(path: Path, selected_families: set[str]):
    with np.load(path, allow_pickle=False) as archive:
        mask = np.isin(archive["source_family_id"].astype(str), list(selected_families))
        phase = archive["source_phase"][mask].astype(str)
        pos = archive["position_delta"][mask]
        vel = archive["velocity_delta"][mask]
        return {
            "samples": int(np.sum(mask)),
            "natural_phase_counts": dict(sorted(Counter(phase.tolist()).items())),
            "position_delta_norm_m": _quantiles(np.linalg.norm(pos, axis=-1).reshape(-1)),
            "velocity_delta_norm_mps": _quantiles(np.linalg.norm(vel, axis=-1).reshape(-1)),
            "initial_wall_clearance_m": _quantiles(archive["initial_wall_clearance"][mask]),
            "initial_pair_clearance_m": _quantiles(archive["initial_pair_clearance"][mask]),
            "mode_signature_match_fraction": float(np.mean(archive["mode_signature_match"][mask])),
        }


def _make_manifests(prereg, data_manifest, training):
    output = STUDY / "manifests"
    output.mkdir(exist_ok=True)
    prereg_hash = _sha(STUDY / "PREREGISTRATION.json")
    root_label = prereg["seed_scheme"]["root_label"]
    training_by_variant = {row["variant"]: row for row in training["variants"]}
    for variant in SCALES:
        data = data_manifest["variants"][variant]
        trained = training_by_variant[variant]
        manifest = {
            "schema": "double_bottleneck_initial_state_coverage_final_manifest_v1",
            "variant": variant,
            "preregistration": str((STUDY / "PREREGISTRATION.json").resolve()),
            "preregistration_sha256": prereg_hash,
            "sampling_config": prereg,
            "initial_state_stream_seeds": {
                regime: _seed(root_label, "initial", "train", regime)
                for regime in REGIMES
            },
            "perturbation_seed_protocol": prereg["uniform_recovery"]["perturbation_seed"],
            "anchor_seed_protocol": prereg["uniform_recovery"]["anchor_rule"],
            "family_ids": data["family_ids"],
            "expert_trajectory_ids": data["expert_trajectory_ids"],
            "dataset_counts": {
                key: data[key]
                for key in (
                    "initial_states",
                    "expert_trajectories",
                    "nominal_transitions",
                    "recovery_transitions",
                    "total_transitions",
                )
            },
            "training_configuration": trained,
            "checkpoint": trained["checkpoint"],
            "checkpoint_sha256": trained["checkpoint_sha256"],
        }
        (output / f"{variant.lower().replace('-', '_')}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    for split in ("validation", "untouched_test"):
        pool_path = STUDY / f"data/{split}_pool/manifest.json"
        pool = _read(pool_path)
        manifest = {
            "schema": "double_bottleneck_initial_state_coverage_frozen_pool_v1",
            "scientific_split": split,
            "preregistration_sha256": prereg_hash,
            "pool_manifest": str(pool_path.resolve()),
            "pool_manifest_sha256": _sha(pool_path),
            "initial_state_stream_seeds": {
                regime: _seed(root_label, "initial", split, regime)
                for regime in REGIMES
            },
            "initial_states": pool["quality_report"]["families"],
            "expert_trajectories": pool["quality_report"]["rollouts"],
            "transitions": pool["quality_report"]["transitions"],
            "initial_state_sha256": sorted(
                item["initial_state_sha256"]
                for item in pool["mode_analysis"]["multimodal_initial_state_groups"]
            ),
            "test_used_for_checkpoint_selection": False,
            "test_failures_used_for_data_generation": False,
        }
        (output / f"{split}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


def _protocol_validation(prereg, data_manifest, training):
    prereg_hash = _sha(STUDY / "PREREGISTRATION.json")
    train_pool = _read(STUDY / "data/train_pool/manifest.json")
    validation_pool = _read(STUDY / "data/validation_pool/manifest.json")
    test_pool = _read(STUDY / "data/untouched_test_pool/manifest.json")
    hashes = {}
    for name, pool in (
        ("train", train_pool),
        ("validation", validation_pool),
        ("untouched_test", test_pool),
    ):
        hashes[name] = {
            item["initial_state_sha256"]
            for item in pool["mode_analysis"]["multimodal_initial_state_groups"]
        }
    variants = data_manifest["variants"]
    small = set(variants["S-Small"]["family_ids"])
    medium = set(variants["S-Medium"]["family_ids"])
    large = set(variants["S-Large"]["family_ids"])
    canonical = ROOT / "double_bottleneck/flowbc_4a_agent.py"
    canonical_hash = _sha(canonical)
    expected_hash = "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8"
    checks = {
        "preregistration_hash_propagated_to_data": data_manifest["preregistration_sha256"] == prereg_hash,
        "preregistration_hash_propagated_to_training": all(
            row["preregistration_sha256"] == prereg_hash for row in training["variants"]
        ),
        "nested_training_pools": small < medium < large,
        "split_initial_states_disjoint": not (
            hashes["train"] & hashes["validation"]
            or hashes["train"] & hashes["untouched_test"]
            or hashes["validation"] & hashes["untouched_test"]
        ),
        "requested_pool_sizes": tuple(
            variants[name]["initial_states"] for name in SCALES
        ) == (24, 72, 144),
        "eight_modes_per_state": all(
            pool["mode_analysis"]["identical_initial_states_with_multiple_successful_modes"]
            == pool["quality_report"]["families"]
            and pool["quality_report"]["rollouts"] == 8 * pool["quality_report"]["families"]
            for pool in (train_pool, validation_pool, test_pool)
        ),
        "all_experts_successful": all(
            pool["quality_report"]["success_rate"] == 1.0
            for pool in (train_pool, validation_pool, test_pool)
        ),
        "no_initial_state_rejections": all(
            len(_read(STUDY / f"data/{name}_pool/initial_state_rejections.json")) == 0
            for name in ("train", "validation", "untouched_test")
        ),
        "canonical_agent_hash_unchanged": canonical_hash == expected_hash,
        "no_prohibited_features_in_training_metadata": all(
            not any(row["prohibited_features"].values()) for row in training["variants"]
        ),
    }
    result = {
        "schema": "double_bottleneck_initial_state_coverage_protocol_validation_v1",
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "canonical_agent_sha256": canonical_hash,
        "expected_canonical_agent_sha256": expected_hash,
        "split_initial_state_counts": {name: len(value) for name, value in hashes.items()},
        "split_intersection_counts": {
            "train_validation": len(hashes["train"] & hashes["validation"]),
            "train_test": len(hashes["train"] & hashes["untouched_test"]),
            "validation_test": len(hashes["validation"] & hashes["untouched_test"]),
        },
    }
    (STUDY / "protocol_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    comparison = _read(STUDY / "evaluation/comparison.json")
    prereg = _read(STUDY / "PREREGISTRATION.json")
    data_manifest = _read(STUDY / "data/manifest.json")
    training = _read(STUDY / "models/training_comparison.json")
    train_dataset = FlowBC4ADataset(STUDY / "data/train_pool", "train", seed=0)
    validation_dataset = FlowBC4ADataset(STUDY / "data/validation_pool", "val", seed=0)
    test_dataset = FlowBC4ADataset(STUDY / "data/untouched_test_pool", "val", seed=0)
    train_features = _unique_feature_rows(train_dataset)
    validation_features = _unique_feature_rows(validation_dataset)
    test_features = _unique_feature_rows(test_dataset)

    scale_summary = {}
    for scale in SCALES:
        evaluation = comparison["sets"]["untouched_test"][scale]
        aggregate = evaluation["aggregate"]
        support = aggregate["support"]
        scale_summary[scale] = {
            "independent_initial_states": data_manifest["variants"][scale]["initial_states"],
            "successes": aggregate["successes"],
            "rollouts": aggregate["rollouts"],
            "success_rate": aggregate["successes"] / aggregate["rollouts"],
            "wall_collisions": aggregate["wall_collisions"],
            "agent_collisions": aggregate["agent_collisions"],
            "timeouts": aggregate["timeouts"],
            "median_episode_steps": aggregate["median_episode_steps"],
            "timestep0_mean_distance": support["timestep0_mean_distance"],
            "timestep0_ood_fraction": support["timestep0_ood_fraction_unique_states"],
            "whole_rollout_mean_distance": support["whole_rollout_mean_distance"],
            "whole_rollout_ood_fraction": support["whole_rollout_ood_fraction"],
            "k100_collision_rate": evaluation["k_step"]["summary"]["100"]["collision_rate"],
            "terminal_counts": _terminal_counts(evaluation),
            "success_mode_counts": _mode_counts(evaluation),
            "by_regime": _regime_counts(evaluation),
        }

    selected = {
        scale: set(data_manifest["variants"][scale]["family_ids"])
        for scale in SCALES
    }
    recovery = {
        scale: _recovery_stats(STUDY / "data/recovery_train.npz", selected[scale])
        for scale in SCALES
    }
    result = {
        "schema": "double_bottleneck_initial_state_coverage_posthoc_v1",
        "frozen_comparison_sha256": _sha(STUDY / "evaluation/comparison.json"),
        "posthoc_only": True,
        "adaptive_data_collection_performed": False,
        "dataset_initial_distributions": {
            "train_s_large": _initial_distribution_summary(train_features),
            "validation": _initial_distribution_summary(validation_features),
            "untouched_test": _initial_distribution_summary(test_features),
        },
        "uniform_recovery": recovery,
        "coverage_scaling": scale_summary,
        "initial_variable_success_correlations": _correlations(test_features, comparison),
        "s_large_failure_divergence": _failure_divergence(
            test_dataset, comparison["sets"]["untouched_test"]["S-Large"]
        ),
    }
    (STUDY / "posthoc_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _make_manifests(prereg, data_manifest, training)
    validation = _protocol_validation(prereg, data_manifest, training)
    print(
        json.dumps(
            {
                "coverage_scaling": scale_summary,
                "failure_summary": {
                    key: value
                    for key, value in result["s_large_failure_divergence"].items()
                    if key != "rows"
                },
                "protocol_valid": validation["all_checks_pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
