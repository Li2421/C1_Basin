#!/usr/bin/env python3
"""Create protocol validation, final manifest, numerical summary, and figures."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_hashes(path):
    manifest = read(path)
    return {
        row["initial_state_sha256"]
        for row in manifest["mode_analysis"]["multimodal_initial_state_groups"]
    }


def metrics(result):
    aggregate = result["aggregate"]
    support = aggregate["support"]
    return {
        "rollouts": aggregate["rollouts"],
        "success": aggregate["successes"],
        "wall_collision": aggregate["wall_collisions"],
        "agent_collision": aggregate["agent_collisions"],
        "timeout": aggregate["timeouts"],
        "deadlock": aggregate["deadlocks"],
        "success_rate": aggregate["successes"] / aggregate["rollouts"],
        "mean_episode_steps": aggregate["mean_episode_steps"],
        "median_episode_steps": aggregate["median_episode_steps"],
        "minimum_wall_clearance": aggregate["minimum_wall_clearance"],
        "minimum_pair_clearance": aggregate["minimum_pair_clearance"],
        "x0_mean_distance": support["timestep0_mean_distance"],
        "x0_ood": support["timestep0_ood_fraction_unique_states"],
        "rollout_mean_distance": support["whole_rollout_mean_distance"],
        "rollout_ood": support["whole_rollout_ood_fraction"],
        "teacher_rmse": result["teacher_forced"]["overall"]["joint_action_rmse"],
        "wall_directed_error": result["teacher_forced"]["overall"]["wall_directed_absolute_error"],
        "waiting_transition_rmse": result["teacher_forced"]["waiting_transition"]["joint_action_rmse"],
        "goal_region_rmse": result["teacher_forced"]["goal_region"]["joint_action_rmse"],
        "k_step": result["k_step"]["summary"],
    }


def main():
    prereg = read(STUDY / "PREREGISTRATION.json")
    data = read(STUDY / "data/manifest.json")
    training = read(STUDY / "model/training_summary.json")
    stability = read(STUDY / "checkpoint_stability.json")
    evaluation = read(STUDY / "evaluation/comparison.json")
    parent = read(ROOT / "diagnostics/double_bottleneck_initial_state_coverage/evaluation/comparison.json")

    scales = {}
    for name, count in (("S-Small", 24), ("S-Medium", 72), ("S-Large", 144)):
        scales[name] = {"initial_states": count, **metrics(parent["sets"]["untouched_test"][name])}
    scales["S-XL"] = {
        "initial_states": 288,
        **metrics(evaluation["sets"]["existing_untouched_test"]),
    }
    summary = {
        "schema": "double_bottleneck_sxl_final_summary_v1",
        "dataset": {
            "initial_states": data["variant"]["initial_states"],
            "expert_trajectories": data["variant"]["expert_trajectories"],
            "nominal_transitions": data["variant"]["nominal_transitions"],
            "recovery_transitions": data["variant"]["recovery_transitions"],
            "total_transitions": data["variant"]["total_transitions"],
            "states_per_regime": data["variant"]["states_per_regime"],
            "mode_counts": data["train_pool"]["actual_mode_counts"],
            "episode_length": data["train_pool"]["episode_length"],
            "recovery_phase_counts": data["recovery"]["phase_counts"],
            "recovery_attempted": data["recovery"]["attempted"],
            "recovery_accepted": data["recovery"]["accepted"],
            "recovery_rejected": data["recovery"]["rejected"],
        },
        "training": training,
        "checkpoint_stability": {
            key: {
                "step": value["step"],
                "development_success": value["aggregate"]["successes"],
                "development_wall_collision": value["aggregate"]["wall_collisions"],
                "development_agent_collision": value["aggregate"]["agent_collisions"],
                "development_timeout": value["aggregate"]["timeouts"],
                "teacher_rmse": value["teacher_forced"]["overall"]["joint_action_rmse"],
            }
            for key, value in stability["checkpoints"].items()
        },
        "scaling_existing_untouched": scales,
        "sxl_development": metrics(evaluation["sets"]["development"]),
        "sxl_existing_untouched": metrics(evaluation["sets"]["existing_untouched_test"]),
        "sxl_fresh_untouched": metrics(evaluation["sets"]["fresh_untouched_test"]),
    }
    (STUDY / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    train_manifest = STUDY / "data/train_pool/manifest.json"
    fresh_manifest = STUDY / "data/fresh_test_pool/manifest.json"
    validation_manifest = ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/validation_pool/manifest.json"
    existing_manifest = ROOT / "diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool/manifest.json"
    pools = {
        "train": initial_hashes(train_manifest),
        "validation": initial_hashes(validation_manifest),
        "existing_test": initial_hashes(existing_manifest),
        "fresh_test": initial_hashes(fresh_manifest),
    }
    with np.load(STUDY / "data/recovery_train.npz", allow_pickle=False) as archive:
        source_counts = Counter(archive["source_rollout_id"].astype(str).tolist())
    canonical = ROOT / "double_bottleneck/flowbc_4a_agent.py"
    expected_canonical = prereg["training"]["canonical_agent_sha256"]
    checks = {
        "preregistration_matches_data": data["preregistration_sha256"] == sha(STUDY / "PREREGISTRATION.json"),
        "preregistration_matches_training": training["preregistration_sha256"] == sha(STUDY / "PREREGISTRATION.json"),
        "fresh_test_generated_before_training": data["fresh_test_generated_before_training"],
        "train_validation_disjoint": not (pools["train"] & pools["validation"]),
        "train_existing_test_disjoint": not (pools["train"] & pools["existing_test"]),
        "train_fresh_test_disjoint": not (pools["train"] & pools["fresh_test"]),
        "validation_existing_test_disjoint": not (pools["validation"] & pools["existing_test"]),
        "validation_fresh_test_disjoint": not (pools["validation"] & pools["fresh_test"]),
        "existing_fresh_test_disjoint": not (pools["existing_test"] & pools["fresh_test"]),
        "all_2304_trajectories_have_64_recovery_anchors": len(source_counts) == 2304 and set(source_counts.values()) == {64},
        "eight_modes_each_have_288_trajectories": set(data["train_pool"]["actual_mode_counts"].values()) == {288},
        "canonical_macflow_hash_unchanged": sha(canonical) == expected_canonical,
        "no_prohibited_feature_enabled": not any(training["prohibited_features"].values()),
        "untouched_test_not_loaded_during_training": not training["untouched_test_loaded_or_used"],
        "checkpoint_selection_not_changed_by_stability_audit": not stability["checkpoint_selection_changed_by_this_audit"],
        "evaluation_checkpoint_is_preregistered_selected": evaluation["selected_checkpoint_sha256"] == training["selected_checkpoint_sha256"],
    }
    protocol = {
        "schema": "double_bottleneck_sxl_protocol_validation_v1",
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "pool_sizes": {key: len(value) for key, value in pools.items()},
        "canonical_macflow_sha256": sha(canonical),
        "toy_modified_by_study": False,
        "test_failures_used_for_training": False,
    }
    (STUDY / "protocol_validation.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    manifest = {
        "schema": "double_bottleneck_sxl_final_manifest_v1",
        "preregistration": {"path": str((STUDY / "PREREGISTRATION.json").resolve()), "sha256": sha(STUDY / "PREREGISTRATION.json")},
        "dataset": {"path": str((STUDY / "data/manifest.json").resolve()), "sha256": sha(STUDY / "data/manifest.json")},
        "training": {"path": str((STUDY / "model/training_summary.json").resolve()), "sha256": sha(STUDY / "model/training_summary.json")},
        "checkpoint": {"path": training["selected_checkpoint"], "sha256": training["selected_checkpoint_sha256"]},
        "evaluation": {"path": str((STUDY / "evaluation/comparison.json").resolve()), "sha256": sha(STUDY / "evaluation/comparison.json")},
        "posthoc": {"path": str((STUDY / "posthoc_failure_analysis.json").resolve()), "sha256": sha(STUDY / "posthoc_failure_analysis.json")},
        "protocol_validation": {"path": str((STUDY / "protocol_validation.json").resolve()), "sha256": sha(STUDY / "protocol_validation.json")},
        "initial_state_stream": prereg["seed_scheme"],
        "recovery_anchor_seed_protocol": prereg["uniform_recovery"]["anchor_rule"],
        "recovery_perturbation_seed_protocol": prereg["uniform_recovery"]["perturbation_seed"],
        "expert_trajectory_ids": data["variant"]["expert_trajectory_ids"],
        "family_ids": data["variant"]["family_ids"],
    }
    (STUDY / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    figures = STUDY / "figures"
    figures.mkdir(exist_ok=True)
    labels = ["24", "72", "144", "288"]
    values = [scales[name] for name in ("S-Small", "S-Medium", "S-Large", "S-XL")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
    axes[0].plot(labels, [v["x0_mean_distance"] for v in values], marker="o")
    axes[0].set_ylabel("mean x0 support distance")
    axes[1].plot(labels, [100 * v["rollout_ood"] for v in values], marker="o")
    axes[1].set_ylabel("rollout OOD (%)")
    axes[2].plot(labels, [100 * v["success_rate"] for v in values], marker="o")
    axes[2].set_ylabel("full-task success (%)")
    for axis in axes:
        axis.set_xlabel("independent training states")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "coverage_scaling.svg")
    plt.close(fig)

    existing = summary["sxl_existing_untouched"]
    fresh = summary["sxl_fresh_untouched"]
    names = ["success", "wall_collision", "agent_collision", "timeout"]
    x = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(7.0, 3.6))
    axis.bar(x - 0.18, [existing[k] for k in names], width=0.36, label="existing untouched")
    axis.bar(x + 0.18, [fresh[k] for k in names], width=0.36, label="fresh untouched")
    axis.set_xticks(x, ["success", "wall", "agent", "timeout"])
    axis.set_ylabel("rollouts (of 96)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "sxl_outcomes.svg")
    plt.close(fig)

    metrics_rows = [json.loads(line) for line in (STUDY / "model/metrics.jsonl").read_text().splitlines() if line.strip()]
    fig, axis = plt.subplots(figsize=(6.7, 3.6))
    axis.plot([row["step"] for row in metrics_rows], [row["batch_loss"] for row in metrics_rows], label="minibatch train loss", alpha=0.8)
    axis.plot([row["step"] for row in metrics_rows], [row["fixed_val_loss"] for row in metrics_rows], label="fixed validation", alpha=0.9)
    milestone_steps = []
    milestone_train = []
    milestone_val = []
    for key, value in stability["checkpoints"].items():
        milestone_steps.append(value["step"])
        milestone_train.append(value.get("fixed_train_loss", np.nan))
        milestone_val.append(value.get("fixed_validation_loss", np.nan))
    axis.set_xlabel("optimizer updates")
    axis.set_ylabel("flow-matching loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "training_convergence.svg")
    plt.close(fig)
    print(json.dumps({"all_checks_pass": protocol["all_checks_pass"], "summary": str(STUDY / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
