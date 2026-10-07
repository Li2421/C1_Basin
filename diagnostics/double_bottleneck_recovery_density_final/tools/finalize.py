#!/usr/bin/env python3
"""Create frozen summaries, protocol checks, and compact density figures."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
REFERENCE = ROOT / "diagnostics/double_bottleneck_sxl_baseline_maturation"


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(result):
    a, s, t = result["aggregate"], result["aggregate"]["support"], result["teacher_forced"]
    return {
        "success": a["successes"], "rollouts": a["rollouts"], "success_rate": a["successes"] / a["rollouts"],
        "wall_collision": a["wall_collisions"], "agent_collision": a["agent_collisions"], "timeout": a["timeouts"], "deadlock": a["deadlocks"],
        "mean_episode_steps": a["mean_episode_steps"], "median_episode_steps": a["median_episode_steps"],
        "minimum_wall_clearance": a["minimum_wall_clearance"], "minimum_pair_clearance": a["minimum_pair_clearance"],
        "x0_mean_distance": s["timestep0_mean_distance"], "x0_ood": s["timestep0_ood_fraction_unique_states"], "rollout_ood": s["whole_rollout_ood_fraction"],
        "teacher_rmse": t["overall"]["joint_action_rmse"], "waiting_transition_rmse": t["waiting_transition"]["joint_action_rmse"],
        "goal_region_rmse": t["goal_region"]["joint_action_rmse"], "wall_directed_error": t["overall"]["wall_directed_absolute_error"],
        "k100_position_rmse": result["k_step"]["summary"]["100"]["position_rmse"], "k100_collision": result["k_step"]["summary"]["100"]["collision_rate"],
    }


def direction(result):
    good = [row for row in result["rollouts"] if row["success"]]
    counts = Counter(row["coordination_mode"]["first_direction"] for row in good)
    signatures = Counter(row["coordination_mode"]["signature"] for row in good)
    return {"ltr_first": counts["left_to_right"], "rtl_first": counts["right_to_left"], "ambiguous_or_mixed": len(good)-sum(counts.values()), "unique_complete_order_signatures": len(signatures), "by_regime": dict(sorted(Counter(row["regime"] for row in good).items()))}


def main():
    prereg = read(STUDY / "PREREGISTRATION.json")
    data = read(STUDY / "data/manifest.json")
    train = read(STUDY / "model/training_summary.json")
    new = read(STUDY / "evaluation/comparison.json")
    old = read(REFERENCE / "evaluation/comparison.json")
    sets = ("development", "existing_untouched_test", "fresh_untouched_test")
    summary = {
        "schema": "double_bottleneck_recovery_density_final_summary_v1",
        "reference_sxl64": {name: metric(old["sets"][name]) for name in sets},
        "sxl128": {name: metric(new["sets"][name]) for name in sets},
        "direction": {"sxl64": {name: direction(old["sets"][name]) for name in sets}, "sxl128": {name: direction(new["sets"][name]) for name in sets}},
        "dataset": data, "training": train,
    }
    (STUDY / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with np.load(REFERENCE / "data/recovery_train.npz", allow_pickle=False) as old_rows, np.load(STUDY / "data/recovery_128_train.npz", allow_pickle=False) as all_rows:
        old_count = len(old_rows["actions"])
        prefix_equal = all(np.array_equal(all_rows[key][:old_count], old_rows[key]) for key in old_rows.files if old_rows[key].ndim > 0)
        source_pairs = list(zip(all_rows["source_rollout_id"].astype(str), all_rows["source_step"].astype(int), strict=True))
        counts = Counter(rollout for rollout, _ in source_pairs)
        distinct = len(source_pairs) == len(set(source_pairs))
    canonical = ROOT / "double_bottleneck/flowbc_4a_agent.py"
    checks = {
        "prereg_reference_data_hash_matches": prereg["reference"]["data_manifest_sha256"] == sha(REFERENCE / "data/manifest.json"),
        "prereg_reference_training_hash_matches": prereg["reference"]["training_summary_sha256"] == sha(REFERENCE / "model/training_summary.json"),
        "prereg_reference_evaluation_hash_matches": prereg["reference"]["evaluation_comparison_sha256"] == sha(REFERENCE / "evaluation/comparison.json"),
        "data_uses_exact_reference_nominal_pool": data["reference_train_pool_sha256"] == sha(REFERENCE / "data/train_pool/manifest.json"),
        "all_64_anchor_rows_retained_bytewise": prefix_equal,
        "each_trajectory_has_128_distinct_source_steps": len(counts) == 2304 and set(counts.values()) == {128} and distinct,
        "no_old_new_anchor_overlap": data["combined_recovery_128"]["overlapping_source_steps"] == 0,
        "new_extra_rows_are_globally_balanced_by_direction": data["extra_recovery_64"]["first_direction_counts"] == {"left_to_right": 73728, "right_to_left": 73728},
        "same_nominal_trajectories": data["same_nominal_trajectories"],
        "same_eight_mode_balance": data["same_8_mode_balance"],
        "no_phase_or_failure_dependent_sampling": not data["failure_or_phase_dependent_sampling"],
        "canonical_agent_hash_unchanged": sha(canonical) == prereg["reference"]["canonical_agent_sha256"],
        "no_prohibited_training_feature": not any(train["prohibited_features"].values()),
        "untouched_test_not_used_for_training": not train["untouched_test_loaded_or_used"],
        "evaluation_checkpoint_is_selected": new["selected_checkpoint_sha256"] == train["selected_checkpoint_sha256"],
        "reference_evaluation_hash_recorded": new["reference_sxl64_comparison_sha256"] == sha(REFERENCE / "evaluation/comparison.json"),
    }
    protocol = {"schema": "double_bottleneck_recovery_density_final_protocol_validation_v1", "checks": checks, "all_checks_pass": all(checks.values()), "toy_modified_by_study": False, "safety_eta_or_g_phi_started": False}
    (STUDY / "protocol_validation.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    manifest = {"schema": "double_bottleneck_recovery_density_final_manifest_v1", "preregistration": {"path": str((STUDY/"PREREGISTRATION.json").resolve()), "sha256": sha(STUDY/"PREREGISTRATION.json")}, "data": {"path": str((STUDY/"data/manifest.json").resolve()), "sha256": sha(STUDY/"data/manifest.json")}, "training": {"path": str((STUDY/"model/training_summary.json").resolve()), "sha256": sha(STUDY/"model/training_summary.json")}, "checkpoint": {"path": train["selected_checkpoint"], "sha256": train["selected_checkpoint_sha256"]}, "evaluation": {"path": str((STUDY/"evaluation/comparison.json").resolve()), "sha256": sha(STUDY/"evaluation/comparison.json")}, "summary": {"path": str((STUDY/"summary.json").resolve()), "sha256": sha(STUDY/"summary.json")}, "protocol": {"path": str((STUDY/"protocol_validation.json").resolve()), "sha256": sha(STUDY/"protocol_validation.json")}}
    (STUDY / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    figures = STUDY / "figures"; figures.mkdir(exist_ok=True)
    old_existing, new_existing = summary["reference_sxl64"]["existing_untouched_test"], summary["sxl128"]["existing_untouched_test"]
    old_fresh, new_fresh = summary["reference_sxl64"]["fresh_untouched_test"], summary["sxl128"]["fresh_untouched_test"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4)); labels=["existing", "fresh"]; x=np.arange(2)
    ax[0].bar(x-.18, [100*old_existing["success_rate"],100*old_fresh["success_rate"]], .36, label="64 anchors")
    ax[0].bar(x+.18, [100*new_existing["success_rate"],100*new_fresh["success_rate"]], .36, label="128 anchors")
    ax[0].set_xticks(x,labels); ax[0].set_ylabel("full-task success (%)"); ax[0].set_ylim(0,100); ax[0].legend(frameon=False); ax[0].grid(axis="y",alpha=.25)
    ax[1].bar(x-.18, [100*old_existing["rollout_ood"],100*old_fresh["rollout_ood"]], .36, label="64 anchors")
    ax[1].bar(x+.18, [100*new_existing["rollout_ood"],100*new_fresh["rollout_ood"]], .36, label="128 anchors")
    ax[1].set_xticks(x,labels); ax[1].set_ylabel("rollout OOD (%)"); ax[1].legend(frameon=False); ax[1].grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "density_comparison.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,3.4)); names=["64 existing","128 existing","64 fresh","128 fresh"]; vals=[summary["direction"]["sxl64"]["existing_untouched_test"],summary["direction"]["sxl128"]["existing_untouched_test"],summary["direction"]["sxl64"]["fresh_untouched_test"],summary["direction"]["sxl128"]["fresh_untouched_test"]]; ltr=[x["ltr_first"] for x in vals]; rtl=[x["rtl_first"] for x in vals]; idx=np.arange(4); ax.bar(idx,ltr,label="LTR-first"); ax.bar(idx,rtl,bottom=ltr,label="RTL-first"); ax.set_xticks(idx,names,rotation=18,ha="right"); ax.set_ylabel("successful rollouts"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figures / "direction_outcomes.svg"); plt.close(fig)
    print(json.dumps({"all_checks_pass": protocol["all_checks_pass"], "summary": str((STUDY/"summary.json").resolve())}, indent=2))


if __name__ == "__main__": main()
