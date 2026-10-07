#!/usr/bin/env python3
"""Pre-register the eta domain, populations, samples, and Stage-A jobs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"
HARD = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"
CONTROL_SEED = 832041
SOBOL_SEED = 351903
DOMAIN_LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
DOMAIN_HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)
TOY_ANCHOR = np.asarray((1.0, 0.0, 0.25), dtype=np.float64)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def episode_row(row, population):
    return {
        "episode_id": f"{row['set']}|{row['rollout_id']:03d}",
        "population": population,
        "set": row["set"],
        "family_id": row["family_id"],
        "regime": row["regime"],
        "seed": row["seed"],
        "rollout_id": row["rollout_id"],
        "baseline_outcome": row["outcome"],
        "baseline_steps": row["episode_steps"],
        "baseline_first_direction": row["coordination_mode"]["first_direction"],
    }


def main() -> int:
    prereg = STUDY / "PREREGISTRATION.json"
    if prereg.exists():
        raise FileExistsError(prereg)
    all_rows = []
    hard_paths = []
    for set_name in ("existing_untouched_test", "fresh_untouched_test"):
        path = HARD / f"hard_safety_{set_name}_outcomes.json"
        hard_paths.append(path)
        all_rows.extend(json.loads(path.read_text())["rollouts"])
    targets = [episode_row(row, "safe_timeout_target") for row in all_rows if row["outcome"] == "timeout"]
    successes = [row for row in all_rows if row["outcome"] == "success"]
    if len(targets) != 61 or len(successes) != 131:
        raise AssertionError("frozen hard-safety population mismatch")
    rng = np.random.default_rng(CONTROL_SEED)
    selected = np.sort(rng.choice(len(successes), size=24, replace=False))
    controls = [episode_row(successes[index], "baseline_success_control") for index in selected]
    episodes = {
        "schema": "double_bottleneck_eta3_episode_catalog_v1",
        "control_selection": {
            "rule": "Uniform random sample without replacement from all 131 frozen hard-safety successes, before eta evaluation.",
            "pcg64_seed": CONTROL_SEED,
            "selected_indices_in_sorted_frozen_row_order": selected.tolist(),
        },
        "targets": targets,
        "controls": controls,
    }
    episode_path = STUDY / "episode_catalog.json"
    write_new(episode_path, episodes)

    unit = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED).random_base2(m=6)
    scaled = qmc.scale(unit, DOMAIN_LOW, DOMAIN_HIGH)
    stage_a = [
        {"eta_id": f"A{index:03d}", "eta": scaled[index].tolist(), "sample_type": "stage_a_sobol"}
        for index in range(32)
    ]
    stage_a.append(
        {"eta_id": "A032", "eta": TOY_ANCHOR.tolist(), "sample_type": "toy_reference_anchor"}
    )
    dense = [
        {"eta_id": f"B{index:03d}", "eta": scaled[index + 32].tolist(), "sample_type": "stage_b_dense_sobol"}
        for index in range(32)
    ]
    samples = {
        "schema": "double_bottleneck_eta3_samples_v1",
        "domain": {
            "coordinate_order": ["goal_feedback", "safe_feedback", "relative_feedback"],
            "lower": DOMAIN_LOW.tolist(),
            "upper": DOMAIN_HIGH.tolist(),
            "source": "Exact closed box of the frozen Toy Give-Way SBMA Phase-A lattice; relation-basis mean preserves the per-agent 0.5 scale at N=4.",
        },
        "sobol": {
            "implementation": "scipy.stats.qmc.Sobol(d=3, scramble=True).random_base2(m=6)",
            "seed": SOBOL_SEED,
            "nested_points": 64,
        },
        "zero_sentinel": {"eta_id": "ZERO", "eta": [0.0, 0.0, 0.0], "sample_type": "baseline_sentinel", "in_domain": False},
        "stage_a_global": stage_a,
        "stage_b_dense_global": dense,
    }
    sample_path = STUDY / "eta_samples.json"
    write_new(sample_path, samples)

    frozen_hashes = {
        "checkpoint": sha(ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"),
        "dataset_manifest": sha(ROOT / "diagnostics/double_bottleneck_recovery_density_final/data/manifest.json"),
        "macflow_source": sha(ROOT / "double_bottleneck/flowbc_4a_agent.py"),
        "environment": sha(ROOT / "double_bottleneck/environment.py"),
        "hard_projection": sha(ROOT / "shared_control/hard_projection.py"),
        "diagnostic_corrector": sha(ROOT / "shared_control/diagnostic_corrector.py"),
        "evaluator": sha(STUDY / "tools/run_jobs.py"),
        "hard_safety_existing_outcomes": sha(hard_paths[0]),
        "hard_safety_fresh_outcomes": sha(hard_paths[1]),
        "episode_catalog": sha(episode_path),
        "eta_samples": sha(sample_path),
        "toy_sbma_setup": sha(ROOT / "diagnostics/success_basin_multimodality/setup.py"),
        "toy_sbma_protocol": sha(ROOT / "diagnostics/success_basin_multimodality/protocol.json"),
    }
    protocol = {
        "schema": "double_bottleneck_eta3_basin_preregistration_v1",
        "registered_at": datetime.now().astimezone().isoformat(),
        "scientific_question": "Does the unchanged episode-level 3D eta family admit meaningful success basins for the 61 frozen safe-timeout episodes?",
        "frozen_pipeline": {
            "formula": "u_flow=MACFlow(x); u_safe=Pi_U(x)(u_flow); g=eta1*B_goal+eta2*u_safe+eta3*B_rel; u_exec=Pi_U(x)(u_safe+g)",
            "eta_fixed_for_episode": True,
            "basis_recomputed_each_step": True,
            "second_projection_required": True,
            "horizon_steps": 850,
            "dt_seconds": 0.05,
            "success": "All four agents simultaneously reach their 0.08 m goal regions before runtime termination.",
        },
        "basis": {
            "B_goal": "bound_0.5(goal_i-position_i) per agent",
            "B_safe": "first-projected u_safe_i",
            "B_rel": "(1/(N-1))*sum_{j!=i} bound_0.5(position_i-position_j), with pairwise bound before mean",
            "bound": "radial direction-preserving row bound",
            "raw_g_clipping": "none",
            "n2_reduction": "exact single-opponent bounded relative vector",
        },
        "domain": samples["domain"],
        "domain_reasoning": {
            "toy_values": {"goal": [0.5, 0.75, 1.0, 1.25], "safe": [-0.5, -0.25, 0.0, 0.25, 0.5], "relative": [0.0, 0.25, 0.5, 0.75]},
            "scale_check": "Each B_goal, u_safe, and averaged B_rel row has norm <=0.5. The relation term therefore retains the Toy per-agent scale despite N=4.",
            "maximum_raw_row_norm_bound": 1.25,
            "eta_zero": "Evaluated as an out-of-domain baseline sentinel and excluded from basin-volume denominators.",
            "no_case_specific_range": True,
        },
        "populations": {
            "targets": 61,
            "target_rule": "Every frozen hard-safety timeout across both untouched sets.",
            "controls": 24,
            "control_rule": episodes["control_selection"],
        },
        "stage_a": {
            "global_points": 33,
            "design": "32 scrambled Sobol points plus the frozen Toy reference anchor (1,0,0.25)",
            "same_points_every_episode": True,
            "zero_sentinel_every_episode": True,
            "basin_fraction_denominator": 33,
        },
        "stage_b": {
            "if_stage_a_success": "Refine the successful Stage-A point nearest the Toy anchor in domain-normalized Euclidean distance, tie by eta_id.",
            "local_offsets": "14 normalized offsets: +/-1/16 on each coordinate axis and all 8 cube corners; clip to the same global box and remove duplicates.",
            "if_no_stage_a_success": "Evaluate the next 32 nested Sobol points over the identical global box.",
            "if_dense_global_first_finds_success": "Apply the same 14-point local rule around the dense success nearest the Toy anchor.",
            "no_domain_expansion": True,
        },
        "basin_geometry": {
            "normalized_connectivity_radius": 0.35,
            "warning": "Sample graph components are resolution-dependent and are not a topological proof.",
            "type_i_broad": "Stage-A rho>=0.20, local success fraction>=0.50, and largest Stage-A success component contains >=75% of Stage-A successes.",
            "type_iii_fragmented": "At least two Stage-A success components, each containing at least two points.",
            "type_ii_narrow": "Observed success but neither Type I nor Type III.",
            "type_iv_none": "No success after Stage A and the additional 32-point dense global sample.",
        },
        "correction_metrics": {
            "g_norm": "Joint 8D Euclidean norm per step.",
            "second_projection_correction": "Joint norm ||u_exec-(u_safe+g)||.",
            "projection_active_threshold": 1e-6,
            "substantially_altered": "Second-projection correction >=25% of ||g|| when ||g||>1e-6.",
            "mostly_rewritten": "Second-projection correction >=75% of ||g|| when ||g||>1e-6.",
        },
        "cross_state": {
            "coverage": "For each shared Stage-A eta, number of 61 targets rescued and number of 24 controls preserved.",
            "pairwise_overlap": "Jaccard overlap of Stage-A success membership vectors.",
            "representative_eta": "Successful tested eta with highest local success fraction, then smallest normalized distance to Toy anchor, then eta_id.",
            "no_deployment_optimization": True,
        },
        "deadlock_language": "Only runtime strict deadlock is recorded. The rejected shadow detector is not evaluated or used as ground truth.",
        "prohibitions": {
            "g_phi_training": False,
            "eta_dimension_change": False,
            "time_or_state_dependent_eta": False,
            "basis_redesign": False,
            "horizon_or_success_change": False,
            "per_case_domain_or_adaptive_expansion": False,
        },
        "frozen_hashes": frozen_hashes,
    }
    write_new(prereg, protocol)

    stage_a_jobs = []
    sample_rows = [samples["zero_sentinel"]] + stage_a
    for episode in targets + controls:
        for eta_row in sample_rows:
            stage_a_jobs.append(
                {
                    **episode,
                    "stage": "stage_a",
                    "sample_type": eta_row["sample_type"],
                    "eta_id": eta_row["eta_id"],
                    "eta": eta_row["eta"],
                    "job_id": f"stage_a|{episode['episode_id']}|{eta_row['eta_id']}",
                }
            )
    job_manifest = {
        "schema": "double_bottleneck_eta3_jobs_v1",
        "stage": "stage_a",
        "preregistration_sha256": sha(prereg),
        "jobs": stage_a_jobs,
    }
    write_new(STUDY / "jobs/stage_a.json", job_manifest)
    print(
        json.dumps(
            {
                "preregistration_sha256": sha(prereg),
                "targets": len(targets),
                "controls": len(controls),
                "stage_a_eta_points_plus_zero": len(sample_rows),
                "stage_a_jobs": len(stage_a_jobs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
