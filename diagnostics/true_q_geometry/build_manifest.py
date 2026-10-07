"""Hash the frozen inputs, study code, outputs, and raw-data indexes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    return {"path": label, "sha256": digest(path), "bytes": path.stat().st_size}


def git(command):
    result = subprocess.run(["git", *command], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    source_paths = [
        ROOT / "single_integrator/environment.py",
        ROOT / "single_integrator/cbf.py",
        ROOT / "single_integrator/evaluate.py",
        ROOT / "single_integrator/outcomes.py",
        ROOT / "single_integrator/c1/differentiable_rollout.py",
        ROOT / "single_integrator/c1/models/residual.py",
        ROOT / "flowbc/giveway_flowbc_agent.py",
        ROOT / "diagnostics/cl_fhcb/closed_loop.py",
        ROOT / "diagnostics/cl_fhcb/CL_FHCB_FROZEN_SPEC.md",
    ]
    study_code = [
        HERE / "geometry_rollout.py", HERE / "run_q_map.py", HERE / "analyze_q_map.py",
        HERE / "run_direction_validation.py", HERE / "finalize_analysis.py",
        HERE / "plot_results.py", HERE / "benchmark_backend.py",
    ]
    outputs = [
        HERE / "true_q_geometry_report.md", HERE / "system_and_policy_semantics.md",
        HERE / "full_horizon_q_map.json", HERE / "directional_geometry.json",
        HERE / "escape_vs_delayed_deadlock.md", HERE / "feature_analysis.json",
        HERE / "smoothness_audit.json", HERE / "projection_role.json",
        HERE / "temporal_control_window.json",
    ] + sorted((HERE / "figures").glob("*.png"))
    raw_indexes = [
        HERE / "raw/q_map/manifest.json", HERE / "raw/q_map/state_catalog.json",
        HERE / "raw/directional_selection.json", HERE / "raw/direction_validation/manifest.json",
        HERE / "raw/escape_delay_summary.json", HERE / "raw/final_summary.json",
    ]
    checkpoint = ROOT.parent / "02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
    manifest = {
        "schema": "c1_true_q_geometry_manifest_v1",
        "created_at": "2026-09-22T14:15:00+08:00",
        "study": "diagnostic discovery of (z_t, phi) -> true full-horizon strict-deadlock probability",
        "decision": "CASE B — CONTROL GEOMETRY EXISTS BUT IS NONSMOOTH / MULTIMODAL",
        "method_changes": {
            "new_R_risk": False, "new_certificate": False, "G_phi_trained": False,
            "neural_Q_critic_trained": False, "environment_or_monitor_modified": False,
        },
        "compute": {
            "inference_new_full_continuations": 422,
            "inference_cached_full_continuations_reused": 10,
            "inference_new_physical_steps": 111271,
            "backend_benchmark_full_continuations_excluded_from_inference": 2,
            "backend_benchmark_physical_steps": 160,
            "all_new_full_continuations_executed": 424,
            "all_new_physical_steps_executed": 111431,
            "formal_runs_backend": "CPU",
            "backend_benchmark": {"cpu_seconds_for_80_steps": 0.288726590006263,
                                  "slurm_gpu_seconds_for_80_steps": 0.5633919989995775,
                                  "gpu": "NVIDIA RTX PRO 6000 Blackwell shard"},
        },
        "sampling": {
            "main_samples_per_state_phi": 4,
            "direction_validation_samples_per_state_phi": 8,
            "uncertainty": "Wilson 95% marginal intervals and paired exact McNemar/binomial tests",
            "Flow_randomness_included": True,
            "G_phi_stochastic": False,
            "common_random_numbers": "same absolute-step Flow keys within paired policies; fresh independent seed set for direction validation",
        },
        "validation": {
            "artifact_json_and_raw_npz_hash_index": "PASS",
            "deterministic_unit_tests": "5/5 PASS via unittest",
            "semantic_traces_checked": 432,
            "termination_timeout_boundary_crn_and_G_phi_formula": "PASS",
            "maximum_G_phi_formula_absolute_error": 3.722696567676209e-09,
            "minimum_recorded_second_projection_cbf_residual": -3.3478611957267717e-16,
            "projection_residual_interpretation": "machine-precision feasible",
        },
        "git": {
            "metadata_available": git(["rev-parse", "--is-inside-work-tree"]) == "true",
            "head": git(["rev-parse", "HEAD"]),
            "branch": git(["branch", "--show-current"]),
            "worktree_clean": git(["status", "--porcelain"]) == "",
            "note": "Hashes below, not HEAD alone, identify this dirty/untracked diagnostic snapshot.",
        },
        "frozen_checkpoint": record(checkpoint),
        "protocol": record(HERE / "predeclared_protocol.json"),
        "frozen_sources_and_specs": [record(x) for x in source_paths],
        "study_code": [record(x) for x in study_code],
        "raw_indexes": [record(x) for x in raw_indexes],
        "deliverables_and_figures": [record(x) for x in outputs],
        "raw_trace_integrity": "Each NPZ is individually SHA256-indexed by its raw stage manifest.",
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "decision": manifest["decision"], "head": manifest["git"]["head"],
        "outputs": len(outputs), "manifest_sha256": digest(HERE / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
