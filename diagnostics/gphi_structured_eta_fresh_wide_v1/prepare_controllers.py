"""Lock all controller and implementation assets before fresh rollout."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
MANIFEST = HERE / "fresh_wide_manifest.json"
OVERLAP = HERE / "overlap_audit.json"
FLOW = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
DIRECT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
ETA = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
NORMALIZATION = ROOT / "diagnostics/gphi_training_startup_complete_v1/artifacts/normalization.json"
EXPECTED = {
    "flow": "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32",
    "direct_g": "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e",
    "structured_eta": "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095",
}

sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(WIDE))
from pilot_common import assert_frozen_sources, canonical_json_hash, sha256  # noqa: E402
import run_evaluation as wide_runner  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    if not MANIFEST.is_file() or not OVERLAP.is_file():
        raise RuntimeError("fresh manifest must be frozen first")
    if list((HERE / "runs").rglob("episode_*.json")) if (HERE / "runs").exists() else []:
        raise RuntimeError("controller lock must precede rollout")
    manifest = json.loads(MANIFEST.read_text())
    overlap = json.loads(OVERLAP.read_text())
    if overlap["status"] != "PASS" or manifest["raw_rollout_records_present_at_freeze"] != 0:
        raise RuntimeError("fresh/overlap gate failed")
    for key, path in (("flow", FLOW), ("direct_g", DIRECT), ("structured_eta", ETA)):
        if sha256(path) != EXPECTED[key]:
            raise RuntimeError((key, sha256(path), EXPECTED[key]))
    frozen = assert_frozen_sources()
    extra = wide_runner._audit_extra_sources()
    from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
    from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import barrier_constraints
    from single_integrator.environment import GiveWayEnv
    from single_integrator.outcomes import first_event

    eta_model = FixedDEtaPredictor(ETA)
    resolved = {
        "environment": str(Path(inspect.getsourcefile(GiveWayEnv)).resolve()),
        "projection_constraints": str(Path(inspect.getsourcefile(barrier_constraints)).resolve()),
        "retry_projector": str(Path(inspect.getsourcefile(project_velocity_with_retry)).resolve()),
        "feature_builder": str(Path(inspect.getsourcefile(StartupAwareFeatureBuilder)).resolve()),
        "eta_basis": str(Path(inspect.getsourcefile(DiagnosticPhi)).resolve()),
        "eta_corrector": str(Path(inspect.getsourcefile(DiagnosticCorrector)).resolve()),
        "outcome_classifier": str(Path(inspect.getsourcefile(first_event)).resolve()),
    }
    payload = {
        "schema": "structured_eta_final_wide_controllers_v1",
        "fresh_wide_manifest": str(MANIFEST), "fresh_wide_manifest_sha256": sha256(MANIFEST),
        "fresh_wide_manifest_content_sha256": manifest["content_sha256"],
        "overlap_audit": str(OVERLAP), "overlap_audit_sha256": sha256(OVERLAP),
        "flowbc": {"path": str(FLOW), "sha256": sha256(FLOW)},
        "direct_g": {
            "path": str(DIRECT), "sha256": sha256(DIRECT), "architecture": [214, 128, 128, 4],
            "semantics": "query/apply one step iff global_t mod 8 == 0; otherwise u_exec=u_safe; no hold",
            "normalization": str(NORMALIZATION), "normalization_sha256": sha256(NORMALIZATION),
        },
        "structured_eta": {
            "path": str(ETA), "sha256": sha256(ETA), "architecture": [214, 128, 128, 3],
            "eta_low": eta_model.low.tolist(), "eta_high": eta_model.high.tolist(),
            "semantics": "predict eta once at episode step 0, clip to frozen bounds, freeze for episode; recompute dense basis feedback every step",
        },
        "controllers": ["Safety", "Direct-g H8", "Structured eta"],
        "flow_root_seed": manifest["flow_randomness"]["root_seed"],
        "flow_key_semantics": manifest["flow_randomness"]["semantics"],
        "environment": manifest["environment"], "cbf": manifest["cbf"],
        "frozen_sources": frozen, "extra_frozen_sources": extra, "resolved_sources": resolved,
        "runner": str((HERE / "run_evaluation.py").resolve()),
        "runner_sha256": sha256(HERE / "run_evaluation.py"),
        "forbidden": {
            "training": False, "fine_tuning": False, "eta_search": False,
            "online_oracle": False, "gate": False, "cadence_search": False,
            "post_hoc_model_selection": False,
        },
        "locked_before_rollout": True,
    }
    payload["content_sha256"] = canonical_json_hash(payload)
    atomic_json(HERE / "controller_manifest.json", payload)
    atomic_json(HERE / "resource_audit.json", {
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "gpu_observation": "RTX PRO 6000, 2 MiB used, 0% utilization; no compute process",
        "scheduler_observation": "empty queue",
        "cpu_load": "0.16/3.14/6.94 on 24 logical CPUs",
        "memory_observation": "4.9 GiB used / 125 GiB total; 120 GiB available",
        "other_active_gpu_users": False, "late_night_local": True,
        "requested_shards": 6, "requested_cpu_cores": 12, "requested_memory_gb": 80,
    })
    print(json.dumps({"status": "LOCKED", "controller_manifest_sha256": sha256(HERE / "controller_manifest.json"), "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
