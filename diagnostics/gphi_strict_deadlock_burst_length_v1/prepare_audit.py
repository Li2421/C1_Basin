"""Freeze G_phi burst inputs and fail closed on deployment provenance."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1"
ORACLE_BURST = ROOT / "diagnostics/strict_deadlock_oracle_burst_length_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
CHECKPOINT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
NORMALIZATION = ROOT / "diagnostics/gphi_training_startup_complete_v1/artifacts/normalization.json"
EXPECTED_CHECKPOINT = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
sys.path.insert(0, str(PILOT))

from pilot_common import (  # noqa: E402
    DeterministicGphi, assert_frozen_sources, audit_startup_training_artifacts, sha256,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    oracle_manifest = json.loads((ORACLE_BURST / "manifest.json").read_text())
    oracle_integrity = json.loads((ORACLE_BURST / "integrity_audit.json").read_text())
    if oracle_manifest.get("status") != "COMPLETE" or oracle_integrity.get("status") != "PASS":
        raise RuntimeError("oracle burst audit is not complete/pass")
    if oracle_manifest.get("classification") != "LONG_BURST_REQUIRED":
        raise RuntimeError("unexpected oracle burst classification")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise RuntimeError("original G_phi checkpoint hash mismatch")
    frozen_sources = assert_frozen_sources()
    training_audit = audit_startup_training_artifacts(
        CHECKPOINT, NORMALIZATION, require_startup_complete=True,
    )
    model = DeterministicGphi(CHECKPOINT, NORMALIZATION)
    if model.sha256 != EXPECTED_CHECKPOINT:
        raise RuntimeError("loaded G_phi differs from frozen original")
    oracle_source = json.loads((ORACLE_BURST / "source_manifest.json").read_text())
    states = oracle_source["states"]
    etas = oracle_source["etas"]
    if len(states) != 17 or len(etas) != 17:
        raise RuntimeError("expected 17 aligned states/etas")
    for state in states:
        path = Path(state["state_file"])
        if sha256(path) != state["state_sha256"]:
            raise RuntimeError(f"source state changed: {state['state_id']}")
    with np.load(CHECKPOINT, allow_pickle=False) as values:
        architecture = json.loads(str(values["architecture_json"].item()))
    source_manifest = {
        "schema": "gphi_strict_deadlock_burst_source_v1",
        "source_oracle_burst_directory": str(ORACLE_BURST),
        "source_oracle_burst_manifest_sha256": sha256(ORACLE_BURST / "manifest.json"),
        "source_oracle_burst_classification": oracle_manifest["classification"],
        "states": states, "etas_for_diagnostic_only": etas,
        "state_count": 17, "historical": 11, "fresh_unseen": 6,
        "starting_rule": "same earliest robust oracle-capacity state as oracle burst audit",
        "teacher_diagnostic_rule": "evaluate each source state's already-frozen eta on the G_phi-visited state; no eta search",
    }
    checkpoint_manifest = {
        "schema": "gphi_strict_deadlock_burst_checkpoint_v1",
        "path": str(CHECKPOINT), "sha256": model.sha256,
        "expected_sha256": EXPECTED_CHECKPOINT,
        "architecture": [214, 128, 128, 4], "activation": "SiLU",
        "serialized_architecture": architecture,
        "normalization_path": str(NORMALIZATION), "normalization_sha256": sha256(NORMALIZATION),
        "training_audit": training_audit,
        "later_coverage_retrain_checkpoint_used": False,
    }
    burst_config = {
        "schema": "gphi_strict_deadlock_burst_config_v1",
        "conditions": ["L1", "L2", "L4", "L6", "L8", "H1"],
        "primary_burst_lengths": [1, 2, 4, 6, 8],
        "dense_H1_included": True,
        "trigger_period": 8, "trigger_rule": "absolute global timestep modulo 8 equals zero",
        "burst_rule": "trigger observed after query start activates L consecutive steps; no artificial trigger at query start",
        "Gphi_requery_rule": "recompute feature and query frozen G_phi independently at every active physical step; never hold output",
        "inactive_rule": "g_hat=0 and u_exec=u_safe",
        "special_phase": {"state_id": "old_r106__S_8s", "query_step": 279, "first_trigger": 280},
        "robust_seeds": list(range(95310001, 95310065)),
        "continuation_horizon": "remaining global horizon through absolute step 850",
        "dt": 0.05,
        "new_rollouts_expected": 17 * 5 * 65 + 65,
        "phase_zero_H1_reuse_from_L8": {
            "states": 16, "tuples": 16 * 65,
            "proof": "for a phase-zero queried state, L8 is active at every physical step, so its closed-loop controller is exactly dense H1",
        },
        "L1_reuse": False,
        "L1_reuse_reason": "no prior artifact contains all 17 queried augmented states crossed with the exact 64 robust continuation seeds and original checkpoint",
        "training": False, "eta_search": False, "online_oracle": False, "gate": False,
    }
    integrity = {
        "status": "PREPARED", "frozen_sources": frozen_sources,
        "checkpoint_hash_pass": True, "training_artifact_audit_pass": True,
        "state_hashes_pass": True, "oracle_alignment_pass": True,
        "eta_used_for_control": False, "eta_used_only_for_fixed_teacher_diagnostic": True,
        "eta_search_performed": False, "training_performed": False,
        "global_phase_preserved": True, "Gphi_requeried_every_active_step": True,
    }
    write_json(HERE / "source_manifest.json", source_manifest)
    write_json(HERE / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(HERE / "burst_config.json", burst_config)
    write_json(HERE / "integrity_audit.json", integrity)
    print(json.dumps({
        "status": "PASS", "checkpoint": model.sha256, "states": 17,
        "new_rollouts_expected": burst_config["new_rollouts_expected"],
    }, indent=2))


if __name__ == "__main__":
    main()
