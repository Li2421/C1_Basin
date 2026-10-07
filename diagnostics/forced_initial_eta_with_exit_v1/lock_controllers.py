"""Hash-lock all frozen controllers and implementations before rollout."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/forced_initial_eta_with_exit_v1"
SINGLE = ROOT / "diagnostics/single_segment_recovery_training_v1"
SOURCE = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py"
MANIFEST = HERE / "fresh_dev_manifest.json"
OUTPUT = HERE / "controller_hashes.json"

import sys
sys.path.insert(0, str(ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1"))
from prepare_manifest import atomic_json, canonical_hash, sha256  # noqa: E402


def record(path: Path, expected: str | None = None) -> dict:
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError((path, expected, actual))
    return {"path": str(path.resolve()), "sha256": actual}


def main() -> None:
    if OUTPUT.exists():
        raw_records = list((HERE / "runs/raw").glob("*/episode_*.json"))
        if raw_records:
            raise RuntimeError("refusing to supersede controller lock after rollout output exists")
        archived = HERE / "controller_hashes_preflight_import_failure.json"
        if archived.exists():
            raise RuntimeError("preflight lock archive already exists")
        os.replace(OUTPUT, archived)
    manifest = json.loads(MANIFEST.read_text())
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if canonical_hash(body) != manifest["content_sha256"]:
        raise RuntimeError("fresh development manifest semantic hash mismatch")
    final_policy_path = SINGLE / "full_loop_manifests/final_test/policy_iteration_eta.json"
    final_policy = json.loads(final_policy_path.read_text())
    hashes_path = SINGLE / "controller_and_projection_hashes.json"
    frozen = json.loads(hashes_path.read_text())
    eta = record(Path(frozen["structured_eta_recovery"]["path"]),
                 "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095")
    direct = record(ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz",
                    "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e")
    exit_head = dict(final_policy["policy"]["exit_head"])
    entry_head = dict(final_policy["policy"]["entry_head"])
    record(Path(exit_head["path"]), exit_head["sha256"])
    record(Path(entry_head["path"]), entry_head["sha256"])
    result = {
        "schema": "forced_initial_eta_exit_controller_lock_v2", "status": "LOCKED_BEFORE_ROLLOUT",
        "supersedes_preflight_lock": {
            "path": str(HERE / "controller_hashes_preflight_import_failure.json"),
            "reason": "preflight import-name collision; zero episode records produced",
        },
        "fresh_dev_manifest": str(MANIFEST), "fresh_dev_manifest_sha256": sha256(MANIFEST),
        "assets": {
            "flowbc": record(Path(frozen["flowbc"]["path"]), frozen["flowbc"]["sha256"]),
            "structured_eta": eta, "direct_g_h8": direct,
            "environment": record(Path(frozen["environment"]["path"]), frozen["environment"]["sha256"]),
            "projection_constraints": record(Path(frozen["projection_constraints"]["path"]), frozen["projection_constraints"]["sha256"]),
            "projection_retry": record(Path(frozen["projection_retry"]["path"]), frozen["projection_retry"]["sha256"]),
            "event_priority_and_monitor": record(Path(frozen["event_priority_and_monitor"]["path"]), frozen["event_priority_and_monitor"]["sha256"]),
            "startup_feature_builder": record(Path(frozen["startup_feature_builder"]["path"]), frozen["startup_feature_builder"]["sha256"]),
            "eta_basis": record(Path(frozen["eta_basis"]["path"]), frozen["eta_basis"]["sha256"]),
            "eta_inference_loader": record(ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/eta_model.py"),
            "state_machine": record(SINGLE / "state_machine.py", "7561536f0fcde671de53948f0527e4ecac7f4d909e3dc8c8ca382d333dc8b429"),
            "audit_runner": record(HERE / "run_audit.py"),
        },
        "exit_head": exit_head, "entry_head_reference": entry_head,
        "authoritative_single_segment_policy": record(final_policy_path,
            "fe26dde62ae6c8b664950f600139f17ef391699abccfa4c9ad0d787413a390cf"),
        "forced_condition_semantics": {
            "learned_entry_head": None, "start": "STRUCTURED_AT_STEP_0",
            "eta_queries": 1, "minimum_structured_transitions_before_exit": 1,
            "exit_threshold_probability": exit_head["threshold_probability"],
            "exit_threshold_logit": exit_head["threshold_logit"],
            "exit_executes_safety_same_step": True, "no_reentry": True,
        },
        "no_learned_entry_in_forced_condition": True,
        "references_are_not_components": ["Learned-Entry+Exit", "Direct-g H8"],
        "forbidden": {"training": False, "online_eta_search": False, "oracle": False,
                      "periodic_start": False, "H_or_L_in_forced_condition": False},
    }
    result["content_sha256"] = canonical_hash(result)
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": "LOCKED", "sha256": sha256(OUTPUT), "content_sha256": result["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
