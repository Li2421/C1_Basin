"""Freeze inputs for the retrained-G_phi dense strict-deadlock audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1"
SOURCE = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1/source_manifest.json"
SOURCE_INTEGRITY = ROOT / "diagnostics/gphi_strict_deadlock_burst_length_v1/integrity_audit.json"
CHECKPOINT = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz"
SELECTION = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/selected_checkpoint.json"
EXPECTED = "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
sys.path.insert(0, str(PILOT))
from pilot_common import assert_frozen_sources, sha256  # noqa: E402


def state_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "active_logs", "traces", "logs"):
        (HERE / name).mkdir(exist_ok=True)
    observed = sha256(CHECKPOINT)
    if observed != EXPECTED:
        raise RuntimeError(("checkpoint hash", observed, EXPECTED))
    selection = json.loads(SELECTION.read_text())
    if selection["checkpoint_sha256"] != EXPECTED or selection["selected_seed"] != 41 or selection["selected_epoch"] != 1177:
        raise RuntimeError("checkpoint selection provenance mismatch")
    previous = json.loads(SOURCE.read_text())
    previous_integrity = json.loads(SOURCE_INTEGRITY.read_text())
    if previous_integrity.get("status") != "PASS" or previous_integrity.get("state_hashes_pass") is not True:
        raise RuntimeError("source audit is not frozen/PASS")
    states = previous["states"]
    if len(states) != 17 or sum(row["benchmark"] == "historical" for row in states) != 11:
        raise RuntimeError("unexpected source cohort")
    mismatched = [row["state_id"] for row in states if state_hash(row["state_file"]) != row["state_sha256"]]
    if mismatched:
        raise RuntimeError(("state hash mismatch", mismatched))
    source = dict(previous)
    source.update({
        "schema": "gphi_retrained_dense_strict_deadlock_source_v1",
        "source_manifest_path": str(SOURCE),
        "source_manifest_sha256": sha256(SOURCE),
        "conditions": ["H1", "L8"],
        "robust_seeds": list(range(95310001, 95310065)),
        "eta_role": "fixed diagnostic teacher only; never used for control or searched",
    })
    write(HERE / "source_manifest.json", source)
    frozen = assert_frozen_sources()
    write(HERE / "checkpoint_integrity.json", {
        "status": "PREPARED",
        "path": str(CHECKPOINT),
        "expected_sha256": EXPECTED,
        "observed_sha256": observed,
        "hash_pass": True,
        "selected_seed": 41,
        "selected_epoch": 1177,
        "selection_manifest": str(SELECTION),
        "selection_manifest_sha256": sha256(SELECTION),
        "source_state_count": 17,
        "historical_states": 11,
        "fresh_diagnostic_states": 6,
        "state_hashes_pass": True,
        "frozen_sources": frozen,
        "no_training": True,
        "no_eta_search": True,
        "no_gate": True,
        "conditions_only": ["dense H1", "H8 trigger + L8"],
        "phase_zero_L8_H1_semantic_reuse_planned": 16,
        "off_phase_states_run_independently": ["old_r106__S_8s"],
    })
    print(json.dumps({"status": "PREPARED", "states": 17, "checkpoint": observed}, indent=2))


if __name__ == "__main__":
    main()
