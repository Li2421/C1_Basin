"""Freeze and integrity-check the structured-eta closed-loop inputs."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
UPSTREAM = ROOT / "diagnostics/gphi_teacher_takeover_recoverability_v1/source_manifest.json"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1/strict_deadlock_manifest.json"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
CHECKPOINT = HERE / "best_fixed_d_eta_checkpoint.npz"

sys.path.insert(0, str(PILOT))
from pilot_common import assert_frozen_sources, sha256  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def state_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for name in ("raw", "temporal_raw", "logs"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    frozen = assert_frozen_sources()
    selection = json.loads((HERE / "selected_checkpoint.json").read_text())
    observed = sha256(CHECKPOINT)
    if observed != selection["checkpoint_sha256"]:
        raise RuntimeError("selected eta checkpoint hash mismatch")
    source = json.loads(UPSTREAM.read_text())
    states = source["states"]
    if len(states) != 17 or sum(row["benchmark"] == "historical" for row in states) != 11:
        raise RuntimeError("unexpected source cohort")
    mismatches = [row["state_id"] for row in states if state_hash(Path(row["state_file"])) != row["state_sha256"]]
    if mismatches:
        raise RuntimeError(("state hash mismatch", mismatches))
    if {row["state_id"] for row in source["etas_for_diagnostic_only"]} != {row["state_id"] for row in states}:
        raise RuntimeError("source eta mismatch")
    source.update({
        "schema": "fixed_d_structured_eta_source_v1",
        "upstream_manifest": str(UPSTREAM),
        "upstream_manifest_sha256": sha256(UPSTREAM),
        "state_hashes_pass": True,
    })
    atomic_json(HERE / "source_manifest.json", source)
    capacity = json.loads(CAPACITY.read_text())
    integrity = {
        "status": "PASS",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": observed,
        "architecture": [214, 128, 128, 3],
        "parameter_count": 44419,
        "frozen_sources": frozen,
        "capacity_manifest": str(CAPACITY),
        "capacity_manifest_sha256": sha256(CAPACITY),
        "source_state_count": len(states),
        "historical_states": 11,
        "fresh_diagnostic_states": 6,
        "no_online_eta_search": True,
        "no_g_phi_requery_after_onset": True,
        "no_gate": True,
        "no_dagger": True,
        "environment": capacity["environment"],
        "cbf": capacity["cbf"],
    }
    atomic_json(HERE / "integrity_audit.json", integrity)
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
