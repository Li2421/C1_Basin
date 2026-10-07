"""Freeze and integrity-check inputs for the teacher-takeover audit."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_teacher_takeover_recoverability_v1"
SOURCE = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1/source_manifest.json"
SOURCE_GATE = ROOT / "diagnostics/gphi_retrained_dense_strict_deadlock_v1/checkpoint_integrity.json"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1/strict_deadlock_manifest.json"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
CHECKPOINTS = {
    "coverage": (
        ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz",
        "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700",
    ),
    "k1": (
        ROOT / "diagnostics/gphi_dagger_k1_diagnostic_v1/best_dagger_k1_checkpoint.npz",
        "83c704f2e1ce0fbe50abd5a0d3e96dd202b4a89256a4ea0e6a954eda0340f70a",
    ),
}
KS = [0, 1, 2, 4, 8, 16, 32]
ROBUST_SEEDS = list(range(95310001, 95310065))

sys.path.insert(0, str(PILOT))
from pilot_common import assert_frozen_sources, sha256  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    for name in ("raw", "reference", "logs"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    frozen = assert_frozen_sources()
    source = json.loads(SOURCE.read_text())
    source_gate = json.loads(SOURCE_GATE.read_text())
    capacity = json.loads(CAPACITY.read_text())
    states = source["states"]
    if len(states) != 17 or sum(row["benchmark"] == "historical" for row in states) != 11:
        raise RuntimeError("unexpected frozen source cohort")
    mismatches = [
        row["state_id"] for row in states
        if state_hash(Path(row["state_file"])) != row["state_sha256"]
    ]
    if mismatches:
        raise RuntimeError(("state hash mismatch", mismatches))
    checkpoint_rows = []
    for learner, (path, expected) in CHECKPOINTS.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError((learner, observed, expected))
        checkpoint_rows.append({
            "learner": learner, "path": str(path), "expected_sha256": expected,
            "observed_sha256": observed, "hash_pass": True,
        })
    eta_rows = source["etas_for_diagnostic_only"]
    if {row["state_id"] for row in eta_rows} != {row["state_id"] for row in states}:
        raise RuntimeError("eta/state key mismatch")
    frozen_source = dict(source)
    frozen_source.update({
        "schema": "gphi_teacher_takeover_source_v1",
        "upstream_manifest_path": str(SOURCE),
        "upstream_manifest_sha256": sha256(SOURCE),
        "source_gate_sha256": sha256(SOURCE_GATE),
        "capacity_manifest_sha256": sha256(CAPACITY),
        "state_hashes_pass": True,
    })
    atomic_json(HERE / "source_manifest.json", frozen_source)
    atomic_json(HERE / "learner_checkpoint_manifest.json", {
        "status": "PASS", "checkpoints": checkpoint_rows,
        "no_training": True, "no_fine_tuning": True,
        "no_eta_search": True, "no_gate": True,
    })
    atomic_json(HERE / "takeover_config.json", {
        "schema": "gphi_teacher_takeover_config_v1",
        "takeover_steps": KS,
        "robust_seeds": ROBUST_SEEDS,
        "rollouts_per_learner_per_k": len(states) * len(ROBUST_SEEDS),
        "planned_takeover_rollouts": 2 * len(KS) * len(states) * len(ROBUST_SEEDS),
        "learner_prefix": "dense H1 G_phi for exactly k transitions",
        "post_takeover": "same frozen eta* recomputed densely forever; no return to G_phi",
        "flow_rng": "fold_in(PRNGKey(seed), state rng_namespace), then fold_in absolute global timestep",
        "continuation_horizon": "remaining frozen global episode horizon (max_steps=850)",
        "teacher_action_diagnostic": "learner and fixed eta* evaluated on identical learner-visited z_hat_k",
        "oracle_geometry_reference": "dense eta* from source, same matched Flow stream, sampled before actions at k",
        "environment": capacity["environment"], "cbf": capacity["cbf"],
        "resource_cap": {"shards": 3, "memory_fraction_total_max": 0.35, "gpu_memory_fraction_per_process": 0.10},
        "frozen_sources": frozen,
        "source_gate_status": source_gate.get("status"),
    })
    print(json.dumps({"status": "PREPARED", "states": 17, "checkpoints": checkpoint_rows}, indent=2))


if __name__ == "__main__":
    main()
