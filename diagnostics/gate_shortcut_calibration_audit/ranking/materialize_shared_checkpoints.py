"""Materialize and verify the three frozen LOGO MLP folds needed by A/B/C."""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CV = ROOT / "diagnostics/hard_stable_boundary_crossval"
OOF = ROOT / "diagnostics/oof_gate_flow_variant_error_audit"
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
SHARED = ROOT / "diagnostics/gate_shortcut_calibration_audit/shared_checkpoints"
sys.path.insert(0, str(OOF))
import run_missing_fold_and_audit as audit  # noqa: E402


GROUPS = ("anchor_D2_pair228", "anchor_D4_pair227", "qual_pair226")


def main() -> None:
    started = time.perf_counter()
    SHARED.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)

    confidence = list(csv.DictReader((CONF / "oracle_confidence_dataset.csv").open()))
    stable = [r for r in confidence if r["oracle_confidence_class"] in ("ORACLE_STABLE_ZERO", "ORACLE_STABLE_NONZERO")]
    state = {r["state_id"]: r for r in stable}
    stable_ids = set(state)
    label = {sid: int(r["original_gate_label"]) for sid, r in state.items()}
    with np.load(DATA / "samples.npz", allow_pickle=False) as loaded:
        arrays = {k: np.asarray(loaded[k]) for k in loaded.files}
    schema = json.loads((DATA / "feature_schema.json").read_text())
    manifest = json.loads((CV / "fold_manifest.json").read_text())
    fold_by_group = {f["heldout_source_group"]: f for f in manifest["folds"]}
    saved = audit.read_csv(CV / "out_of_fold_predictions.csv")

    all_checks = []
    folds = []
    for group in GROUPS:
        recorded = fold_by_group[group]
        split = audit.build_split(stable, group, recorded)
        fold = audit.prepare_fold(split, arrays, stable_ids, label, schema)
        destination = SHARED / group
        destination.mkdir(exist_ok=True)
        trained = audit.train_fold(fold, label, destination)
        checks = audit.reproduction_check(trained, saved, recorded["fold_id"])
        if not all(bool(row["passed"]) for row in checks):
            raise RuntimeError((group, checks))
        all_checks.extend(checks)
        metadata = {
            "fold_id": recorded["fold_id"],
            "outer_group": group,
            "train_state_ids": split["train_ids"],
            "validation_state_ids": split["validation_ids"],
            "test_state_ids": split["test_ids"],
            "train_source_groups": split["train_groups"],
            "validation_source_groups": split["validation_groups"],
            "ensemble_validation_selected_threshold": trained["ensemble_threshold"],
            "training_results": trained["training_rows"],
        }
        (destination / "fold_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=float) + "\n")
        folds.append(metadata)

    audit.write_csv(SHARED / "reproduction_checks.csv", all_checks)
    ready = {
        "status": "READY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "groups": list(GROUPS),
        "seeds": list(audit.SEEDS),
        "all_reproduction_checks_passed": all(bool(r["passed"]) for r in all_checks),
        "max_probability_difference": max(float(r["max_abs_probability_difference"]) for r in all_checks),
        "max_threshold_difference": max(float(r["max_abs_threshold_difference"]) for r in all_checks),
        "tolerance": audit.REPRODUCTION_TOLERANCE,
        "folds": folds,
        "wall_seconds": time.perf_counter() - started,
        "jax_backend": jax.default_backend(),
    }
    audit.write_json(SHARED / "READY.json", ready)
    print(json.dumps({"status": "READY", "wall_s": ready["wall_seconds"], "backend": ready["jax_backend"]}, indent=2))


if __name__ == "__main__":
    main()
