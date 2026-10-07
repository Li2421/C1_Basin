"""Preflight the requested OOF Flow-variant audit and record hard blockers.

No model is trained and no inference is run if any primary state lacks a genuine
pre-existing outer source-group-held-out fold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FLOW = ROOT / "diagnostics/flow_variant_representation_collision_audit"
CV = ROOT / "diagnostics/hard_stable_boundary_crossval"
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc)
    HERE.mkdir(parents=True, exist_ok=True)

    pair_rows = read_csv(FLOW / "audited_pairs.csv")
    confidence = {row["state_id"]: row for row in read_csv(CONF / "oracle_confidence_dataset.csv")}
    fold_manifest = json.loads((CV / "fold_manifest.json").read_text())
    folds = fold_manifest["folds"]
    fold_by_outer = {fold["heldout_source_group"]: fold for fold in folds}
    predictions = read_csv(CV / "out_of_fold_predictions.csv")
    prediction_key = {
        (row["state_id"], row["model"], row["training_seed"]): row
        for row in predictions
    }

    audited = []
    seen = set()
    for pair in pair_rows:
        if pair["pair_role"] not in ("primary_collision", "preexisting_noncollision_control"):
            continue
        for role, key in (("gate0", "zero_state_id"), ("gate1", "nonzero_state_id")):
            state_id = pair[key]
            record_key = (pair["pair_id"], state_id)
            if record_key in seen:
                continue
            seen.add(record_key)
            source = confidence[state_id]
            source_group = source["source_group"]
            fold = fold_by_outer.get(source_group)
            saved = prediction_key.get((state_id, "MLP_64x64", "SEED_MEAN"))
            audited.append(
                {
                    "pair_id": pair["pair_id"],
                    "pair_role": pair["pair_role"],
                    "state_role": role,
                    "state_id": state_id,
                    "source_group": source_group,
                    "oracle_class": source["original_gate_label"],
                    "exact_outer_fold_exists": fold is not None,
                    "exact_outer_fold_id": fold["fold_id"] if fold else "",
                    "saved_oof_seed_mean_probability": saved["p_gate"] if saved else "",
                    "saved_oof_threshold": saved["validation_selected_threshold"] if saved else "",
                    "saved_oof_prediction": saved["predicted_label"] if saved else "",
                    "saved_oof_correct": saved["correct"] if saved else "",
                }
            )

    primary = [row for row in audited if row["pair_role"] == "primary_collision"]
    missing = [row for row in primary if not row["exact_outer_fold_exists"]]
    assert len(primary) == 4
    assert [row["state_id"] for row in missing] == ["R_D1_s95106004_p40"]

    checks = []
    for row in audited:
        if row["exact_outer_fold_exists"]:
            status = "AVAILABLE_NOT_REPRODUCED_DUE_GLOBAL_PREFLIGHT_STOP"
            detail = "No checkpoint was saved; exact reproduction would be possible from the recorded fold, but was not launched after a primary-state blocker was found."
        else:
            status = "BLOCKED_NO_PREEXISTING_OUTER_FOLD"
            detail = "No outer fold held out this source group; no existing OOF gate satisfies the no-train/no-normalization/no-threshold/no-model-selection requirement."
        checks.append(
            {
                "pair_id": row["pair_id"], "state_id": row["state_id"],
                "source_group": row["source_group"], "fold_id": row["exact_outer_fold_id"],
                "checkpoint_saved": False, "status": status, "detail": detail,
            }
        )

    audit_fields = [
        "pair_id", "pair_role", "state_role", "state_id", "source_group", "oracle_class",
        "exact_outer_fold_exists", "exact_outer_fold_id", "saved_oof_seed_mean_probability",
        "saved_oof_threshold", "saved_oof_prediction", "saved_oof_correct",
    ]
    write_csv(HERE / "audited_states.csv", audited, audit_fields)
    write_csv(HERE / "fold_reproduction_checks.csv", checks, ["pair_id", "state_id", "source_group", "fold_id", "checkpoint_saved", "status", "detail"])

    blocked_note = "BLOCKED: inference not run because R_D1_s95106004_p40 has no pre-existing valid outer source-group-held-out fold."
    empty_specs = {
        "per_variant_predictions.csv": ["state_id", "source_group", "oracle_class", "flow_seed", "p_gate", "validation_selected_threshold", "predicted_class", "correct", "gate_logit"],
        "per_state_prediction_statistics.csv": ["state_id", "correct_count", "wrong_count", "mean_p_gate", "std_p_gate", "median_p_gate", "min_p_gate", "max_p_gate", "p5", "p95", "classification", "status"],
        "pair_score_separation.csv": ["pair_id", "mean_p_zero", "mean_p_nonzero", "score_overlap", "ranking_accuracy", "cloud_AUC", "mean_score_margin", "status"],
        "threshold_vs_ranking.csv": ["pair_id", "threshold_classification_accuracy", "ranking_accuracy", "failure_type", "status"],
        "flow_dependence_analysis.csv": ["state_id", "p_gate_std", "feature_group", "logit_correlation", "status"],
        "training_neighbor_score_comparison.csv": ["state_id", "neighbor_role", "neighbor_state_id", "mean_p_gate", "status"],
        "source_group_shortcut_analysis.csv": ["state_id", "source_group", "comparison", "value", "status"],
        "control_pair_results.csv": ["pair_id", "state_id", "correct_count", "ranking_accuracy", "score_margin", "status"],
    }
    for name, fields in empty_specs.items():
        write_csv(HERE / name, [], fields)
        with (HERE / name).open("a") as handle:
            handle.write("# " + blocked_note + "\n")

    report = f"""# OOF gate Flow-variant error audit — preflight stop

## Outcome

The requested inference audit was **not launched** because one of the four primary states has no valid pre-existing OOF fold.

`R_D1_s95106004_p40` belongs to source group `anchor_D1_pair231`. The existing hard-stable cross-validation contains outer folds only for:

{os.linesep.join(f'- `{fold["heldout_source_group"]}`' for fold in folds)}

Consequently, no saved/reproducible existing gate exists for which `anchor_D1_pair231` was absent from training, normalization fitting, validation threshold selection, and model selection. In `LOGO_00_anchor_D2_pair228` this group is an inner-validation group, so using that gate would explicitly violate the requested OOF condition. In the other relevant folds it is training data.

No fold checkpoints were saved. Three primary states have a recorded exact outer fold and could be reproduced, but pair 2 cannot receive the requested valid pair-level ranking/AUC without creating a new seventh outer fold. Creating that fold would be a new evaluation, not reproduction of an existing OOF gate.

Per the task's explicit instruction, “If exact reproduction fails, STOP and report why,” no training or per-variant inference was performed. No final failure-mode classification is scientifically valid from an incomplete pair audit.

## Smallest justified next experiment

Pre-register and run one additional `LOGO_anchor_D1_pair231` fold using the unchanged deterministic split/training recipe and seeds 17/23/41, then rerun this audit. This supplies the single missing OOF gate without changing architecture, features, oracle, states, or rollouts.
"""
    (HERE / "flow_variant_gate_audit_report.md").write_text(report)

    sanity = {
        "status": "BLOCKED_NO_EXISTING_OOF_FOLD",
        "preflight_passed": False,
        "primary_state_count": 4,
        "primary_states_with_exact_outer_fold": 3,
        "primary_states_missing_exact_outer_fold": [row["state_id"] for row in missing],
        "blocking_source_group": "anchor_D1_pair231",
        "fold_checkpoints_saved": False,
        "models_trained": 0,
        "per_variant_inferences": 0,
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "prior_artifacts_modified": False,
    }
    write_json(HERE / "sanity_checks.json", sanity)
    runtime = {
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started,
        "CPU_processes": 1,
        "CPU_threads_cap": 1,
        "GPU_used": False,
        "GPU_shards": 0,
        "resource_snapshot": {
            "scheduler_jobs": 0,
            "GPU_memory_total_MiB": 97887,
            "GPU_memory_used_MiB": 2,
            "GPU_memory_free_MiB": 97247,
            "GPU_utilization_percent": 0,
            "system_load_average_at_preflight": [4.02, 4.03, 3.38],
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    output_names = [
        "flow_variant_gate_audit_report.md", "audited_states.csv", "fold_reproduction_checks.csv",
        *empty_specs.keys(), "sanity_checks.json", "runtime_statistics.json",
    ]
    input_paths = [
        FLOW / "audited_pairs.csv", CV / "fold_manifest.json",
        CV / "out_of_fold_predictions.csv", CONF / "oracle_confidence_dataset.csv",
    ]
    manifest = {
        "experiment": "OOF_GATE_FLOW_VARIANT_ERROR_AUDIT",
        "status": "BLOCKED_NO_EXISTING_OOF_FOLD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blocker": "R_D1_s95106004_p40 / anchor_D1_pair231 has no pre-existing outer-held-out fold",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in input_paths],
        "outputs": [{"path": name, "sha256": sha(HERE / name)} for name in output_names],
        "analysis_script": {"path": "run_preflight.py", "sha256": sha(HERE / "run_preflight.py")},
        "constraints": {"models_trained": 0, "new_states": 0, "new_oracle_rollouts": 0, "GPU_shards": 0},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "blocker": manifest["blocker"], "wall_s": runtime["wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
