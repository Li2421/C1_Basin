"""Freeze the exact seven OOF folds and sampling semantics for this audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "diagnostics/hard_stable_boundary_crossval/fold_manifest.json"
MISSING = ROOT / "diagnostics/oof_gate_flow_variant_error_audit/missing_fold_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    original = json.loads(ORIGINAL.read_text())
    missing = json.loads(MISSING.read_text())
    folds = list(original["folds"])
    folds.append(
        {
            "fold_id": missing["fold_id"],
            "heldout_source_group": missing["outer_test_source_group"],
            "train_source_groups": missing["train_source_groups"],
            "validation_source_groups": missing["validation_source_groups"],
            "normalization_fit_state_ids": missing["train_state_ids"],
            "all_stable_test_state_ids": missing["test_state_ids"],
            "difficult_test_state_ids": ["R_D1_s95106004_p40"],
            "counts": {
                "train_states": missing["counts"]["train_states"],
                "validation_states": missing["counts"]["validation_states"],
                "all_stable_test_states": missing["counts"]["test_states"],
                "difficult_test_states": 1,
                "train_zero": missing["counts"]["train_zero_states"],
                "train_nonzero": missing["counts"]["train_nonzero_states"],
                "validation_zero": missing["counts"]["validation_zero_states"],
                "validation_nonzero": missing["counts"]["validation_nonzero_states"],
            },
            "origin": "previously validated missing OOF fold; no split recomputation",
        }
    )
    result = {
        "experiment": "SOURCE_GROUP_BALANCED_GATE_TRAINING",
        "design": "seven frozen strict source-group-held-out folds: original six plus validated anchor_D1_pair231",
        "architecture": [214, 64, 64, 1],
        "activation": "SiLU",
        "loss": "ordinary BCE",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)",
        "training_seeds": [17, 23, 41],
        "normalization": "train-only mean/std; boolean dimensions remain literal mean=0 scale=1",
        "threshold_selection": "unchanged validation-only balanced-accuracy rule from hard_stable_boundary_crossval",
        "fold_count": len(folds),
        "folds": folds,
        "sources": {
            "original_six_fold_manifest": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": sha(ORIGINAL)},
            "validated_missing_fold_manifest": {"path": str(MISSING.relative_to(ROOT)), "sha256": sha(MISSING)},
        },
    }
    sampler = {
        "BASELINE_SAMPLE_UNIFORM": {
            "draw": "uniform saved sample", "state_weight": "1/N_train_states because every stable state has exactly 64 saved variants", "source_group_weight": "n_states_in_group/N_train_states", "replacement": True,
        },
        "STATE_BALANCED_ONLY": {
            "draw": "uniform train state then uniform one of that state's 64 saved variants", "state_weight": "1/N_train_states", "source_group_weight": "n_states_in_group/N_train_states", "replacement": True,
        },
        "SOURCE_GROUP_BALANCED": {
            "draw": "uniform train source group, uniform state inside group, uniform one of 64 saved variants", "state_weight": "1/(N_train_groups*n_states_in_group)", "source_group_weight": "1/N_train_groups", "replacement": True,
        },
        "epoch_semantics": "Each optimizer epoch has exactly the original training sample count draws; only draw probabilities change.",
        "no_physical_duplication": True,
        "test_or_validation_labels_used_for_sampling": False,
    }
    (HERE / "frozen_fold_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (HERE / "sampler_definitions.json").write_text(json.dumps(sampler, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold_count": len(folds), "missing_group": missing["outer_test_source_group"]}))


if __name__ == "__main__":
    main()
