"""Create isolated, immutable metadata for the conditional-invariance audit."""

from __future__ import annotations

import shutil
from pathlib import Path

import conditional_invariance_gate_runner as runner


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    source = ROOT / "diagnostics" / "gate_source_balancing_v1" / "frozen_fold_manifest.json"
    target = HERE / "frozen_fold_manifest.json"
    if not target.exists():
        shutil.copy2(source, target)
    elif source.read_bytes() != target.read_bytes():
        raise RuntimeError("conditional-invariance manifest differs from frozen source-balanced manifest")
    runner.write_config(HERE / "config.json")
    runner.write_preserved_loss_audit(HERE / "preserved_loss_code_audit.md")
    context, folds = runner.load_context_and_folds()
    eligibility = runner.cell_eligibility_rows(folds, context)
    runner.base.write_csv(HERE / "training_group_class_cell_eligibility.csv", eligibility)
    summary = []
    for fold in folds:
        for class_value in (0, 1):
            rows = [row for row in eligibility if row["fold_id"] == fold["fold_id"] and int(row["oracle_class"]) == class_value]
            included = [row for row in rows if row["included_in_conditional_invariance"]]
            summary.append({
                "fold_id": fold["fold_id"],
                "heldout_source_group": fold["outer_group"],
                "oracle_class": class_value,
                "training_source_group_count": len(rows),
                "nonempty_complete_training_group_class_cells": len(included),
                "minimum_nonempty_cell_samples": min((int(row["complete_training_cell_sample_count"]) for row in included), default=0),
                "maximum_nonempty_cell_samples": max((int(row["complete_training_cell_sample_count"]) for row in included), default=0),
                "class_included_in_penalty": bool(included and included[0]["class_has_at_least_two_training_source_cells"]),
                "uses_validation_or_outer_test_data": False,
            })
    runner.base.write_json(HERE / "training_group_class_eligibility_summary.json", summary)
    runner.base.write_json(HERE / "implementation_manifest.json", {
        "frozen_manifest_byte_identical": source.read_bytes() == target.read_bytes(),
        "source_manifest": str(source),
        "implementation": "conditional_invariance_gate_runner.py",
        "lambda_grid": list(runner.LAMBDA_GRID),
        "total_cell_rows": len(eligibility),
        "included_cell_rows": sum(bool(row["included_in_conditional_invariance"]) for row in eligibility),
        "new_states": 0, "new_oracle_rollouts": 0,
        "feature_schema_changed": False, "oracle_labels_changed": False,
    })


if __name__ == "__main__":
    main()
