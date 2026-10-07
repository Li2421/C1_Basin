"""Cross-evaluate frozen V1 and V2 checkpoints on identical held-out states."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_V1 = ROOT / "diagnostics/gphi_training_dataset_v1"
DATA_V2 = ROOT / "diagnostics/gphi_training_dataset_v2"
TRAIN_V1 = ROOT / "diagnostics/gphi_pilot_training_v1"
sys.path.insert(0, str(ROOT))

from diagnostics.gphi_pilot_training_v2.train_and_evaluate import (
    normalize, predict_mlp, subset_metrics, write_json,
)


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        mean = np.asarray(saved["normalization_mean"])
        scale = np.asarray(saved["normalization_scale"])
        params = []
        index = 0
        while f"layer_{index}_weight" in saved:
            params.append({
                "w": np.asarray(saved[f"layer_{index}_weight"]),
                "b": np.asarray(saved[f"layer_{index}_bias"]),
            })
            index += 1
    return params, mean, scale


def grouped(prediction, target, state_id) -> dict:
    metrics = subset_metrics(prediction, target, state_id)
    return {
        "state_count": metrics["state_count"],
        "sample_count": metrics["sample_count"],
        "state_grouped_mean_l2": metrics["state_grouped_mean_l2"],
        "state_grouped_rmse": metrics["state_grouped_rmse"],
        "mean_relative_l2_nonzero": metrics["mean_relative_l2_nonzero"],
    }


def main() -> None:
    jax.config.update("jax_platform_name", "cpu")
    with np.load(DATA_V2 / "samples.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in (
            "features", "targets", "state_id", "split", "category")}
    predictions = {}
    for version, path in (("V1", TRAIN_V1 / "best_checkpoint.npz"),
                          ("V2", HERE / "best_checkpoint.npz")):
        params, mean, scale = load_checkpoint(path)
        predictions[version] = predict_mlp(
            params, normalize(arrays["features"], mean, scale).astype(np.float32))

    retained_v1_test = set(json.loads((DATA_V1 / "split_manifest.json").read_text())["state_ids"]["test"])
    cohorts = {
        "expanded_v2_test": arrays["split"] == "test",
        "retained_original_v1_test": (
            (arrays["split"] == "test")
            & np.isin(arrays["state_id"], sorted(retained_v1_test))
        ),
    }
    result = {
        "purpose": "separate state-coverage gain from changed held-out test composition",
        "selection_used_cross_evaluation": False,
        "cohorts": {},
    }
    zero_all = np.linalg.norm(arrays["targets"], axis=1) <= 1e-14
    for cohort, mask in cohorts.items():
        entry = {"models": {}, "category_state_grouped_mean_l2": {}}
        for version in ("V1", "V2"):
            prediction = predictions[version]
            entry["models"][version] = {
                "all": grouped(prediction[mask], arrays["targets"][mask], arrays["state_id"][mask]),
                "zero_label": grouped(
                    prediction[mask & zero_all], arrays["targets"][mask & zero_all],
                    arrays["state_id"][mask & zero_all]),
                "nonzero_label": grouped(
                    prediction[mask & ~zero_all], arrays["targets"][mask & ~zero_all],
                    arrays["state_id"][mask & ~zero_all]),
            }
        for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
            chosen = mask & (arrays["category"] == category)
            entry["category_state_grouped_mean_l2"][category] = {
                version: grouped(
                    predictions[version][chosen], arrays["targets"][chosen], arrays["state_id"][chosen]
                )["state_grouped_mean_l2"]
                for version in ("V1", "V2")
            }
        result["cohorts"][cohort] = entry

    expanded = result["cohorts"]["expanded_v2_test"]["models"]
    retained = result["cohorts"]["retained_original_v1_test"]["models"]
    result["coverage_effect"] = {
        "expanded_test_zero_false_intervention_fractional_reduction_v1_to_v2": (
            1.0 - expanded["V2"]["zero_label"]["state_grouped_mean_l2"]
            / expanded["V1"]["zero_label"]["state_grouped_mean_l2"]),
        "expanded_test_all_error_fractional_reduction_v1_to_v2": (
            1.0 - expanded["V2"]["all"]["state_grouped_mean_l2"]
            / expanded["V1"]["all"]["state_grouped_mean_l2"]),
        "retained_v1_test_zero_false_intervention_fractional_reduction_v1_to_v2": (
            1.0 - retained["V2"]["zero_label"]["state_grouped_mean_l2"]
            / retained["V1"]["zero_label"]["state_grouped_mean_l2"]),
        "retained_v1_test_all_error_fractional_reduction_v1_to_v2": (
            1.0 - retained["V2"]["all"]["state_grouped_mean_l2"]
            / retained["V1"]["all"]["state_grouped_mean_l2"]),
        "diagnosis": (
            "Coverage produces a large matched-state gain, but the expanded held-out zero-label "
            "absolute error remains poor and scaling is still improving: STILL_DATA_LIMITED."
        ),
    }
    write_json(HERE / "v1_v2_matched_comparison.json", result)
    print(json.dumps(result["coverage_effect"], indent=2))


if __name__ == "__main__":
    main()
