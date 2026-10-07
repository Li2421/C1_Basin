"""Unit tests for probe math and grouped splitting; no rollout/probe execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("entry_probes", HERE / "run_identifiability_probes.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_group_splits_have_no_source_leakage_and_full_oof_coverage() -> None:
    groups = np.asarray([f"root_{index // 2:03d}" for index in range(40)])
    splits, assignments = module.make_group_folds(groups)
    assert len(splits) == 15
    for repeat in range(3):
        seen = []
        for split in [row for row in splits if row["repeat"] == repeat]:
            train = set(groups[split["train"]]); val = set(groups[split["validation"]]); test = set(groups[split["test"]])
            assert not train & val and not train & test and not val & test
            seen.extend(split["test"].tolist())
        assert sorted(seen) == list(range(len(groups)))
    assert len(assignments) == 3 * 5 * len(set(groups))


def test_paired_advantage_identity_and_threshold_policy() -> None:
    q_n = np.asarray([1.0, 0.0, 0.5, 0.5])
    q_r = np.asarray([0.0, 1.0, 0.5, 0.75])
    rescue = np.asarray([0.0, 1.0, 0.0, 0.25])
    breaks = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(q_r - q_n, rescue - breaks)
    score = q_r - q_n
    threshold = module.choose_policy_threshold(score, q_n, q_r)
    metrics = module.policy_metrics(score, threshold, q_n, q_r, rescue, breaks)
    assert metrics["expected_success"] >= max(np.mean(q_n), np.mean(q_r))
    assert np.isfinite(threshold)

    all_recovery = module.choose_policy_threshold(
        score, np.zeros_like(q_n), np.ones_like(q_r)
    )
    assert np.isfinite(all_recovery)
    assert np.all(score > all_recovery)


def test_ridge_and_metrics_are_finite() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 9)); y = x[:, 0] - 0.5 * x[:, 1]
    weights, intercept = module.ridge_fit(x, y, 0.1)
    prediction = x @ weights + intercept
    metrics = module.continuous_metrics(y, prediction)
    assert metrics["pearson"] is not None and metrics["pearson"] > 0.99
    assert metrics["mae"] < 0.05


def test_binary_metrics_ties_and_class_support() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([-2.0, -1.0, 1.0, 2.0])
    threshold = module.choose_binary_threshold(labels, scores)
    assert np.isfinite(threshold)
    metrics = module.binary_metrics(labels, scores, threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
            print("PASS", name)
