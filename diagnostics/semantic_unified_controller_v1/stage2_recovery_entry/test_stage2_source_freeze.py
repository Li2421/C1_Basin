from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_prepare():
    path = HERE / "prepare_stage2_sources.py"
    spec = importlib.util.spec_from_file_location("prepare_stage2_sources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_anchor_and_bounds() -> None:
    module = load_prepare()
    value = module.anchor_for_source("semantic_stage2_v1_0042")
    assert value == module.anchor_for_source("semantic_stage2_v1_0042")
    assert 0 <= value < 850


def test_index_split_partition() -> None:
    module = load_prepare()
    assert [module.split_for_index(i) for i in (0, 79, 80, 119, 120, 159)] == [
        "train", "train", "validation", "validation", "calibration", "calibration"
    ]


def test_frozen_manifest_and_plan_if_present() -> None:
    manifest_path = HERE / "source_split_manifest.json"
    plan_path = HERE / "anchor_plan.csv"
    if not manifest_path.is_file():
        return
    module = load_prepare()
    manifest = json.loads(manifest_path.read_text())
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    assert module.canonical_hash(body) == manifest["content_sha256"]
    assert manifest["status"] == "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION"
    assert manifest["split_counts"] == {"train": 80, "validation": 40, "calibration": 40}
    assert manifest["source_count"] == 160
    assert manifest["official_horizon_steps"] == 850
    assert manifest["dt_seconds"] == 0.05
    ids = []
    for split, rows in manifest["sources"].items():
        for row in rows:
            assert row["split"] == split
            assert len(row["requested_anchor_steps"]) == 1
            ids.append(row["source_id"])
    assert len(ids) == len(set(ids)) == 160
    with plan_path.open(newline="") as handle:
        plan = list(csv.DictReader(handle))
    assert len(plan) == 160
    assert {row["source_id"] for row in plan} == set(ids)


def test_collector_is_rollout_only() -> None:
    text = (HERE / "collect_stage2_sources.py").read_text()
    for forbidden in ("train_decision", "branch_label", "eta_search", "final_test"):
        assert forbidden not in text
