#!/usr/bin/env python3
"""Freeze the state-conditioned Q audit before any new outcome is evaluated."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_q_learnability_v2"
DATASET = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1/dataset_424_manifest.json"
ETA_SOURCE = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json"
STATE_SEED = "orthoflow3_q_v2_state_groups_20260927"
CONDITION_SEED = "orthoflow3_q_v2_h_condition_20260927"
FUTURE_ROOT_SEED = 2026092702


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(token: str, value: str) -> str:
    return hashlib.sha256(f"{token}|{value}".encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_feature_variants() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    # Startup-complete is authoritative where present. v1 supplies 11 historical states.
    for priority, name in enumerate((
        "gphi_training_dataset_startup_complete_v1",
        "gphi_training_dataset_v1",
        "gphi_training_dataset_strict_deadlock_v1",
    )):
        directory = ROOT / "diagnostics" / name
        rows = [json.loads(line) for line in (directory / "sample_metadata.jsonl").read_text().splitlines() if line.strip()]
        arrays = np.load(directory / "samples.npz")
        features = np.asarray(arrays["features"], dtype=np.float64)
        if len(rows) != len(features):
            raise RuntimeError((name, len(rows), len(features)))
        for index, row in enumerate(rows):
            result[row["state_id"]].append({
                "priority": priority,
                "dataset": name,
                "row_index": index,
                "flow_seed": int(row["flow_seed"]),
                "sample_id": row["sample_id"],
                "feature": features[index],
            })
    # Remove duplicate copies by flow seed, preferring startup-complete.
    for state_id, rows in list(result.items()):
        chosen = {}
        for row in sorted(rows, key=lambda item: (item["priority"], item["flow_seed"])):
            chosen.setdefault(row["flow_seed"], row)
        result[state_id] = list(chosen.values())
    return result


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(DATASET.read_text())
    variants = load_feature_variants()
    all_states = manifest["states"]
    # Historical strict-deadlock additions do not retain the RNG namespace needed
    # to bind their archived h variant to an exact current Flow key. They remain in
    # the 424-state source manifest but are ineligible for Q(h,eta) conditioning.
    excluded = [row for row in all_states if row.get("rng_namespace") is None]
    states = [row for row in all_states if row.get("rng_namespace") is not None]
    missing = sorted(row["state_id"] for row in states if row["state_id"] not in variants)
    if missing:
        raise RuntimeError(("states without deployment feature variants", missing))

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in states:
        groups[row["leakage_group"]].append(row)
    ordered_groups = sorted(groups, key=lambda value: rank(STATE_SEED, value))
    if len(ordered_groups) < 120:
        raise RuntimeError(("too few leakage groups", len(ordered_groups)))
    selected_groups = {
        "train": ordered_groups[:80],
        "val": ordered_groups[80:100],
        "test": ordered_groups[100:120],
    }

    selected = []
    feature_rows = []
    for split, group_ids in selected_groups.items():
        for group_id in group_ids:
            # One outcome-blind representative per group maximizes independent-source diversity.
            state = min(groups[group_id], key=lambda item: rank(STATE_SEED + "|state", item["state_id"]))
            candidate_variants = variants[state["state_id"]]
            variant = min(candidate_variants, key=lambda item: rank(CONDITION_SEED, item["sample_id"]))
            h = np.asarray(variant["feature"], dtype=np.float64)
            if h.shape != (214,) or not np.isfinite(h).all():
                raise RuntimeError((state["state_id"], "invalid h", h.shape))
            feature_index = len(feature_rows)
            feature_rows.append(h)
            selected.append({
                "state_id": state["state_id"],
                "source_trajectory": state["source_trajectory"],
                "source_group": group_id,
                "split": split,
                "absolute_step": int(state["absolute_step"]),
                "rng_namespace": int(state["rng_namespace"]),
                "state_file": state["state_file"],
                "state_sha256": state["state_sha256"],
                "h_conditioning_identifier": variant["sample_id"],
                "flow_seed": int(variant["flow_seed"]),
                "feature_source": variant["dataset"],
                "feature_row_index": int(variant["row_index"]),
                "feature_index": feature_index,
                "feature_sha256": hashlib.sha256(h.tobytes()).hexdigest(),
            })
    if len(selected) != 120 or len({row["source_group"] for row in selected}) != 120:
        raise AssertionError("state/group selection is not 120 independent groups")
    np.savez_compressed(HERE / "conditioning_features.npz", features=np.stack(feature_rows))

    eta_design = json.loads(ETA_SOURCE.read_text())
    eta_rows = [{"probe_id": "zero", "eta": [0.0, 0.0, 0.0], "source": "explicit_zero"}]
    for point in eta_design["points"][:23]:
        eta_rows.append({"probe_id": point["parameter_id"], "eta": point["theta"], "source": "authoritative_sobol_prefix"})
    if len(eta_rows) != 24 or len({tuple(row["eta"]) for row in eta_rows}) != 24:
        raise AssertionError("invalid eta cloud")
    with (HERE / "eta_probe_cloud.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["probe_id", "eta1", "eta2", "eta3", "source"])
        for row in eta_rows:
            writer.writerow([row["probe_id"], *row["eta"], row["source"]])

    state_manifest = {
        "schema": "orthoflow3_q_v2_eligible_state_manifest",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_pool_path": str(DATASET), "eligible_pool_sha256": sha(DATASET),
        "source_pool_unique_states": len(all_states),
        "eligible_unique_states": len(states), "eligible_leakage_groups": len(groups),
        "excluded_states": [{"state_id": row["state_id"], "reason": "missing_rng_namespace_for_exact_h_conditioning"} for row in excluded],
        "state_selection_seed": STATE_SEED, "conditioning_selection_seed": CONDITION_SEED,
        "selection_rule": "hash-permute leakage groups; one hash-ranked state per group; one hash-ranked archived Flow variant per state",
        "selected_states": selected,
    }
    write_json(HERE / "eligible_state_manifest.json", state_manifest)
    split_manifest = {
        "schema": "orthoflow3_q_v2_state_split_manifest",
        "frozen_before_outcomes": True,
        "group_unit": "leakage_group",
        "counts": {split: sum(row["split"] == split for row in selected) for split in ("train", "val", "test")},
        "groups": selected_groups,
        "states": selected,
        "overlap": {
            "train_val": sorted(set(selected_groups["train"]) & set(selected_groups["val"])),
            "train_test": sorted(set(selected_groups["train"]) & set(selected_groups["test"])),
            "val_test": sorted(set(selected_groups["val"]) & set(selected_groups["test"])),
        },
    }
    write_json(HERE / "state_split_manifest.json", split_manifest)

    tasks = []
    for state in selected:
        trial_count = 8 if state["split"] == "test" else 4
        for eta_row in eta_rows:
            candidate_id = f"{state['state_id']}__{eta_row['probe_id']}"
            for future_index in range(trial_count):
                tasks.append({
                    "task_id": f"{candidate_id}__f{future_index:02d}",
                    "candidate_id": candidate_id,
                    "state_id": state["state_id"], "split": state["split"],
                    "probe_id": eta_row["probe_id"], "eta": eta_row["eta"],
                    "future_index": future_index,
                })
    plan = {
        "schema": "orthoflow3_q_v2_base_rollout_plan",
        "frozen_before_outcomes": True,
        "orthoflow3_sha256": "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38",
        "future_root_seed": FUTURE_ROOT_SEED,
        "conditioning": "fixed archived current Flow key represented in h; future Flow keys resampled only after query transition",
        "tasks": tasks,
        "maximum_new_continuations": len(tasks),
    }
    write_json(HERE / "base_rollout_plan.json", plan)
    write_json(HERE / "cache_reuse_audit.json", {
        "schema": "orthoflow3_q_v2_cache_reuse_preflight",
        "compatible_reused_continuations": 0,
        "reason": "Archived oracle records integrate over seeds that change the query-step Flow draw; they target Q(z,eta), not the frozen Q(h,eta). State and feature assets are reused, but no outcome is relabeled.",
        "required_exact_tuple": ["state", "h/current Flow key", "eta", "future seed", "OrthoFlow3 hash", "horizon", "projection stack", "monitor/history", "RNG semantics"],
    })
    print(json.dumps({"states": len(selected), "groups": len(set(row['source_group'] for row in selected)), "tasks": len(tasks), "eta": len(eta_rows)}, indent=2))


if __name__ == "__main__":
    main()
