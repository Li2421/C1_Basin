"""Freeze outcome-blind WIDE source splits and generic absolute-time anchors.

This script performs no environment rollout and reads no outcome table.  It
allocates four independent source groups before any new outcome evaluation.
The two requested decision times for train/validation roots are absolute
episode steps sampled before the trajectory exists; a later collector may
materialize a request only if the episode is still nonterminal at that step.
Unavailable requests are skipped and are never replaced.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/single_segment_recovery_training_v1"
AUTHORITATIVE_HELPER = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py"
REFERENCE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json"
REFERENCE_ASSET = Path(
    "/home/zhihan/research/02_C1_Toy_GiveWay/"
    "baseline_309_314/planning/wide_initial_states_200.npz"
)
PROTOCOL = HERE / "protocol.json"
SPLIT_MANIFEST = HERE / "source_split_manifest.json"
FINAL_MANIFEST = HERE / "final_test_manifest.json"
ANCHOR_PLAN = HERE / "decision_anchor_plan.csv"
OVERLAP_AUDIT = HERE / "source_overlap_audit.json"

# Each source group has independent IC and Flow roots.  These values were
# checked against all prior JSON manifests before this script was committed.
SPLITS = {
    "train": {"count": 40, "ic_seed": 2026100101, "flow_seed": 2026100102},
    "validation": {"count": 20, "ic_seed": 2026100111, "flow_seed": 2026100112},
    "calibration": {"count": 20, "ic_seed": 2026100121, "flow_seed": 2026100122},
    "final_test": {"count": 200, "ic_seed": 2026100131, "flow_seed": 2026100132},
}

# Absolute indices, fixed before outcomes.  The range deliberately excludes
# the first 80 startup transitions and the last 80 horizon transitions.  It
# does not depend on future failure time, terminal time, geometry, or outcome.
ANCHOR_RULE_VERSION = "uniform_absolute_step_v1"
ANCHOR_ROOT_SEED = 2026100199
ANCHOR_LOW = 80
ANCHOR_HIGH_INCLUSIVE = 769
ANCHORS_PER_SOURCE = 2
ANCHOR_SPLITS = {"train", "validation"}
HORIZON_STEPS = 850
DT_SECONDS = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_authoritative_helper():
    spec = importlib.util.spec_from_file_location("frozen_wide_helper", AUTHORITATIVE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import authoritative WIDE generator helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor_steps(source_id: str) -> list[int]:
    seed_material = f"{ANCHOR_RULE_VERSION}|{ANCHOR_ROOT_SEED}|{source_id}".encode()
    source_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(source_seed)
    values = rng.choice(
        np.arange(ANCHOR_LOW, ANCHOR_HIGH_INCLUSIVE + 1, dtype=np.int64),
        size=ANCHORS_PER_SOURCE,
        replace=False,
    )
    return sorted(int(value) for value in values)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for path in (SPLIT_MANIFEST, FINAL_MANIFEST, ANCHOR_PLAN, OVERLAP_AUDIT):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
    forbidden_outcomes = [
        *HERE.glob("full_episode_outcomes*"),
        *HERE.glob("safety_episode_results*"),
        *HERE.glob("paired_branch_outcomes*"),
    ]
    if forbidden_outcomes or (HERE / "runs").exists():
        raise RuntimeError("source splits must be frozen before any new outcome evaluation")
    if not PROTOCOL.is_file():
        raise RuntimeError("frozen protocol.json is required")

    protocol = json.loads(PROTOCOL.read_text())
    expected_counts = protocol["source_group_plan"]
    for split, config in SPLITS.items():
        if int(expected_counts[split]) != int(config["count"]):
            raise RuntimeError((split, "source count differs from frozen protocol"))
    frozen_rule = protocol["generic_anchor_rule"]["rule"]
    if "[80, 769]" not in frozen_rule or "without replacement" not in frozen_rule:
        raise RuntimeError("anchor implementation does not match frozen protocol")

    source = load_authoritative_helper()
    reference = json.loads(REFERENCE.read_text())
    with np.load(REFERENCE_ASSET, allow_pickle=False) as payload:
        historical = np.asarray(payload["test_initial_positions"], dtype=np.float32)
    if not np.array_equal(source.generate(2026090902, 200), historical):
        raise RuntimeError("authoritative WIDE generator replay mismatch")

    prior, identities, _, prior_ids, audited_assets = source.collect_prior()
    prior_seed_values = {int(value) for _, value, _ in identities}
    proposed_seeds = {
        int(config[key])
        for config in SPLITS.values()
        for key in ("ic_seed", "flow_seed")
    }
    proposed_seeds.add(ANCHOR_ROOT_SEED)
    seed_collisions = sorted(prior_seed_values.intersection(proposed_seeds))
    if seed_collisions:
        raise RuntimeError(("prior seed collision", seed_collisions))

    records: dict[str, list[dict[str, Any]]] = {}
    all_initials: list[np.ndarray] = []
    all_ids: list[str] = []
    anchor_rows: list[dict[str, Any]] = []
    for split, config in SPLITS.items():
        initials = source.generate(int(config["ic_seed"]), int(config["count"]))
        if len({row.tobytes() for row in initials}) != int(config["count"]):
            raise RuntimeError((split, "duplicate IC within split"))
        split_records: list[dict[str, Any]] = []
        for index, initial in enumerate(initials):
            source_id = f"single_segment_v1_{split}_{index:04d}"
            requested = anchor_steps(source_id) if split in ANCHOR_SPLITS else []
            row = {
                "source_id": source_id,
                "split": split,
                "episode_index": index,
                "rollout_id": index,
                "ic_root_seed": int(config["ic_seed"]),
                "ic_draw_index": index,
                "flow_root_seed": int(config["flow_seed"]),
                "initial_positions": initial.tolist(),
                "requested_anchor_steps": requested,
            }
            split_records.append(row)
            all_initials.append(initial)
            all_ids.append(source_id)
            for slot, step in enumerate(requested):
                anchor_rows.append(
                    {
                        "split": split,
                        "source_id": source_id,
                        "episode_index": index,
                        "anchor_slot": slot,
                        "requested_global_step": step,
                        "requested_time_seconds": step * DT_SECONDS,
                        "status_at_freeze": "REQUESTED_OUTCOME_UNKNOWN",
                        "materialization_rule": (
                            "materialize iff nonterminal before decision at this absolute step; "
                            "otherwise unavailable; never replace"
                        ),
                    }
                )
        records[split] = split_records

    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("source ID collision across new splits")
    id_collisions = sorted(set(all_ids).intersection(prior_ids))
    if id_collisions:
        raise RuntimeError(("prior source ID collision", id_collisions))
    if len({initial.tobytes() for initial in all_initials}) != len(all_initials):
        raise RuntimeError("exact IC overlap among new splits")

    combined = np.stack(all_initials).astype(np.float64)
    distances = np.linalg.norm(combined[:, None] - prior[None].astype(np.float64), axis=(2, 3))
    exact_prior_pairs = np.argwhere(distances == 0)
    if len(exact_prior_pairs):
        raise RuntimeError(("exact prior IC overlap", exact_prior_pairs.tolist()))

    frozen_utc = datetime.now(timezone.utc).isoformat()
    overlap = {
        "schema": "single_segment_source_overlap_audit_v1",
        "status": "PASS",
        "audit_finished_before_new_outcomes": True,
        "audit_utc": frozen_utc,
        "new_source_count": len(all_initials),
        "new_split_counts": {key: len(value) for key, value in records.items()},
        "exact_overlap_between_new_splits": 0,
        "exact_overlap_with_prior_initials": 0,
        "minimum_l2_distance_to_prior_initial": float(distances.min()),
        "prior_initial_records_checked": int(len(prior)),
        "prior_unique_initials": len({row.tobytes() for row in prior}),
        "prior_assets_checked": len(set(audited_assets)),
        "prior_numeric_seed_records_checked": len(identities),
        "seed_collisions": [],
        "source_id_collisions": [],
        "replacements": 0,
        "outcome_or_geometry_filtering": False,
    }
    overlap["content_sha256"] = canonical_hash(overlap)
    atomic_json(OVERLAP_AUDIT, overlap)

    anchor_rule = {
        "version": ANCHOR_RULE_VERSION,
        "anchor_root_seed": ANCHOR_ROOT_SEED,
        "absolute_step_range_inclusive": [ANCHOR_LOW, ANCHOR_HIGH_INCLUSIVE],
        "requested_anchors_per_train_or_validation_source": ANCHORS_PER_SOURCE,
        "sampling": "source-ID-derived deterministic RNG; uniform without replacement",
        "materialization": (
            "after Safety collection, materialize only predeclared steps reached while "
            "nonterminal; an unavailable request is skipped without replacement"
        ),
        "uses_terminal_step_or_future_failure_time": False,
        "uses_position_or_spatial_region": False,
        "uses_outcome_or_counterfactual": False,
        "calibration_and_final_have_no_labeling_anchors": True,
    }
    manifest = {
        "schema": "single_segment_recovery_source_split_manifest_v1",
        "status": "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION",
        "frozen_utc": frozen_utc,
        "outcomes_observed_by_this_script": False,
        "environment_rollouts_launched_by_this_script": 0,
        "source_count": len(all_initials),
        "split_counts": {key: len(value) for key, value in records.items()},
        "source_derivatives_grouped_by_root": True,
        "authoritative_generator": {
            "name": reference["authoritative_suite"]["metadata"]["name"],
            "implementation": (
                "numpy.default_rng(seed); abs_x~U(0.55,1.05),(N,2); "
                "y~U(-0.025,0.025),(N,2); signs [-,+]; final float32"
            ),
            "x_absolute_uniform": [0.55, 1.05],
            "y_uniform": [-0.025, 0.025],
            "authoritative_asset": str(REFERENCE_ASSET),
            "authoritative_asset_sha256": sha256(REFERENCE_ASSET),
            "reference_replay_seed": 2026090902,
            "reference_replay_bitwise_exact": True,
            "no_rejection_sampling": True,
        },
        "split_roots": {
            split: {
                "episode_count": int(config["count"]),
                "ic_root_seed": int(config["ic_seed"]),
                "flow_root_seed": int(config["flow_seed"]),
                "flow_semantics": (
                    f"episode_key=fold_in(PRNGKey({int(config['flow_seed'])}), rollout_id); "
                    "step_key=fold_in(episode_key, absolute_global_step)"
                ),
            }
            for split, config in SPLITS.items()
        },
        "anchor_rule": anchor_rule,
        "anchor_request_counts": {
            split: sum(len(row["requested_anchor_steps"]) for row in values)
            for split, values in records.items()
        },
        "official_horizon_steps": HORIZON_STEPS,
        "dt_seconds": DT_SECONDS,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "overlap_audit_path": str(OVERLAP_AUDIT),
        "overlap_audit_sha256": sha256(OVERLAP_AUDIT),
        "preparation_script": str(Path(__file__).resolve()),
        "preparation_script_sha256": sha256(Path(__file__).resolve()),
        "sources": records,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    atomic_json(SPLIT_MANIFEST, manifest)

    with ANCHOR_PLAN.open("w", newline="") as handle:
        fieldnames = list(anchor_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anchor_rows)

    final_manifest = {
        "schema": "single_segment_recovery_final_test_manifest_v1",
        "status": "RESERVED_UNTOUCHED_BEFORE_MODEL_SELECTION_AND_CALIBRATION",
        "frozen_utc": frozen_utc,
        "outcomes_observed": False,
        "episode_count": len(records["final_test"]),
        "purpose": "single-use final frozen full-episode comparison after all selection",
        "ic_root_seed": SPLITS["final_test"]["ic_seed"],
        "flow_root_seed": SPLITS["final_test"]["flow_seed"],
        "source_split_manifest_path": str(SPLIT_MANIFEST),
        "source_split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "no_decision_labeling_or_checkpoint_selection": True,
        "episodes": records["final_test"],
    }
    final_manifest["content_sha256"] = canonical_hash(final_manifest)
    atomic_json(FINAL_MANIFEST, final_manifest)

    print(
        json.dumps(
            {
                "status": "FROZEN",
                "source_split_manifest": str(SPLIT_MANIFEST),
                "source_split_manifest_sha256": sha256(SPLIT_MANIFEST),
                "final_test_manifest": str(FINAL_MANIFEST),
                "final_test_manifest_sha256": sha256(FINAL_MANIFEST),
                "anchor_plan": str(ANCHOR_PLAN),
                "anchor_plan_sha256": sha256(ANCHOR_PLAN),
                "counts": manifest["split_counts"],
                "anchor_requests": manifest["anchor_request_counts"],
                "overlap": overlap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
