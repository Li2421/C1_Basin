"""Freeze generic Stage-2 WIDE sources and outcome-blind absolute-time anchors.

This executable performs no environment rollout and reads no outcome table.
It uses one authoritative generator draw of 160 sources, assigns splits only
by draw index, and predeclares exactly one location-independent anchor per root.
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
HERE = ROOT / "diagnostics/semantic_unified_controller_v1/stage2_recovery_entry"
HELPER = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py"
REFERENCE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json"
REFERENCE_ASSET = Path(
    "/home/zhihan/research/02_C1_Toy_GiveWay/"
    "baseline_309_314/planning/wide_initial_states_200.npz"
)
PROTOCOL = HERE / "collection_protocol.json"
MANIFEST = HERE / "source_split_manifest.json"
ANCHOR_PLAN = HERE / "anchor_plan.csv"
OVERLAP = HERE / "overlap_audit.json"

SOURCE_COUNT = 160
IC_ROOT_SEED = 2026102201
FLOW_ROOT_SEED = 2026102202
ANCHOR_ROOT_SEED = 2026102299
HORIZON = 850
DT = 0.05
SPLIT_RANGES = {
    "train": range(0, 80),
    "validation": range(80, 120),
    "calibration": range(120, 160),
}


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


def load_helper():
    spec = importlib.util.spec_from_file_location("authoritative_wide_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load authoritative WIDE helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_for_index(index: int) -> str:
    for name, indices in SPLIT_RANGES.items():
        if index in indices:
            return name
    raise ValueError(index)


def anchor_for_source(source_id: str) -> int:
    material = f"semantic_stage2_uniform_absolute_v1|{ANCHOR_ROOT_SEED}|{source_id}".encode()
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return int(np.random.default_rng(local_seed).integers(0, HORIZON))


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for path in (MANIFEST, ANCHOR_PLAN, OVERLAP):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
    if (HERE / "runs").exists() or list(HERE.glob("*outcome*")):
        raise RuntimeError("source manifest must be frozen before rollout/outcomes")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["source_count"] != SOURCE_COUNT:
        raise RuntimeError("protocol/source-count mismatch")
    if protocol["official_horizon_steps"] != HORIZON or protocol["dt_seconds"] != DT:
        raise RuntimeError("protocol horizon/dt mismatch")

    helper = load_helper()
    reference = json.loads(REFERENCE.read_text())
    with np.load(REFERENCE_ASSET, allow_pickle=False) as payload:
        historical = np.asarray(payload["test_initial_positions"], dtype=np.float32)
    if not np.array_equal(helper.generate(2026090902, 200), historical):
        raise RuntimeError("authoritative WIDE generator replay mismatch")

    prior, identities, _, prior_ids, audited_assets = helper.collect_prior()
    prior_seed_values = {int(value) for _, value, _ in identities}
    proposed = {IC_ROOT_SEED, FLOW_ROOT_SEED, ANCHOR_ROOT_SEED}
    collisions = sorted(prior_seed_values.intersection(proposed))
    if collisions:
        raise RuntimeError(("prior seed collision", collisions))

    initials = helper.generate(IC_ROOT_SEED, SOURCE_COUNT)
    if len({row.tobytes() for row in initials}) != SOURCE_COUNT:
        raise RuntimeError("duplicate initial condition within Stage-2 source draw")
    source_ids = [f"semantic_stage2_v1_{index:04d}" for index in range(SOURCE_COUNT)]
    id_collisions = sorted(set(source_ids).intersection(prior_ids))
    if id_collisions:
        raise RuntimeError(("prior source-ID collision", id_collisions))
    distances = np.linalg.norm(
        initials[:, None].astype(np.float64) - prior[None].astype(np.float64), axis=(2, 3)
    )
    exact = np.argwhere(distances == 0)
    if len(exact):
        raise RuntimeError(("exact prior IC overlap", exact.tolist()))

    sources: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_RANGES}
    anchor_rows: list[dict[str, Any]] = []
    for index, (source_id, initial) in enumerate(zip(source_ids, initials, strict=True)):
        split = split_for_index(index)
        anchor = anchor_for_source(source_id)
        record = {
            "source_id": source_id,
            "split": split,
            "episode_index": index,
            "rollout_id": index,
            "ic_root_seed": IC_ROOT_SEED,
            "ic_draw_index": index,
            "flow_root_seed": FLOW_ROOT_SEED,
            "initial_positions": initial.tolist(),
            "requested_anchor_steps": [anchor],
        }
        sources[split].append(record)
        anchor_rows.append({
            "split": split,
            "source_id": source_id,
            "episode_index": index,
            "anchor_slot": 0,
            "requested_global_step": anchor,
            "requested_time_seconds": anchor * DT,
            "status_at_freeze": "REQUESTED_OUTCOME_UNKNOWN",
            "materialization_rule": (
                "materialize iff nonterminal before decision at this absolute step; "
                "otherwise unavailable; never replace"
            ),
        })
    expected = protocol["source_group_plan"]
    if {key: len(value) for key, value in sources.items()} != expected:
        raise RuntimeError("index split does not match frozen protocol")

    frozen_utc = datetime.now(timezone.utc).isoformat()
    overlap = {
        "schema": "semantic_stage2_source_overlap_audit_v1",
        "status": "PASS",
        "audit_finished_before_rollout": True,
        "audit_utc": frozen_utc,
        "new_source_count": SOURCE_COUNT,
        "exact_overlap_with_prior_initials": 0,
        "exact_overlap_among_new_initials": 0,
        "minimum_l2_distance_to_prior_initial": float(distances.min()),
        "prior_initial_records_checked": int(len(prior)),
        "prior_unique_initials": len({row.tobytes() for row in prior}),
        "prior_assets_checked": len(set(audited_assets)),
        "prior_numeric_seed_records_checked": len(identities),
        "seed_collisions": [],
        "source_id_collisions": [],
        "replacement_count": 0,
        "outcome_or_geometry_filtering": False,
    }
    overlap["content_sha256"] = canonical_hash(overlap)
    atomic_json(OVERLAP, overlap)

    manifest = {
        "schema": "semantic_stage2_generic_source_split_manifest_v1",
        "status": "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION",
        "frozen_utc": frozen_utc,
        "outcomes_observed_by_this_script": False,
        "environment_rollouts_launched_by_this_script": 0,
        "source_count": SOURCE_COUNT,
        "split_counts": {key: len(value) for key, value in sources.items()},
        "split_rule": "generation index: [0,80) train; [80,120) validation; [120,160) calibration",
        "source_derivatives_grouped_by_root": True,
        "final_test_generated": False,
        "authoritative_generator": {
            "name": reference["authoritative_suite"]["metadata"]["name"],
            "implementation": (
                "numpy.default_rng(seed); abs_x~U(0.55,1.05),(N,2); "
                "y~U(-0.025,0.025),(N,2); signs [-,+]; final float32"
            ),
            "x_absolute_uniform": [0.55, 1.05],
            "y_uniform": [-0.025, 0.025],
            "ic_root_seed": IC_ROOT_SEED,
            "authoritative_asset": str(REFERENCE_ASSET),
            "authoritative_asset_sha256": sha256(REFERENCE_ASSET),
            "reference_replay_seed": 2026090902,
            "reference_replay_bitwise_exact": True,
            "no_rejection_sampling": True,
        },
        "flow_randomness": {
            "root_seed": FLOW_ROOT_SEED,
            "semantics": (
                "episode_key=fold_in(PRNGKey(flow_root_seed),rollout_id); "
                "step_key=fold_in(episode_key,absolute_global_step)"
            ),
        },
        "anchor_rule": {
            "version": "semantic_stage2_uniform_absolute_v1",
            "anchor_root_seed": ANCHOR_ROOT_SEED,
            "absolute_step_range_inclusive": [0, 849],
            "requested_anchors_per_source": 1,
            "sampling": "source-ID-derived deterministic RNG; discrete uniform",
            "uses_terminal_step_or_future_failure_time": False,
            "uses_position_or_spatial_region": False,
            "uses_prior_model_error": False,
            "uses_counterfactual_outcome": False,
            "unavailable_anchor_replacement": False,
        },
        "anchor_request_counts": {key: len(value) for key, value in sources.items()},
        "official_horizon_steps": HORIZON,
        "dt_seconds": DT,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "overlap_audit_path": str(OVERLAP),
        "overlap_audit_sha256": sha256(OVERLAP),
        "preparation_script": str(Path(__file__).resolve()),
        "preparation_script_sha256": sha256(Path(__file__).resolve()),
        "sources": sources,
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    atomic_json(MANIFEST, manifest)
    with ANCHOR_PLAN.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(anchor_rows[0]))
        writer.writeheader()
        writer.writerows(anchor_rows)

    print(json.dumps({
        "status": "FROZEN",
        "source_split_manifest": str(MANIFEST),
        "source_split_manifest_sha256": sha256(MANIFEST),
        "anchor_plan": str(ANCHOR_PLAN),
        "anchor_plan_sha256": sha256(ANCHOR_PLAN),
        "overlap_audit": str(OVERLAP),
        "split_counts": manifest["split_counts"],
        "maximum_source_rollouts": SOURCE_COUNT,
        "maximum_source_steps": SOURCE_COUNT * HORIZON,
    }, indent=2))


if __name__ == "__main__":
    main()
