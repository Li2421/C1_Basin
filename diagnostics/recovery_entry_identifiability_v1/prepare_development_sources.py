"""Freeze new development-only WIDE roots and generic anchor requests.

This executable performs no environment rollout and reads no new outcome
record.  It must run once before ``collect_safety_sources.py``.  The two
absolute transition requests per root are fixed independently of trajectory
length, terminal type, location, and all recovery outcomes.  A later collector
may materialize a request only if the frozen Safety trajectory reaches it.
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
HERE = ROOT / "diagnostics/recovery_entry_identifiability_v1"
AUTHORITATIVE_HELPER = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py"
REFERENCE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json"
REFERENCE_ASSET = Path(
    "/home/zhihan/research/02_C1_Toy_GiveWay/"
    "baseline_309_314/planning/wide_initial_states_200.npz"
)
PROTOCOL = HERE / "collection_protocol.json"
BUDGET = HERE / "experiment_budget.json"
MANIFEST = HERE / "development_source_manifest.json"
ANCHOR_PLAN = HERE / "generic_anchor_plan.csv"
OVERLAP_AUDIT = HERE / "development_source_overlap_audit.json"

SOURCE_COUNT = 120
IC_ROOT_SEED = 2026102601
FLOW_ROOT_SEED = 2026102602
ANCHOR_ROOT_SEED = 2026102699
ANCHOR_RULE_VERSION = "recovery_entry_uniform_absolute_v1"
ANCHORS_PER_SOURCE = 2
ANCHOR_LOW = 0
ANCHOR_HIGH_INCLUSIVE = 849
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty anchor plan")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_authoritative_helper():
    spec = importlib.util.spec_from_file_location("authoritative_wide_helper", AUTHORITATIVE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import authoritative WIDE generator helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor_steps(source_id: str) -> list[int]:
    """Return two deterministic uniform absolute steps without replacement."""
    material = f"{ANCHOR_RULE_VERSION}|{ANCHOR_ROOT_SEED}|{source_id}".encode()
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    values = rng.choice(
        np.arange(ANCHOR_LOW, ANCHOR_HIGH_INCLUSIVE + 1, dtype=np.int64),
        size=ANCHORS_PER_SOURCE,
        replace=False,
    )
    return sorted(int(value) for value in values)


def build_source_records(initials: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if initials.shape != (SOURCE_COUNT, 2, 2) or not np.isfinite(initials).all():
        raise ValueError(("invalid authoritative initial-state array", initials.shape))
    sources: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    for index, initial in enumerate(initials):
        source_id = f"recovery_entry_id_dev_v1_{index:04d}"
        requested = anchor_steps(source_id)
        sources.append({
            "source_id": source_id,
            "split": "development",
            "development_only": True,
            "episode_index": index,
            "rollout_id": index,
            "ic_root_seed": IC_ROOT_SEED,
            "ic_draw_index": index,
            "flow_root_seed": FLOW_ROOT_SEED,
            "initial_positions": initial.tolist(),
            "requested_anchor_steps": requested,
        })
        for slot, step in enumerate(requested):
            anchors.append({
                "development_cohort": "development",
                "source_id": source_id,
                "episode_index": index,
                "anchor_slot": slot,
                "requested_global_step": step,
                "requested_time_seconds": step * DT_SECONDS,
                "status_at_freeze": "REQUESTED_OUTCOME_UNKNOWN",
                "materialization_rule": (
                    "materialize iff nonterminal immediately before this predeclared absolute "
                    "transition; otherwise unavailable; never replace"
                ),
            })
    return sources, anchors


def assert_preparation_boundary() -> None:
    for path in (MANIFEST, ANCHOR_PLAN, OVERLAP_AUDIT):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
    outcome_artifacts = (
        HERE / "paired_branch_outcomes.csv",
        HERE / "statewise_recovery_advantage.csv",
        HERE / "queried_state_manifest.json",
        HERE / "queried_state_manifest.csv",
    )
    if (HERE / "runs").exists() or any(path.exists() for path in outcome_artifacts):
        raise RuntimeError("development roots must be frozen before every new outcome rollout")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    assert_preparation_boundary()
    protocol = json.loads(PROTOCOL.read_text())
    budget = json.loads(BUDGET.read_text())
    if protocol["development_root_count"] != SOURCE_COUNT:
        raise RuntimeError("protocol/source count mismatch")
    rule = protocol["generic_anchor_rule"]
    if rule["requested_anchors_per_root"] != ANCHORS_PER_SOURCE:
        raise RuntimeError("protocol/anchor count mismatch")
    if rule["absolute_step_range_inclusive"] != [ANCHOR_LOW, ANCHOR_HIGH_INCLUSIVE]:
        raise RuntimeError("protocol/anchor range mismatch")
    if protocol["official_horizon_steps"] != HORIZON_STEPS or protocol["dt_seconds"] != DT_SECONDS:
        raise RuntimeError("protocol horizon/dt mismatch")
    if budget["development_root_sources"] != SOURCE_COUNT:
        raise RuntimeError("budget/source count mismatch")
    if budget["requested_states_per_root"] != ANCHORS_PER_SOURCE:
        raise RuntimeError("budget/anchor count mismatch")
    if budget["maximum_new_branch_continuations"] > 12000 or budget["cpu_threads_max"] > 6:
        raise RuntimeError("global resource budget exceeds the predeclared ceiling")
    if budget.get("final_test_used") is not False or protocol.get("final_test_generated") is not False:
        raise RuntimeError("this preparation path is development-only")

    helper = load_authoritative_helper()
    reference = json.loads(REFERENCE.read_text())
    with np.load(REFERENCE_ASSET, allow_pickle=False) as payload:
        historical = np.asarray(payload["test_initial_positions"], dtype=np.float32)
    if not np.array_equal(helper.generate(2026090902, 200), historical):
        raise RuntimeError("authoritative WIDE generator replay mismatch")

    # collect_prior is intentionally called before writing any source/anchor
    # artifact.  It broadly audits prior manifests, datasets, and WIDE assets.
    prior, identities, _, prior_ids, audited_assets = helper.collect_prior()
    prior_seed_values = {int(value) for _, value, _ in identities}
    proposed_seeds = {IC_ROOT_SEED, FLOW_ROOT_SEED, ANCHOR_ROOT_SEED}
    seed_collisions = sorted(prior_seed_values.intersection(proposed_seeds))
    if seed_collisions:
        raise RuntimeError(("prior numeric-seed collision", seed_collisions))

    initials = helper.generate(IC_ROOT_SEED, SOURCE_COUNT)
    if len({row.tobytes() for row in initials}) != SOURCE_COUNT:
        raise RuntimeError("duplicate initial condition within the new development draw")
    sources, anchor_rows = build_source_records(initials)
    source_ids = {row["source_id"] for row in sources}
    id_collisions = sorted(source_ids.intersection(prior_ids))
    if id_collisions:
        raise RuntimeError(("prior source-ID collision", id_collisions))
    distances = np.linalg.norm(
        initials[:, None].astype(np.float64) - prior[None].astype(np.float64), axis=(2, 3)
    )
    exact_pairs = np.argwhere(distances == 0)
    if len(exact_pairs):
        raise RuntimeError(("exact prior initial-condition overlap", exact_pairs.tolist()))

    frozen_utc = datetime.now(timezone.utc).isoformat()
    overlap = {
        "schema": "recovery_entry_development_source_overlap_audit_v1",
        "status": "PASS",
        "audit_finished_before_new_outcomes": True,
        "audit_utc": frozen_utc,
        "new_source_count": SOURCE_COUNT,
        "exact_overlap_among_new_initials": 0,
        "exact_overlap_with_prior_initials": 0,
        "minimum_l2_distance_to_prior_initial": float(distances.min()),
        "prior_initial_records_checked": int(len(prior)),
        "prior_unique_initials": len({row.tobytes() for row in prior}),
        "prior_assets_checked": len(set(audited_assets)),
        "prior_numeric_seed_records_checked": len(identities),
        "seed_collisions": [],
        "source_id_collisions": [],
        "replacement_count": 0,
        "rejection_outcome_or_geometry_filtering": False,
    }
    overlap["content_sha256"] = canonical_hash(overlap)
    atomic_json(OVERLAP_AUDIT, overlap)

    anchor_rule = {
        "version": ANCHOR_RULE_VERSION,
        "anchor_root_seed": ANCHOR_ROOT_SEED,
        "absolute_step_range_inclusive": [ANCHOR_LOW, ANCHOR_HIGH_INCLUSIVE],
        "requested_anchors_per_root": ANCHORS_PER_SOURCE,
        "sampling": "source-ID-derived deterministic RNG; discrete uniform without replacement",
        "materialization": (
            "materialize only predeclared steps reached while Safety is nonterminal; "
            "unavailable requests are skipped without replacement"
        ),
        "uses_terminal_event_or_failure_type": False,
        "uses_terminal_relative_or_deadlock_onset_time": False,
        "uses_position_or_spatial_region": False,
        "uses_prior_error_or_recovery_outcome": False,
    }
    manifest = {
        "schema": "recovery_entry_development_source_manifest_v1",
        "status": "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION",
        "frozen_utc": frozen_utc,
        "outcomes_observed_by_this_script": False,
        "environment_rollouts_launched_by_this_script": 0,
        "development_only": True,
        "final_test_generated_or_used": False,
        "source_count": SOURCE_COUNT,
        "source_derivatives_grouped_by_root": True,
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
        "anchor_rule": anchor_rule,
        "requested_anchor_count": len(anchor_rows),
        "official_horizon_steps": HORIZON_STEPS,
        "dt_seconds": DT_SECONDS,
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "budget_path": str(BUDGET),
        "budget_sha256": sha256(BUDGET),
        "overlap_audit_path": str(OVERLAP_AUDIT),
        "overlap_audit_sha256": sha256(OVERLAP_AUDIT),
        "preparation_script": str(Path(__file__).resolve()),
        "preparation_script_sha256": sha256(Path(__file__).resolve()),
        "sources": {"development": sources},
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    atomic_json(MANIFEST, manifest)
    atomic_csv(ANCHOR_PLAN, anchor_rows)

    print(json.dumps({
        "status": "FROZEN",
        "development_source_manifest": str(MANIFEST),
        "development_source_manifest_sha256": sha256(MANIFEST),
        "anchor_plan": str(ANCHOR_PLAN),
        "anchor_plan_sha256": sha256(ANCHOR_PLAN),
        "root_sources": SOURCE_COUNT,
        "requested_generic_states": len(anchor_rows),
        "environment_rollouts": 0,
        "final_test_generated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
