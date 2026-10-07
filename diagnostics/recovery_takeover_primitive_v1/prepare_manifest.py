"""Freeze a new development-only WIDE cohort and takeover protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/recovery_takeover_primitive_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
SOURCE_PREP = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py"
REFERENCE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json"
REFERENCE_ASSET = SYSROOT / "baseline_309_314/planning/wide_initial_states_200.npz"
FLOW = SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
DIRECT = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz"
ETA = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz"
MANIFEST = HERE / "development_manifest.json"
IC_SEED = 2026092701
FLOW_SEED = 2026092702
EPISODES = 200
EXPECTED = {
    "flow": "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32",
    "direct": "340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700",
    "eta": "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_source_module():
    spec = importlib.util.spec_from_file_location("frozen_wide_manifest_helpers", SOURCE_PREP)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import authoritative WIDE helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if MANIFEST.exists() or list((HERE / "runs").rglob("*.json")):
        raise RuntimeError("development manifest must be frozen before rollout")
    for name, path in (("flow", FLOW), ("direct", DIRECT), ("eta", ETA)):
        if sha256(path) != EXPECTED[name]:
            raise RuntimeError((name, "checkpoint mismatch", sha256(path)))
    source = load_source_module()
    reference = json.loads(REFERENCE.read_text())
    with np.load(REFERENCE_ASSET, allow_pickle=False) as payload:
        authoritative = np.asarray(payload["test_initial_positions"], dtype=np.float32)
    if not np.array_equal(source.generate(2026090902, EPISODES), authoritative):
        raise RuntimeError("authoritative WIDE generator replay mismatch")
    fresh = source.generate(IC_SEED, EPISODES)
    if len({row.tobytes() for row in fresh}) != EPISODES:
        raise RuntimeError("duplicate development IC")
    prior, identities, _, prior_ids, audited_assets = source.collect_prior()
    distances = np.linalg.norm(fresh[:, None].astype(np.float64) - prior[None].astype(np.float64), axis=(2, 3))
    exact = np.argwhere(distances == 0)
    seed_values = {value for _, value, _ in identities}
    seed_collision = sorted(seed_values.intersection({IC_SEED, FLOW_SEED}))
    source_ids = {f"recovery_takeover_dev_v1_{index:04d}" for index in range(EPISODES)}
    id_collision = sorted(source_ids.intersection(prior_ids))
    if len(exact) or seed_collision or id_collision:
        raise RuntimeError(("development overlap", exact.tolist(), seed_collision, id_collision))
    frozen = datetime.now(timezone.utc).isoformat()
    flow_semantics = f"episode_key=fold_in(PRNGKey({FLOW_SEED}), rollout_id); step_key=fold_in(episode_key, global_physical_step)"
    manifest = {
        "schema": "recovery_takeover_development_manifest_v1",
        "purpose": "development_only_recovery_primitive_feasibility",
        "frozen_before_any_outcome": True, "frozen_utc": frozen,
        "episode_count": EPISODES, "raw_records_present_at_freeze": 0,
        "generator": {
            "name": reference["authoritative_suite"]["metadata"]["name"],
            "implementation": "numpy.default_rng(seed); abs_x~U(0.55,1.05),(N,2); y~U(-0.025,0.025),(N,2); signs [-,+]; final float32",
            "ic_root_seed": IC_SEED, "x_absolute_uniform": [0.55, 1.05], "y_uniform": [-0.025, 0.025],
            "authoritative_asset": str(REFERENCE_ASSET), "authoritative_asset_sha256": sha256(REFERENCE_ASSET),
            "authoritative_replay_seed": 2026090902, "authoritative_replay_bitwise_exact": True,
            "no_rejection_or_targeting": True,
        },
        "flow_randomness": {"root_seed": FLOW_SEED, "semantics": flow_semantics, "stateless_global_step_keys": True},
        "environment": reference["environment"], "cbf": reference["cbf"], "outcome_protocol": reference["outcome_protocol"],
        "controllers": {
            "continue_safety": "u_flow -> Pi_safe -> environment",
            "dense_direct_g": {"path": str(DIRECT), "sha256": sha256(DIRECT), "semantics": "re-query direct-g every post-takeover physical step; no H8"},
            "persistent_structured_eta": {"path": str(ETA), "sha256": sha256(ETA), "semantics": "predict once at exact takeover state, freeze eta, recompute basis feedback densely until termination"},
            "flowbc": {"path": str(FLOW), "sha256": sha256(FLOW)},
        },
        "takeover_protocol": {
            "terminal_relative_offsets_steps": [160, 80, 40, 20],
            "terminal_relative_offsets_seconds": [8.0, 4.0, 2.0, 1.0],
            "controllers": ["continue_safety", "dense_direct_g", "persistent_structured_eta"],
            "no_exit_to_safety": True, "no_periodic_or_burst_semantics": True,
            "no_hybrid": True, "no_trigger_training": True, "no_oracle_online": True,
        },
        "nominal_matching_rule": {
            "frozen_before_safety_outcomes": True,
            "candidates": "Safety-success episodes with terminal_step > takeover global step, sorted by episode_index",
            "choice": "sha256('recovery_nominal_match_v1|failure_episode|offset_steps|takeover_step') interpreted as integer modulo candidate count",
            "reuse_allowed": True, "no_recovery_outcome_used": True,
        },
        "freshness_audit": {
            "status": "PASS", "exact_prior_initial_overlap": 0,
            "minimum_l2_to_prior_initial": float(distances.min()), "prior_initial_records_checked": int(len(prior)),
            "prior_unique_initials": len({row.tobytes() for row in prior}),
            "prior_assets_checked": len(set(audited_assets)), "numeric_seed_values_checked": len(identities),
            "seed_collisions": [], "source_id_collisions": [],
        },
        "episodes": [
            {"episode_index": index, "source_id": f"recovery_takeover_dev_v1_{index:04d}",
             "rollout_id": index, "ic_draw_index": index, "initial_positions": positions.tolist()}
            for index, positions in enumerate(fresh)
        ],
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    atomic_json(MANIFEST, manifest)
    print(json.dumps({"status": "FROZEN", "path": str(MANIFEST), "sha256": sha256(MANIFEST),
                      "content_sha256": manifest["content_sha256"], "freshness": manifest["freshness_audit"]}, indent=2))


if __name__ == "__main__":
    main()
