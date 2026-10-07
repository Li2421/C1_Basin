"""Freeze a fresh IID WIDE cohort and prove zero prior-source overlap."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_h8_fresh_unseen_generalization_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
REFERENCE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json"
REFERENCE_ASSET = SYSROOT / "baseline_309_314/planning/wide_initial_states_200.npz"
MANIFEST = HERE / "fresh_test_manifest.json"
OVERLAP = HERE / "overlap_audit.json"
EPISODES = 200
IC_GENERATOR_SEED = 2026092501
FLOW_ROOT_SEED = 2026092502
X_RANGE = (0.55, 1.05)
Y_RANGE = (-0.025, 0.025)
FLOW_SEMANTICS = (
    "episode_key=fold_in(PRNGKey(2026092502), rollout_id); "
    "step_key=fold_in(episode_key, physical_step)"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def generate(seed: int, count: int) -> np.ndarray:
    """Exact authoritative call order, validated against the frozen old test set."""
    rng = np.random.default_rng(seed)
    absolute_x = rng.uniform(X_RANGE[0], X_RANGE[1], size=(count, 2))
    y = rng.uniform(Y_RANGE[0], Y_RANGE[1], size=(count, 2))
    positions = np.empty((count, 2, 2), dtype=np.float64)
    positions[:, 0, 0] = -absolute_x[:, 0]
    positions[:, 1, 0] = absolute_x[:, 1]
    positions[:, :, 1] = y
    return positions.astype(np.float32)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def initial_from_npz(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "initial_positions" not in z.files:
                return None
            value = np.asarray(z["initial_positions"], dtype=np.float32)
    except (OSError, ValueError):
        return None
    return value if value.shape == (2, 2) else None


def collect_prior_initials() -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    values: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    numeric_identities: list[dict[str, Any]] = []

    dataset_dirs = [
        ROOT / f"diagnostics/gphi_training_dataset_v{version}"
        for version in (1, 2, 3, 4)
    ] + [ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"]
    source_paths: dict[str, dict[str, Any]] = {}
    for directory in dataset_dirs:
        for name in ("state_manifest.jsonl", "startup_state_manifest.jsonl"):
            path = directory / name
            if not path.is_file():
                continue
            for row in read_jsonl(path):
                for key in ("source_seed", "source_rng_id", "source_rollout_id", "rng_namespace"):
                    if row.get(key) is not None:
                        numeric_identities.append({
                            "asset": str(path), "field": key,
                            "value": int(row[key]), "state_id": str(row.get("state_id")),
                        })
                source = row.get("source_path")
                if source:
                    source_paths.setdefault(str(Path(source).resolve()), {
                        "asset": str(path), "state_id": str(row.get("state_id")),
                    })
    for source, meta in source_paths.items():
        path = Path(source)
        if path.is_file():
            initial = initial_from_npz(path)
            if initial is not None:
                values.append(initial)
                provenance.append({"kind": "gphi_oracle_source", "path": source, **meta})

    flow_raw = SYSROOT / "datasets/give_way_si_short_v1/raw"
    for pair in range(250):
        path = flow_raw / f"episode_{2*pair:04d}.npz"
        initial = initial_from_npz(path)
        if initial is not None:
            values.append(initial)
            provenance.append({"kind": "flowbc_training_source", "path": str(path), "pair": pair})

    with np.load(REFERENCE_ASSET, allow_pickle=False) as z:
        for key in ("val_initial_positions", "test_initial_positions"):
            for index, initial in enumerate(np.asarray(z[key], dtype=np.float32)):
                values.append(initial)
                provenance.append({"kind": "historical_wide", "path": str(REFERENCE_ASSET), "array": key, "index": index})

    prior_manifests = [
        ROOT / "diagnostics/gphi_closed_loop_pilot_v1/evaluation_seed_manifest.json",
        ROOT / "diagnostics/gphi_fixed_cadence_ablation_v1/evaluation_seed_manifest.json",
        ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json",
    ]
    for path in prior_manifests:
        payload = json.loads(path.read_text())
        for episode in payload.get("episodes", []):
            initial = np.asarray(episode["initial_positions"], dtype=np.float32)
            values.append(initial)
            provenance.append({"kind": "prior_evaluation", "path": str(path), "episode_index": episode.get("episode_index")})
            for key in ("ic_seed", "flow_seed", "rollout_id"):
                if episode.get(key) is not None:
                    numeric_identities.append({"asset": str(path), "field": key, "value": int(episode[key]), "episode_index": episode.get("episode_index")})
        flow = payload.get("flow_randomness")
        if isinstance(flow, dict) and flow.get("root_seed") is not None:
            numeric_identities.append({"asset": str(path), "field": "flow_root_seed", "value": int(flow["root_seed"])})

    if not values:
        raise RuntimeError("no prior initials discovered")
    return np.stack(values), provenance, numeric_identities


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    raw_root = HERE / "runs/production/raw"
    raw_count = len(list(raw_root.rglob("episode_*.json"))) if raw_root.exists() else 0
    if raw_count:
        raise RuntimeError("manifest must be frozen before rollout records exist")
    if MANIFEST.exists() or OVERLAP.exists():
        raise RuntimeError("fresh manifest already frozen; refuse regeneration")

    reference = json.loads(REFERENCE.read_text())
    with np.load(REFERENCE_ASSET, allow_pickle=False) as z:
        historical_test = np.asarray(z["test_initial_positions"], dtype=np.float32)
    generator_replay = generate(2026090902, 200)
    if not np.array_equal(generator_replay, historical_test):
        raise RuntimeError("recovered WIDE generator does not exactly replay authoritative suite")

    fresh = generate(IC_GENERATOR_SEED, EPISODES)
    if len({row.tobytes() for row in fresh}) != EPISODES:
        raise RuntimeError("fresh generator produced duplicate ICs")
    prior, provenance, numeric = collect_prior_initials()
    distances = np.linalg.norm(
        fresh[:, None].astype(np.float64) - prior[None].astype(np.float64),
        axis=(2, 3),
    )
    exact_pairs = np.argwhere(distances == 0.0)
    prior_values = {item["value"] for item in numeric}
    seed_collisions = sorted(prior_values.intersection({IC_GENERATOR_SEED, FLOW_ROOT_SEED}))
    if len(exact_pairs) or seed_collisions:
        raise RuntimeError(("fresh overlap detected before evaluation", exact_pairs.tolist(), seed_collisions))

    frozen_utc = datetime.now(timezone.utc).isoformat()
    overlap = {
        "schema": "gphi_h8_fresh_overlap_audit_v1",
        "status": "PASS",
        "fresh_episode_count": EPISODES,
        "exact_initial_state_matches": [],
        "exact_initial_state_match_count": 0,
        "minimum_l2_distance_to_any_prior_initial": float(distances.min()),
        "prior_initial_records_checked": len(prior),
        "prior_initial_unique_count": len({row.tobytes() for row in prior}),
        "prior_source_assets_checked": len({item["path"] for item in provenance}),
        "numeric_seed_or_stream_values_checked": len(numeric),
        "new_seed_collisions": seed_collisions,
        "new_ic_generator_seed": IC_GENERATOR_SEED,
        "new_flow_root_seed": FLOW_ROOT_SEED,
        "gphi_dataset_versions_audited": ["V1", "V2", "V3", "V4", "startup_complete_v1"],
        "prior_evaluations_audited": [
            "narrow startup-complete pilot", "fixed-cadence nominal ablation",
            "historical WIDE cadence benchmark", "extended-horizon audit (same historical WIDE cohort)",
        ],
        "replacement_count": 0,
        "rejection_or_outcome_filtering": False,
        "audit_finished_before_rollout": True,
        "audit_utc": frozen_utc,
    }
    overlap["content_sha256"] = canonical_hash(overlap)
    write_json(OVERLAP, overlap)

    payload = {
        "schema": "gphi_h8_fresh_wide_test_manifest_v1",
        "purpose": "single_use_fully_unseen_generalization_test",
        "frozen_before_rollout": True,
        "frozen_utc": frozen_utc,
        "raw_rollout_records_present_at_freeze": 0,
        "episode_count": EPISODES,
        "generator": {
            "name": reference["authoritative_suite"]["metadata"]["name"],
            "implementation": "numpy.random.default_rng(seed); absolute_x=uniform(0.55,1.05,size=(N,2)); y=uniform(-0.025,0.025,size=(N,2)); x signs fixed [-,+]; cast final [N,2,2] to float32",
            "ic_generator_seed": IC_GENERATOR_SEED,
            "x_absolute_uniform": list(X_RANGE),
            "y_uniform": list(Y_RANGE),
            "no_rejection_sampling": True,
            "authoritative_replay_seed": 2026090902,
            "authoritative_replay_bitwise_exact": True,
            "authoritative_asset": str(REFERENCE_ASSET),
            "authoritative_asset_sha256": sha256(REFERENCE_ASSET),
            "generator_script": str(Path(__file__).resolve()),
            "generator_script_sha256": sha256(Path(__file__).resolve()),
        },
        "flow_randomness": {
            "root_seed": FLOW_ROOT_SEED,
            "semantics": FLOW_SEMANTICS,
            "matched_across_controllers": True,
            "new_root_seed_not_used_by_audited_prior_assets": True,
        },
        "controller_protocol": {
            "controllers": ["Safety", "H=8"],
            "cadence_h": 8,
            "correction_one_step_only": True,
            "held_correction": False,
            "horizon_steps": 850,
            "dt_seconds": 0.05,
            "no_tuning": True,
        },
        "environment": reference["environment"],
        "cbf": reference["cbf"],
        "outcome_protocol": reference["outcome_protocol"],
        "overlap_audit_path": str(OVERLAP),
        "overlap_audit_sha256": sha256(OVERLAP),
        "episodes": [
            {
                "episode_index": index,
                "source_id": f"fresh_wide_h8_v1_{index:04d}",
                "ic_generator_seed": IC_GENERATOR_SEED,
                "ic_draw_index": index,
                "rollout_id": index,
                "flow_root_seed": FLOW_ROOT_SEED,
                "initial_positions": initial.tolist(),
            }
            for index, initial in enumerate(fresh)
        ],
    }
    payload["content_sha256"] = canonical_hash(payload)
    write_json(MANIFEST, payload)
    print(json.dumps({
        "status": "FROZEN", "manifest": str(MANIFEST),
        "manifest_file_sha256": sha256(MANIFEST),
        "manifest_content_sha256": payload["content_sha256"],
        "overlap": overlap,
        "observed_ranges": {
            "absolute_x": [float(np.abs(fresh[:, :, 0]).min()), float(np.abs(fresh[:, :, 0]).max())],
            "y": [float(fresh[:, :, 1].min()), float(fresh[:, :, 1].max())],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
