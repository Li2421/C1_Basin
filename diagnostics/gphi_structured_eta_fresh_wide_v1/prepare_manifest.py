"""Freeze the final fresh unconditioned WIDE cohort before any rollout."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
REFERENCE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json"
REFERENCE_ASSET = SYSROOT / "baseline_309_314/planning/wide_initial_states_200.npz"
MANIFEST = HERE / "fresh_wide_manifest.json"
OVERLAP = HERE / "overlap_audit.json"
EPISODES = 200
IC_GENERATOR_SEED = 2026092601
FLOW_ROOT_SEED = 2026092602
X_RANGE = (0.55, 1.05)
Y_RANGE = (-0.025, 0.025)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def generate(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    absolute_x = rng.uniform(X_RANGE[0], X_RANGE[1], size=(count, 2))
    y = rng.uniform(Y_RANGE[0], Y_RANGE[1], size=(count, 2))
    positions = np.empty((count, 2, 2), dtype=np.float64)
    positions[:, 0, 0] = -absolute_x[:, 0]
    positions[:, 1, 0] = absolute_x[:, 1]
    positions[:, :, 1] = y
    return positions.astype(np.float32)


def extract_json(value: Any, asset: Path, initials: list, identities: list, ids: set[str]) -> None:
    if isinstance(value, dict):
        if "initial_positions" in value:
            try:
                initial = np.asarray(value["initial_positions"], dtype=np.float32)
                if initial.shape == (2, 2) and np.isfinite(initial).all():
                    initials.append((initial, str(asset)))
            except (TypeError, ValueError):
                pass
        for key, item in value.items():
            if key in {"ic_seed", "ic_generator_seed", "flow_root_seed", "source_seed", "flow_seed", "root_seed"}:
                try:
                    identities.append((key, int(item), str(asset)))
                except (TypeError, ValueError):
                    pass
            if key in {"source_id", "episode_id", "case_id"} and isinstance(item, str):
                ids.add(item)
            extract_json(item, asset, initials, identities, ids)
    elif isinstance(value, list):
        for item in value:
            extract_json(item, asset, initials, identities, ids)


def initial_from_npz(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "initial_positions" not in payload.files:
                return None
            value = np.asarray(payload["initial_positions"], dtype=np.float32)
    except (OSError, ValueError):
        return None
    return value if value.shape == (2, 2) and np.isfinite(value).all() else None


def collect_prior() -> tuple[np.ndarray, list, list, set[str], list[str]]:
    initials: list[tuple[np.ndarray, str]] = []
    identities: list[tuple[str, int, str]] = []
    ids: set[str] = set()
    audited_assets: list[str] = []

    # Broad manifest audit captures all earlier full-episode benchmarks and
    # downstream diagnostic source manifests without assuming their names.
    for path in sorted((ROOT / "diagnostics").rglob("*.json")):
        if HERE in path.parents or path.stat().st_size > 50_000_000:
            continue
        if not any(token in path.name.lower() for token in ("manifest", "seed", "config", "audit")):
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        extract_json(payload, path, initials, identities, ids)
        audited_assets.append(str(path))

    # Oracle dataset state manifests reference complete source episodes.
    source_paths: set[Path] = set()
    for directory in [
        *(ROOT / f"diagnostics/gphi_training_dataset_v{version}" for version in (1, 2, 3, 4)),
        ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1",
        ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1",
        ROOT / "diagnostics/gphi_training_dataset_dagger_k1_v1",
    ]:
        for name in ("state_manifest.jsonl", "startup_state_manifest.jsonl", "sample_metadata.jsonl"):
            path = directory / name
            if not path.is_file():
                continue
            audited_assets.append(str(path))
            for line in path.read_text().splitlines():
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = row.get("source_path") or row.get("source_episode_path")
                if source:
                    source_paths.add(Path(source))
                extract_json(row, path, initials, identities, ids)
    for path in sorted(source_paths):
        if path.is_file():
            value = initial_from_npz(path)
            if value is not None:
                initials.append((value, str(path)))

    # FlowBC training source episodes are also excluded explicitly.
    raw_root = SYSROOT / "datasets/give_way_si_short_v1/raw"
    for pair in range(250):
        path = raw_root / f"episode_{2 * pair:04d}.npz"
        value = initial_from_npz(path)
        if value is not None:
            initials.append((value, str(path)))
    audited_assets.append(str(raw_root))

    # Both frozen authoritative WIDE validation/test arrays.
    with np.load(REFERENCE_ASSET, allow_pickle=False) as payload:
        for key in ("val_initial_positions", "test_initial_positions"):
            for value in np.asarray(payload[key], dtype=np.float32):
                initials.append((value, f"{REFERENCE_ASSET}:{key}"))
    audited_assets.append(str(REFERENCE_ASSET))
    if not initials:
        raise RuntimeError("no prior initial states found")
    return np.stack([row[0] for row in initials]), identities, initials, ids, audited_assets


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    existing_raw = list((HERE / "runs").rglob("episode_*.json")) if (HERE / "runs").exists() else []
    if existing_raw:
        raise RuntimeError("manifest must precede every rollout")
    if MANIFEST.exists() or OVERLAP.exists():
        raise RuntimeError("frozen manifest already exists")

    reference = json.loads(REFERENCE.read_text())
    with np.load(REFERENCE_ASSET, allow_pickle=False) as payload:
        historical = np.asarray(payload["test_initial_positions"], dtype=np.float32)
    if not np.array_equal(generate(2026090902, 200), historical):
        raise RuntimeError("authoritative generator replay mismatch")

    fresh = generate(IC_GENERATOR_SEED, EPISODES)
    if len({row.tobytes() for row in fresh}) != EPISODES:
        raise RuntimeError("duplicate fresh IC")
    prior, identities, initial_records, prior_ids, audited_assets = collect_prior()
    distances = np.linalg.norm(fresh[:, None].astype(np.float64) - prior[None].astype(np.float64), axis=(2, 3))
    exact_pairs = np.argwhere(distances == 0)
    prior_seed_values = {value for _, value, _ in identities}
    seed_collisions = sorted(prior_seed_values.intersection({IC_GENERATOR_SEED, FLOW_ROOT_SEED}))
    new_source_ids = {f"structured_eta_final_wide_v1_{index:04d}" for index in range(EPISODES)}
    source_id_collisions = sorted(new_source_ids.intersection(prior_ids))
    if len(exact_pairs) or seed_collisions or source_id_collisions:
        raise RuntimeError(("fresh overlap", exact_pairs.tolist(), seed_collisions, source_id_collisions))

    frozen_utc = datetime.now(timezone.utc).isoformat()
    overlap = {
        "schema": "structured_eta_final_wide_overlap_v1", "status": "PASS",
        "fresh_episode_count": EPISODES, "exact_initial_state_match_count": 0,
        "exact_initial_state_matches": [], "minimum_l2_distance_to_any_prior_initial": float(distances.min()),
        "prior_initial_records_checked": int(len(prior)),
        "prior_initial_unique_count": len({row.tobytes() for row in prior}),
        "prior_json_or_source_assets_checked": len(set(audited_assets)),
        "numeric_seed_values_checked": len(identities), "new_seed_collisions": [],
        "source_id_collisions": [], "new_ic_generator_seed": IC_GENERATOR_SEED,
        "new_flow_root_seed": FLOW_ROOT_SEED, "replacement_count": 0,
        "rejection_or_outcome_filtering": False, "audit_finished_before_rollout": True,
        "audit_utc": frozen_utc,
        "cohorts_explicitly_covered": [
            "G_phi V1/V2/V3/V4/startup-complete/strict-deadlock/k1 datasets",
            "historical WIDE", "previous fresh WIDE", "narrow pilot", "cadence/burst audits",
            "strict-deadlock diagnostics", "teacher-takeover diagnostics", "FlowBC training sources",
        ],
    }
    overlap["content_sha256"] = canonical_hash(overlap)
    atomic_json(OVERLAP, overlap)

    flow_semantics = (
        f"episode_key=fold_in(PRNGKey({FLOW_ROOT_SEED}), rollout_id); "
        "step_key=fold_in(episode_key, physical_step)"
    )
    manifest = {
        "schema": "structured_eta_final_fresh_wide_manifest_v1",
        "purpose": "final_single_use_heldout_generalization_test",
        "frozen_before_rollout": True, "frozen_utc": frozen_utc,
        "raw_rollout_records_present_at_freeze": 0, "episode_count": EPISODES,
        "generator": {
            "name": reference["authoritative_suite"]["metadata"]["name"],
            "implementation": (
                "numpy.default_rng(seed); abs_x~U(0.55,1.05),(N,2); "
                "y~U(-0.025,0.025),(N,2); signs [-,+]; final float32"
            ),
            "ic_generator_seed": IC_GENERATOR_SEED,
            "x_absolute_uniform": list(X_RANGE), "y_uniform": list(Y_RANGE),
            "no_rejection_sampling": True, "authoritative_replay_seed": 2026090902,
            "authoritative_replay_bitwise_exact": True,
            "authoritative_asset": str(REFERENCE_ASSET),
            "authoritative_asset_sha256": sha256(REFERENCE_ASSET),
            "generator_script": str(Path(__file__).resolve()),
            "generator_script_sha256": sha256(Path(__file__).resolve()),
        },
        "flow_randomness": {
            "root_seed": FLOW_ROOT_SEED, "semantics": flow_semantics,
            "matched_across_controllers": True, "new_root_seed": True,
        },
        "controller_protocol": {
            "controllers": ["Safety", "Direct-g H8", "Structured eta"],
            "direct_g_cadence_h": 8, "direct_g_burst_length": 1,
            "structured_eta_prediction_steps": [0], "structured_eta_fixed_for_episode": True,
            "structured_feedback_dense": True, "horizon_steps": 850, "dt_seconds": 0.05,
            "no_tuning": True,
        },
        "environment": reference["environment"], "cbf": reference["cbf"],
        "outcome_protocol": reference["outcome_protocol"],
        "overlap_audit_path": str(OVERLAP), "overlap_audit_sha256": sha256(OVERLAP),
        "episodes": [
            {
                "episode_index": index, "source_id": f"structured_eta_final_wide_v1_{index:04d}",
                "ic_generator_seed": IC_GENERATOR_SEED, "ic_draw_index": index,
                "rollout_id": index, "flow_root_seed": FLOW_ROOT_SEED,
                "initial_positions": value.tolist(),
            }
            for index, value in enumerate(fresh)
        ],
    }
    manifest["content_sha256"] = canonical_hash(manifest)
    atomic_json(MANIFEST, manifest)
    print(json.dumps({
        "status": "FROZEN", "manifest": str(MANIFEST), "manifest_file_sha256": sha256(MANIFEST),
        "manifest_content_sha256": manifest["content_sha256"], "overlap": overlap,
        "observed_ranges": {
            "absolute_x": [float(np.abs(fresh[:, :, 0]).min()), float(np.abs(fresh[:, :, 0]).max())],
            "y": [float(fresh[:, :, 1].min()), float(fresh[:, :, 1].max())],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
