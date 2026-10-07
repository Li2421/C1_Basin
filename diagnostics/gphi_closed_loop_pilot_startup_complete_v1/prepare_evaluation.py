"""Freeze fresh matched IC/Flow seeds before any pilot controller is run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pilot_common import (
    DATASET, FLOW_CHECKPOINT, HERE, ROOT, SYSROOT, V3_DATASET,
    assert_frozen_sources, canonical_json_hash, read_jsonl, sha256, write_json,
)


DEFAULT_IC_BASE = 3_307_000_000
DEFAULT_FLOW_BASE = 3_317_000_000


def numeric_provenance(rows: list[dict[str, Any]]) -> set[int]:
    names = {
        "source_seed", "source_rng_id", "rng_namespace", "flow_seed",
        "ic_seed", "evaluation_seed", "seed", "source_rollout_id",
    }
    result: set[int] = set()
    for row in rows:
        for name in names:
            value = row.get(name)
            if isinstance(value, (int, np.integer)):
                result.add(int(value))
    return result


def source_initial_positions(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    seen_paths: set[Path] = set()
    values: list[np.ndarray] = []
    for row in rows:
        raw = row.get("source_path")
        if not raw:
            continue
        path = Path(raw)
        if path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                if "initial_positions" in data.files:
                    value = np.asarray(data["initial_positions"], dtype=np.float64)
                    if value.shape == (2, 2):
                        values.append(value)
        except (OSError, ValueError):
            continue
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--ic-seed-base", type=int, default=DEFAULT_IC_BASE)
    parser.add_argument("--flow-seed-base", type=int, default=DEFAULT_FLOW_BASE)
    parser.add_argument("--output", type=Path, default=HERE / "evaluation_seed_manifest.json")
    parser.add_argument("--purpose", default="production", choices=("production", "smoke"))
    args = parser.parse_args()
    if not 1 <= args.episodes <= 256:
        raise ValueError("episode count must be in [1,256]")
    if not all(0 <= value <= 2**32 - args.episodes - 1 for value in (args.ic_seed_base, args.flow_seed_base)):
        raise ValueError("seed bases must admit the requested uint32 range")
    source_audit = assert_frozen_sources()
    sys.path.insert(0, str(SYSROOT / "scripts"))
    from giveway_initial_state import sample_initial_positions

    v3_manifest = V3_DATASET / "state_manifest.jsonl"
    startup_manifest = DATASET / "startup_state_manifest.jsonl"
    if not v3_manifest.is_file() or not startup_manifest.is_file():
        raise FileNotFoundError("V3/startup state manifests must exist before seed freezing")
    source_rows = read_jsonl(v3_manifest) + read_jsonl(startup_manifest)
    used_numbers = numeric_provenance(source_rows)
    audited_seed_assets: dict[str, str] = {
        str(v3_manifest): sha256(v3_manifest), str(startup_manifest): sha256(startup_manifest),
    }
    # Include the oracle/Flow seed domains that created supervised samples, not
    # only source-trajectory identifiers from the state manifests.
    for protocol_path in (V3_DATASET / "protocol.json", DATASET / "protocol.json"):
        if protocol_path.exists():
            protocol = json.loads(protocol_path.read_text())
            used_numbers.update(int(value) for value in protocol.get("oracle_seed_default", []))
            audited_seed_assets[str(protocol_path)] = sha256(protocol_path)
    for samples_path in (V3_DATASET / "samples.npz", DATASET / "samples.npz"):
        if samples_path.exists():
            with np.load(samples_path, allow_pickle=False) as values:
                if "flow_seed" in values.files:
                    used_numbers.update(int(value) for value in np.unique(values["flow_seed"]))
            audited_seed_assets[str(samples_path)] = sha256(samples_path)
    source_initials = source_initial_positions(source_rows)
    episodes = []
    evaluation_initials = []
    seed_collisions = []
    for episode_index in range(args.episodes):
        ic_seed = args.ic_seed_base + episode_index
        flow_seed = args.flow_seed_base + episode_index
        if ic_seed in used_numbers or flow_seed in used_numbers:
            seed_collisions.append({"episode_index": episode_index, "ic_seed": ic_seed, "flow_seed": flow_seed})
        initial = np.asarray(
            sample_initial_positions(np.random.default_rng(ic_seed)), dtype=np.float64
        )
        evaluation_initials.append(initial)
        episodes.append({
            "episode_index": episode_index, "ic_seed": ic_seed, "flow_seed": flow_seed,
            "initial_positions": initial.tolist(),
        })
    stacked = np.stack(evaluation_initials)
    flat = stacked.reshape(len(stacked), -1)
    unique_initials = len({row.tobytes() for row in flat})
    exact_source_matches = []
    minimum_source_distance = float("inf")
    for eval_index, initial in enumerate(evaluation_initials):
        for source_index, source in enumerate(source_initials):
            distance = float(np.linalg.norm(initial - source))
            minimum_source_distance = min(minimum_source_distance, distance)
            if np.array_equal(initial, source):
                exact_source_matches.append({"episode_index": eval_index, "source_initial_index": source_index})
    if not source_initials:
        minimum_source_distance = float("nan")
    overlap = {
        "status": "PASS" if not seed_collisions and not exact_source_matches and unique_initials == args.episodes else "FAIL",
        "numeric_seed_or_namespace_collisions": seed_collisions,
        "exact_initial_position_matches": exact_source_matches,
        "source_initial_conditions_audited": len(source_initials),
        "unique_evaluation_initial_conditions": unique_initials,
        "minimum_l2_to_known_training_source_initial": minimum_source_distance,
        "audited_seed_assets": audited_seed_assets,
        "note": "Numeric IDs are conservatively compared across source_seed/source_rng_id/rng_namespace/flow_seed fields; exact IC arrays are also compared when source trajectories expose them.",
    }
    if overlap["status"] != "PASS":
        raise RuntimeError(("evaluation overlap audit failed", overlap))
    payload = {
        "schema": "startup_complete_gphi_fresh_matched_evaluation_v1",
        "purpose": args.purpose, "episodes": episodes,
        "episode_count": args.episodes, "ic_seed_base": args.ic_seed_base,
        "flow_seed_base": args.flow_seed_base,
        "initial_condition_distribution": {
            "implementation": str(SYSROOT / "scripts/giveway_initial_state.py"),
            "sha256": sha256(SYSROOT / "scripts/giveway_initial_state.py"),
        },
        "flow_randomness": "per-episode PRNGKey(flow_seed), then fold_in(key, physical_step)",
        "controller_pairing": "Safety and learned G_phi use identical initial_positions and Flow keys",
        "training_overlap_audit": overlap,
        "frozen_source_audit": source_audit,
        "flow_checkpoint": str(FLOW_CHECKPOINT), "flow_checkpoint_sha256": sha256(FLOW_CHECKPOINT),
    }
    payload["content_sha256"] = canonical_json_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if canonical_json_hash({k: v for k, v in existing.items() if k != "content_sha256"}) != payload["content_sha256"]:
            raise RuntimeError(f"refusing to overwrite different frozen seed manifest: {args.output}")
    else:
        write_json(args.output, payload)
    print(json.dumps({
        "status": "PASS", "output": str(args.output), "episodes": args.episodes,
        "content_sha256": payload["content_sha256"], "minimum_source_ic_distance": minimum_source_distance,
    }, indent=2))


if __name__ == "__main__":
    main()
