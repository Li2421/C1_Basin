"""Freeze the authoritative historical WIDE benchmark before any rollout.

The benchmark is not regenerated.  This script copies the exact float32
``test_initial_positions`` values from the historical 200-case NPZ into a
content-hashed JSON manifest and audits whether those starts were used by the
offline G_phi dataset.  The latter is descriptive: the historical WIDE suite
is authoritative even when overlap is detected.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
PILOT = ROOT / "diagnostics/gphi_closed_loop_pilot_v1"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
WIDE_SUITE = SYSROOT / "baseline_309_314/planning/wide_initial_states_200.npz"
HISTORICAL_CONFIG = (
    SYSROOT / "baseline_309_314/planning/seed0_paired/config.json"
)
HISTORICAL_RESULT = (
    SYSROOT / "baseline_309_314/planning/stalled_outcomes_v3.json"
)
OUTPUT = HERE / "frozen_benchmark_manifest.json"
DATASET = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
FLOWBC_DATASET = SYSROOT / "datasets/give_way_si_short_v1"

EXPECTED_WIDE_SHA256 = (
    "30a575df16d56b65cb92b97a95fbcad8b454f49621de45a4359e6bbf947090cb"
)
EXPECTED_HISTORICAL_CONFIG_SHA256 = (
    "9f1894478a628c5be059ecb53a3b7b04449fe59d92a473eea052300d6b8dec19"
)
EXPECTED_FLOW_CHECKPOINT_SHA256 = (
    "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"
)
FLOW_ROOT_SEED = 42
EPISODE_COUNT = 200
FLOW_KEY_SEMANTICS = (
    "episode_key=fold_in(PRNGKey(42), rollout_id); "
    "step_key=fold_in(episode_key, physical_step)"
)

sys.path.insert(0, str(PILOT))
from pilot_common import (  # noqa: E402
    FLOW_CHECKPOINT,
    assert_frozen_sources,
    canonical_json_hash,
    load_environment_config,
    sha256,
    write_json,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _source_initial(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as values:
            if "initial_positions" not in values.files:
                return None
            initial = np.asarray(values["initial_positions"])
    except (OSError, ValueError):
        return None
    return initial if initial.shape == (2, 2) else None


def audit_gphi_overlap(initials: np.ndarray) -> dict[str, Any]:
    """Audit exact WIDE-start reuse by the final startup-complete dataset."""
    manifest_paths = (
        DATASET / "state_manifest.jsonl",
        DATASET / "startup_state_manifest.jsonl",
    )
    # state_manifest.jsonl already contains the startup states.  The dedicated
    # startup manifest is retained as an audited asset, but state_id is the
    # deduplication key so startup rows are never counted twice.
    unique_states: dict[str, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        for row in _read_jsonl(manifest_path):
            state_id = str(row.get("state_id"))
            candidate = {
                "manifest": manifest_path.name,
                "split": str(row.get("split")),
                "state_id": state_id,
                "source_path": str(Path(row["source_path"]).resolve()),
                "source_rng_id": row.get(
                    "source_rng_id", row.get("source_rollout_id")
                ),
                "source_seed": row.get("source_seed"),
            }
            previous = unique_states.get(state_id)
            if previous is not None:
                comparable = {
                    key: value for key, value in candidate.items()
                    if key != "manifest"
                }
                old_comparable = {
                    key: value for key, value in previous.items()
                    if key != "manifest"
                }
                if comparable != old_comparable:
                    raise RuntimeError(("inconsistent duplicate state_id", state_id))
                continue
            unique_states[state_id] = candidate

    rows_by_source: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_states.values():
        source_path = Path(row["source_path"])
        rows_by_source[source_path].append(
            {key: value for key, value in row.items() if key != "source_path"}
        )

    matches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_path, source_rows in rows_by_source.items():
        if not source_path.is_file():
            continue
        source_initial = _source_initial(source_path)
        if source_initial is None:
            continue
        indices = np.flatnonzero(
            np.all(initials == source_initial[None, :, :], axis=(1, 2))
        )
        for index in indices:
            for row in source_rows:
                matches[int(index)].append(
                    {"source_path": str(source_path), **row}
                )

    episode_splits: dict[str, list[int]] = defaultdict(list)
    exact_flow_matches: list[int] = []
    details = []
    for episode_index in sorted(matches):
        rows = matches[episode_index]
        splits = sorted({row["split"] for row in rows})
        for split in splits:
            episode_splits[split].append(episode_index)
        if any(
            row["source_seed"] == FLOW_ROOT_SEED
            and row["source_rng_id"] == episode_index
            for row in rows
        ):
            exact_flow_matches.append(episode_index)
        details.append(
            {
                "episode_index": episode_index,
                "splits": splits,
                "exact_flow_root_seed_and_rollout_id_match": (
                    episode_index in exact_flow_matches
                ),
                "source_records": rows,
            }
        )

    split_source_state_counts = Counter(
        row["split"] for rows in matches.values() for row in rows
    )
    matched_state_ids = {
        row["state_id"]
        for rows in matches.values()
        for row in rows
    }
    with np.load(DATASET / "samples.npz", allow_pickle=False) as values:
        sample_state_ids = np.asarray(values["state_id"]).astype(str)
        sample_splits = np.asarray(values["split"]).astype(str)
    supervised_state_ids = set(sample_state_ids)
    split_supervised_state_counts = Counter(
        row["split"]
        for rows in matches.values()
        for row in rows
        if row["state_id"] in supervised_state_ids
    )
    matched_sample_counts = Counter(
        split
        for state_id, split in zip(sample_state_ids, sample_splits)
        if state_id in matched_state_ids
    )
    train_validation = sorted(
        set(episode_splits.get("train", []))
        | set(episode_splits.get("validation", []))
    )
    return {
        "status": "OVERLAP_DETECTED" if matches else "NO_OVERLAP_DETECTED",
        "interpretation": (
            "The authoritative historical WIDE benchmark is retained, but "
            "efficacy is not an unseen-IC estimate because exact source "
            "trajectories contributed states to the offline G_phi dataset."
            if matches
            else "No exact source-trajectory initial-condition overlap found."
        ),
        "audit_scope": [str(path.resolve()) for path in manifest_paths],
        "audit_scope_sha256": {
            str(path.resolve()): sha256(path) for path in manifest_paths
        },
        "unique_final_states_checked": len(unique_states),
        "source_paths_checked": len(rows_by_source),
        "matched_benchmark_episode_count": len(matches),
        "matched_train_or_validation_episode_count": len(train_validation),
        "matched_episode_count_by_split": {
            split: len(set(indices))
            for split, indices in sorted(episode_splits.items())
        },
        "matched_unique_state_count_by_split": dict(
            sorted(split_supervised_state_counts.items())
        ),
        "matched_source_state_count_including_empty_by_split": dict(
            sorted(split_source_state_counts.items())
        ),
        "matched_supervised_sample_count_by_split": dict(
            sorted(matched_sample_counts.items())
        ),
        "exact_flow_root_seed_and_rollout_id_match_count": len(
            exact_flow_matches
        ),
        "matched_episode_indices_by_split": {
            split: sorted(set(indices))
            for split, indices in sorted(episode_splits.items())
        },
        "exact_flow_match_episode_indices": exact_flow_matches,
        "details": details,
    }


def audit_flowbc_initial_overlap(initials: np.ndarray) -> dict[str, Any]:
    """Compare WIDE starts to the 250 unique FlowBC source starts."""
    raw = FLOWBC_DATASET / "raw"
    source_initials = []
    source_files = []
    for pair in range(250):
        path = raw / f"episode_{2 * pair:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        initial = _source_initial(path)
        if initial is None:
            raise RuntimeError(("FlowBC source has invalid initial", path))
        source_initials.append(initial)
        source_files.append(path)
    sources = np.stack(source_initials)
    distances = np.linalg.norm(
        initials[:, None, :, :].astype(np.float64)
        - sources[None, :, :, :].astype(np.float64),
        axis=(2, 3),
    )
    exact = np.argwhere(distances == 0.0)
    return {
        "status": "PASS" if len(exact) == 0 else "OVERLAP_DETECTED",
        "dataset": str(FLOWBC_DATASET.resolve()),
        "environment_json_sha256": sha256(FLOWBC_DATASET / "environment.json"),
        "unique_source_initial_count": len(
            {initial.tobytes() for initial in sources}
        ),
        "exact_match_count": int(len(exact)),
        "exact_matches": [
            {"wide_episode_index": int(i), "flowbc_pair": int(j)}
            for i, j in exact
        ],
        "minimum_l2_distance": float(distances.min()),
        "source_file_count": len(source_files),
    }


def build_payload() -> dict[str, Any]:
    frozen_sources = assert_frozen_sources()
    required = (WIDE_SUITE, HISTORICAL_CONFIG, HISTORICAL_RESULT)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(WIDE_SUITE) != EXPECTED_WIDE_SHA256:
        raise RuntimeError("authoritative WIDE suite hash mismatch")
    if sha256(HISTORICAL_CONFIG) != EXPECTED_HISTORICAL_CONFIG_SHA256:
        raise RuntimeError("historical WIDE evaluator config hash mismatch")
    if sha256(FLOW_CHECKPOINT) != EXPECTED_FLOW_CHECKPOINT_SHA256:
        raise RuntimeError("frozen FlowBC checkpoint hash mismatch")

    with np.load(WIDE_SUITE, allow_pickle=False) as values:
        if set(values.files) != {
            "val_initial_positions", "test_initial_positions", "metadata_json"
        }:
            raise RuntimeError(("unexpected WIDE suite fields", values.files))
        initials = np.asarray(values["test_initial_positions"])
        metadata = json.loads(str(values["metadata_json"].item()))
    if initials.shape != (EPISODE_COUNT, 2, 2):
        raise RuntimeError(("invalid WIDE test shape", initials.shape))
    if initials.dtype != np.float32 or not np.isfinite(initials).all():
        raise RuntimeError(("invalid WIDE test dtype/values", initials.dtype))
    if len({row.tobytes() for row in initials}) != EPISODE_COUNT:
        raise RuntimeError("WIDE benchmark contains duplicate initial states")

    historical = json.loads(HISTORICAL_CONFIG.read_text())
    expected_environment = load_environment_config(DATASET)
    checks = {
        "suite_hash": historical["initial_suite"]["sha256"]
        == EXPECTED_WIDE_SHA256,
        "checkpoint_hash": historical["checkpoint_sha256"]
        == EXPECTED_FLOW_CHECKPOINT_SHA256,
        "flow_root_seed": historical["seed"] == FLOW_ROOT_SEED,
        "rollout_count": historical["n_rollouts"] == EPISODE_COUNT,
        "rng_semantics": historical["rng_protocol"]
        == "fold_in(fold_in(PRNGKey(seed), rollout_id), step)",
        "outcome_protocol": historical["outcome_protocol"]
        == "exclusive_first_terminal_event_v2",
        "environment": historical["environment"] == expected_environment,
        "initials_exact": np.array_equal(
            np.asarray(historical["initial_positions"], dtype=np.float32),
            initials,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(("historical WIDE contract mismatch", checks))

    x_abs = np.abs(initials[:, :, 0])
    y = initials[:, :, 1]
    overlap = audit_gphi_overlap(initials)
    partition_by_episode = {}
    for split, indices in overlap["matched_episode_indices_by_split"].items():
        for index in indices:
            if index in partition_by_episode:
                raise RuntimeError(("episode appears in multiple G_phi splits", index))
            partition_by_episode[index] = split
    payload: dict[str, Any] = {
        "schema": "gphi_frozen_historical_wide_benchmark_v1",
        "purpose": "production",
        "authoritative_suite": {
            "path": str(WIDE_SUITE.resolve()),
            "sha256": EXPECTED_WIDE_SHA256,
            "array_key": "test_initial_positions",
            "array_shape": list(initials.shape),
            "array_dtype": str(initials.dtype),
            "metadata": metadata,
            "observed_ranges": {
                "absolute_x": [float(x_abs.min()), float(x_abs.max())],
                "y": [float(y.min()), float(y.max())],
            },
        },
        "episode_count": EPISODE_COUNT,
        "episodes": [
            {
                "episode_index": index,
                "rollout_id": index,
                "initial_positions": initial.tolist(),
                "gphi_overlap_partition": partition_by_episode.get(
                    index, "unseen"
                ),
            }
            for index, initial in enumerate(initials)
        ],
        "flow_randomness": {
            "root_seed": FLOW_ROOT_SEED,
            "semantics": FLOW_KEY_SEMANTICS,
            "historical_spelling": historical["rng_protocol"],
            "direct_per_episode_prngkey_forbidden": True,
            "matched_across_controllers": True,
        },
        "environment": historical["environment"],
        "horizon_steps": historical["environment"]["max_steps"],
        "dt_seconds": historical["environment"]["dt"],
        "cbf": historical["cbf"],
        "outcome_protocol": historical["outcome_protocol"],
        "historical_evaluator_contract": {
            "config_path": str(HISTORICAL_CONFIG.resolve()),
            "config_sha256": sha256(HISTORICAL_CONFIG),
            "result_path": str(HISTORICAL_RESULT.resolve()),
            "result_sha256": sha256(HISTORICAL_RESULT),
            "checks": checks,
        },
        "flow_checkpoint": {
            "path": str(FLOW_CHECKPOINT.resolve()),
            "sha256": EXPECTED_FLOW_CHECKPOINT_SHA256,
        },
        "gphi_training_overlap_audit": overlap,
        "flowbc_initial_overlap_audit": audit_flowbc_initial_overlap(initials),
        "frozen_source_audit": frozen_sources,
    }
    payload["content_sha256"] = canonical_json_hash(payload)
    return payload


def main() -> None:
    payload = build_payload()
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text())
        existing_content = {
            key: value for key, value in existing.items()
            if key != "content_sha256"
        }
        if canonical_json_hash(existing_content) != payload["content_sha256"]:
            raise RuntimeError(
                f"refusing to overwrite changed frozen benchmark: {OUTPUT}"
            )
    else:
        write_json(OUTPUT, payload)
    overlap = payload["gphi_training_overlap_audit"]
    print(json.dumps({
        "status": "PASS",
        "output": str(OUTPUT),
        "content_sha256": payload["content_sha256"],
        "episodes": payload["episode_count"],
        "overlap_status": overlap["status"],
        "matched_benchmark_episodes": overlap[
            "matched_benchmark_episode_count"
        ],
        "matched_train_or_validation_episodes": overlap[
            "matched_train_or_validation_episode_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
