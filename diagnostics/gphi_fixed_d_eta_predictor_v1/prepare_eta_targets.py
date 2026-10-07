"""Recover canonical frozen eta targets for the controlled fixed-D audit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
BASE = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
COVERAGE = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1"
EXPECTED_SHA = "79d7da0492d9b414c03ce53f9ee826c54ac7dce3509b2cd3d086fc1cf852deb9"
EXPECTED_SAMPLES = 27136
EXPECTED_STATES = 424
EXPECTED_VARIANTS = 64

# Union envelope of the authoritative V1/V3 refined candidate coordinates and
# the frozen strict-deadlock capacity domain.  Zero is deliberately included.
ETA_LOW = np.asarray([0.0, -0.53125, -0.125], dtype=np.float64)
ETA_HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def eta_tuple(value: object) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(values) != 3 or not np.isfinite(values).all():
        raise RuntimeError(("invalid eta", value))
    return values


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    samples_path = DATASET / "samples.npz"
    if sha256(samples_path) != EXPECTED_SHA:
        raise RuntimeError("authoritative coverage dataset hash mismatch")
    with np.load(samples_path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    if arrays["features"].shape != (EXPECTED_SAMPLES, 214):
        raise RuntimeError(arrays["features"].shape)
    if len(set(arrays["state_id"].tolist())) != EXPECTED_STATES:
        raise RuntimeError("unexpected state count")

    metadata = [
        json.loads(line)
        for line in (DATASET / "sample_metadata.jsonl").read_text().splitlines()
        if line
    ]
    if len(metadata) != EXPECTED_SAMPLES:
        raise RuntimeError(("metadata rows", len(metadata)))
    if any(str(row["sample_id"]) != str(arrays["sample_id"][index])
           for index, row in enumerate(metadata)):
        raise RuntimeError("metadata/sample alignment mismatch")

    eta = np.asarray([eta_tuple(row["eta_best_metadata_only"]) for row in metadata])
    if np.any(eta < ETA_LOW - 1e-12) or np.any(eta > ETA_HIGH + 1e-12):
        raise RuntimeError(("eta outside authoritative union envelope", eta.min(0), eta.max(0)))
    eta_norm = (eta - ETA_LOW) / (ETA_HIGH - ETA_LOW)

    by_state: dict[str, list[int]] = defaultdict(list)
    for index, state_id in enumerate(arrays["state_id"].tolist()):
        by_state[str(state_id)].append(index)
    consistency_rows = []
    ambiguous_states = 0
    multiple_equivalent = 0
    zero_states = 0
    for state_id, indices in by_state.items():
        values = eta[indices]
        unique = np.unique(np.round(values, 12), axis=0)
        spread = values.max(axis=0) - values.min(axis=0)
        if len(unique) != 1:
            ambiguous_states += 1
        representative = metadata[indices[0]]
        near = representative.get("E_near_metadata_only", [])
        if isinstance(near, str):
            near = json.loads(near)
        if len(near) > 1:
            multiple_equivalent += 1
        if np.allclose(values[0], 0.0, atol=1e-12):
            zero_states += 1
        consistency_rows.append({
            "state_id": state_id,
            "split": str(arrays["split"][indices[0]]),
            "category": str(arrays["category"][indices[0]]),
            "flow_variants": len(indices),
            "unique_frozen_eta_labels": len(unique),
            "eta1": values[0, 0],
            "eta2": values[0, 1],
            "eta3": values[0, 2],
            "eta1_flow_spread": spread[0],
            "eta2_flow_spread": spread[1],
            "eta3_flow_spread": spread[2],
            "equivalent_near_candidate_count": len(near),
            "multiple_equivalent_candidates": len(near) > 1,
            "is_zero_eta": bool(np.allclose(values[0], 0.0, atol=1e-12)),
        })
    if ambiguous_states:
        raise RuntimeError(("ETA_TARGET_NOT_WELL_DEFINED", ambiguous_states))

    np.savez_compressed(
        HERE / "eta_targets.npz",
        eta_physical=eta.astype(np.float64),
        eta_normalized=eta_norm.astype(np.float64),
        eta_low=ETA_LOW,
        eta_high=ETA_HIGH,
        sample_id=arrays["sample_id"],
        state_id=arrays["state_id"],
        split=arrays["split"],
    )
    write_csv(HERE / "eta_target_consistency.csv", consistency_rows)

    state_split = Counter((str(row["split"]), str(row["state_id"])) for row in consistency_rows)
    sample_split = Counter(str(value) for value in arrays["split"].tolist())
    split_payload = {
        "source": str(DATASET / "split_manifest.json"),
        "source_sha256": sha256(DATASET / "split_manifest.json"),
        "grouping": "source augmented state; all 64 Flow variants remain together",
        "samples": dict(sorted(sample_split.items())),
        "states": {
            split: sum(1 for source_split, _ in state_split if source_split == split)
            for split in sorted(sample_split)
        },
        "historical_11_augmentation": "training-only",
        "fresh_6_used": False,
    }
    write_json(HERE / "split_manifest.json", split_payload)
    write_json(HERE / "source_dataset_manifest.json", {
        "authoritative_dataset": str(DATASET),
        "samples": str(samples_path),
        "samples_sha256": sha256(samples_path),
        "sample_metadata": str(DATASET / "sample_metadata.jsonl"),
        "sample_metadata_sha256": sha256(DATASET / "sample_metadata.jsonl"),
        "source_coverage_manifest": str(COVERAGE / "augmented_dataset_manifest.json"),
        "startup_complete_lineage": str(BASE),
        "unique_states": EXPECTED_STATES,
        "samples_count": EXPECTED_SAMPLES,
        "flow_variants_per_state": EXPECTED_VARIANTS,
        "feature_dim": 214,
        "lineage": "startup-complete base + historical-11 strict-deadlock onset augmentation",
        "k1_dagger_data_used": False,
        "fresh6_data_used": False,
    })
    audit = {
        "classification": "CANONICAL_ETA_TARGETS_RECOVERED",
        "canonical_eta_source": "eta_best_metadata_only from authoritative oracle-generated sample metadata",
        "eta_inferred_from_g": False,
        "oracle_rerun_required": False,
        "unique_states": EXPECTED_STATES,
        "samples": EXPECTED_SAMPLES,
        "states_with_one_canonical_eta": EXPECTED_STATES - ambiguous_states,
        "states_with_flow_variant_eta_conflict": ambiguous_states,
        "states_with_multiple_equivalent_near_candidates": multiple_equivalent,
        "coordinate_wise_max_flow_spread": np.max([
            [row["eta1_flow_spread"], row["eta2_flow_spread"], row["eta3_flow_spread"]]
            for row in consistency_rows
        ], axis=0).tolist(),
        "zero_eta_states": zero_states,
        "active_eta_states": EXPECTED_STATES - zero_states,
        "unique_eta_vectors": int(len(np.unique(np.round(eta, 12), axis=0))),
        "observed_eta_min": eta.min(axis=0).tolist(),
        "observed_eta_max": eta.max(axis=0).tolist(),
        "normalization_low": ETA_LOW.tolist(),
        "normalization_high": ETA_HIGH.tolist(),
        "normalization_rationale": (
            "union envelope of frozen V1/V3 refined candidates and frozen strict-deadlock "
            "eta domain, enlarged only to include authoritative eta=0 exactly"
        ),
    }
    write_json(HERE / "eta_target_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
