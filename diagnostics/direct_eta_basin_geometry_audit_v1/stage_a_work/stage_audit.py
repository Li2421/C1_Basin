#!/usr/bin/env python3
"""Stage A: exhaustive canonical-eta geometry over 424 unique states.

This is a read-only audit of frozen inputs.  It writes only beside this script.
No continuation simulation or model training is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
PREDICTOR = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
STARTUP = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
STRICT = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
OUT = Path(__file__).resolve().parent

EXPECTED_STATES = 424
EXPECTED_SAMPLES = 27_136
EXPECTED_VARIANTS = 64
ETA_LOW = np.asarray([0.0, -0.53125, -0.125], dtype=np.float64)
ETA_HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)
DISTINCT_ATOL = 1e-12
NEAR_BOUNDARY_NORMALIZED_TOL = 0.01


ASSETS = {
    "predictor_checkpoint": PREDICTOR / "best_fixed_d_eta_checkpoint.npz",
    "predictor_eta_targets": PREDICTOR / "eta_targets.npz",
    "predictor_eta_target_audit": PREDICTOR / "eta_target_audit.json",
    "predictor_source_dataset_manifest": PREDICTOR / "source_dataset_manifest.json",
    "predictor_prepare_eta_targets": PREDICTOR / "prepare_eta_targets.py",
    "dataset_samples": DATASET / "samples.npz",
    "dataset_sample_metadata": DATASET / "sample_metadata.jsonl",
    "dataset_state_manifest": DATASET / "state_manifest.jsonl",
    "dataset_manifest": DATASET / "manifest.json",
    "dataset_split_manifest": DATASET / "split_manifest.json",
    "dataset_feature_schema": DATASET / "feature_schema.json",
    "v3_oracle_results": V3 / "oracle_search_results.jsonl",
    "v3_protocol": V3 / "protocol.json",
    "startup_oracle_results": STARTUP / "startup_oracle_search_results.jsonl",
    "startup_protocol": STARTUP / "protocol.json",
    "strict_robust_basin_validation": STRICT / "robust_basin_validation.csv",
    "strict_best_eta_by_state": STRICT / "best_eta_by_state.csv",
    "strict_eta_search_config": STRICT / "eta_search_config.json",
    "strict_manifest": STRICT / "manifest.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def vector_key(value: Any) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise RuntimeError(("invalid eta", value))
    return tuple(float(item) for item in np.round(array, 12))


def bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "mean": float(np.mean(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "std_population": float(np.std(values)),
        "distinct_at_1e-12": int(len(set(float(x) for x in np.round(values, 12)))),
    }


def schema_for(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(path.read_text())
        return {
            "format": "json",
            "top_level_type": type(value).__name__,
            "top_level_keys": sorted(value) if isinstance(value, dict) else None,
            "declared_schema": value.get("schema") if isinstance(value, dict) else None,
        }
    if suffix == ".jsonl":
        first = json.loads(next(line for line in path.read_text().splitlines() if line))
        return {
            "format": "jsonl",
            "row_keys": sorted(first),
            "rows": sum(1 for line in path.open() if line.strip()),
        }
    if suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = sum(1 for _ in reader)
        return {"format": "csv", "columns": header, "rows": rows}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            arrays = {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                for key in data.files
            }
        return {"format": "npz", "arrays": arrays}
    if suffix == ".py":
        return {"format": "python_source"}
    return {"format": suffix.lstrip(".")}


def main() -> None:
    for name, path in ASSETS.items():
        if not path.is_file():
            raise FileNotFoundError((name, path))

    dataset_manifest = json.loads((DATASET / "manifest.json").read_text())
    if dataset_manifest["final_unique_states"] != EXPECTED_STATES:
        raise RuntimeError(dataset_manifest)
    if dataset_manifest["final_samples"] != EXPECTED_SAMPLES:
        raise RuntimeError(dataset_manifest)

    state_manifest = load_jsonl(DATASET / "state_manifest.jsonl")
    sample_metadata = load_jsonl(DATASET / "sample_metadata.jsonl")
    if len(state_manifest) != EXPECTED_STATES or len(sample_metadata) != EXPECTED_SAMPLES:
        raise RuntimeError((len(state_manifest), len(sample_metadata)))
    if len({row["state_id"] for row in state_manifest}) != EXPECTED_STATES:
        raise RuntimeError("state_manifest state IDs are not unique")
    state_meta = {str(row["state_id"]): row for row in state_manifest}

    with np.load(DATASET / "samples.npz", allow_pickle=False) as source:
        npz_state_id = np.asarray(source["state_id"]).copy()
        npz_split = np.asarray(source["split"]).copy()
        npz_category = np.asarray(source["category"]).copy()
        npz_sample_id = np.asarray(source["sample_id"]).copy()
        feature_shape = list(source["features"].shape)
    if feature_shape != [EXPECTED_SAMPLES, 214]:
        raise RuntimeError(feature_shape)
    if any(str(row["sample_id"]) != str(npz_sample_id[index]) for index, row in enumerate(sample_metadata)):
        raise RuntimeError("sample metadata and samples.npz are not aligned")

    metadata_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_indices_by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(sample_metadata):
        state_id = str(row["state_id"])
        if state_id != str(npz_state_id[index]):
            raise RuntimeError(("state alignment", index, state_id, npz_state_id[index]))
        metadata_by_state[state_id].append(row)
        sample_indices_by_state[state_id].append(index)
    if set(metadata_by_state) != set(state_meta):
        raise RuntimeError("state identity mismatch between manifests")

    with np.load(PREDICTOR / "eta_targets.npz", allow_pickle=False) as eta_file:
        eta_physical = np.asarray(eta_file["eta_physical"], dtype=np.float64)
        eta_low = np.asarray(eta_file["eta_low"], dtype=np.float64)
        eta_high = np.asarray(eta_file["eta_high"], dtype=np.float64)
        eta_file_sample_id = np.asarray(eta_file["sample_id"])
    if not np.array_equal(eta_file_sample_id, npz_sample_id):
        raise RuntimeError("eta_targets.npz sample alignment mismatch")
    if not np.array_equal(eta_low, ETA_LOW) or not np.array_equal(eta_high, ETA_HIGH):
        raise RuntimeError((eta_low, eta_high))

    v3_oracles = {row["state_id"]: row for row in load_jsonl(V3 / "oracle_search_results.jsonl")}
    startup_oracles = {
        row["state_id"]: row for row in load_jsonl(STARTUP / "startup_oracle_search_results.jsonl")
    }
    strict_validation: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (STRICT / "robust_basin_validation.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            strict_validation[row["state_id"]].append(row)

    state_records: list[dict[str, Any]] = []
    canonical: dict[str, tuple[float, float, float]] = {}
    for state in sorted(state_manifest, key=lambda row: int(row["state_index"])):
        state_id = str(state["state_id"])
        variants = metadata_by_state[state_id]
        indices = sample_indices_by_state[state_id]
        if len(variants) != EXPECTED_VARIANTS or len(indices) != EXPECTED_VARIANTS:
            raise RuntimeError((state_id, len(variants), len(indices)))
        target_values = {vector_key(row["eta_best_metadata_only"]) for row in variants}
        npz_values = {vector_key(eta_physical[index]) for index in indices}
        if len(target_values) != 1 or target_values != npz_values:
            raise RuntimeError((state_id, target_values, npz_values))
        eta = next(iter(target_values))
        canonical[state_id] = eta
        array = np.asarray(eta, dtype=np.float64)
        normalized = (array - ETA_LOW) / (ETA_HIGH - ETA_LOW)
        zero = bool(np.array_equal(array, np.zeros(3, dtype=np.float64)))

        category = str(state["category"])
        oracle_kind: str
        oracle_record: dict[str, Any] | None
        robust_values: list[Any]
        evaluated_candidate_count: int
        near_values: list[Any] | None
        if category == "STRICT_DEADLOCK":
            oracle_kind = "strict_deadlock_robust_basin_validation"
            oracle_record = None
            candidate_rows = strict_validation[state_id]
            if not candidate_rows:
                raise RuntimeError(("missing strict validation", state_id))
            robust_values = [json.loads(row["eta"]) for row in candidate_rows if row["B_63_member"] == "True"]
            evaluated_candidate_count = len(candidate_rows)
            near_values = None
        elif state.get("dataset_origin") == "STARTUP":
            oracle_kind = "startup_oracle_search_results"
            oracle_record = startup_oracles[state_id]
            robust_values = oracle_record["B_63"]
            evaluated_candidate_count = len(oracle_record["cells"])
            near_values = oracle_record["E_near"]
        else:
            oracle_kind = "v3_oracle_search_results"
            oracle_record = v3_oracles[state_id]
            robust_values = oracle_record["B_63"]
            evaluated_candidate_count = len(oracle_record["cells"])
            near_values = oracle_record["E_near"]
        robust_keys = sorted({vector_key(value) for value in robust_values})
        if eta not in robust_keys:
            raise RuntimeError(("canonical target absent from stored robust candidates", state_id, eta, robust_keys))
        if oracle_record is not None and vector_key(oracle_record["eta_best"]) != eta:
            raise RuntimeError(("oracle target mismatch", state_id))
        near_count = None if near_values is None else len({vector_key(value) for value in near_values})

        exact_low = np.isclose(array, ETA_LOW, rtol=0.0, atol=DISTINCT_ATOL)
        exact_high = np.isclose(array, ETA_HIGH, rtol=0.0, atol=DISTINCT_ATOL)
        boundary_distance = np.minimum(normalized, 1.0 - normalized)
        near_or_on = boundary_distance <= NEAR_BOUNDARY_NORMALIZED_TOL + 1e-15
        exact_any = bool(np.any(exact_low | exact_high))
        near_any = bool(np.any(near_or_on))

        first = variants[0]
        if str(first["split"]) != str(state["split"]):
            raise RuntimeError(("split mismatch", state_id))
        if str(first["category"]) != category:
            raise RuntimeError(("category mismatch", state_id))
        if any(str(npz_split[index]) != str(state["split"]) for index in indices):
            raise RuntimeError(("npz split mismatch", state_id))
        if any(str(npz_category[index]) != category for index in indices):
            raise RuntimeError(("npz category mismatch", state_id))

        state_records.append({
            "state_index": int(state["state_index"]),
            "state_id": state_id,
            "split": state["split"],
            "category": category,
            "dataset_origin": state.get("dataset_origin", ""),
            "source_type": state.get("source_type", ""),
            "source_trajectory": state.get("source_trajectory", ""),
            "leakage_group": state.get("leakage_group", ""),
            "flow_variant_count": len(variants),
            "eta1": eta[0],
            "eta2": eta[1],
            "eta3": eta[2],
            "eta_norm": float(np.linalg.norm(array)),
            "eta_normalized_norm": float(np.linalg.norm(normalized)),
            "is_zero_eta": bool_text(zero),
            "oracle_metadata_source": oracle_kind,
            "stored_evaluated_candidate_count": evaluated_candidate_count,
            "stored_robust_candidate_count": len(robust_keys),
            "has_multiple_stored_robust_candidates": bool_text(len(robust_keys) > 1),
            "near_equivalent_metadata_available": bool_text(near_values is not None),
            "near_equivalent_candidate_count": "" if near_count is None else near_count,
            "multiple_near_equivalent_candidates": bool_text(None if near_count is None else near_count > 1),
            "eta1_exact_low_boundary": bool_text(bool(exact_low[0])),
            "eta1_exact_high_boundary": bool_text(bool(exact_high[0])),
            "eta2_exact_low_boundary": bool_text(bool(exact_low[1])),
            "eta2_exact_high_boundary": bool_text(bool(exact_high[1])),
            "eta3_exact_low_boundary": bool_text(bool(exact_low[2])),
            "eta3_exact_high_boundary": bool_text(bool(exact_high[2])),
            "any_exact_union_envelope_boundary": bool_text(exact_any),
            "any_near_or_on_union_envelope_boundary_1pct": bool_text(near_any),
            "minimum_normalized_boundary_distance": float(np.min(boundary_distance)),
        })

    if len(state_records) != EXPECTED_STATES:
        raise RuntimeError(len(state_records))

    frequencies = Counter(canonical.values())
    frequency_rows: list[dict[str, Any]] = []
    sorted_frequencies = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    for rank, (eta, count) in enumerate(sorted_frequencies, start=1):
        members = [row for row in state_records if vector_key([row["eta1"], row["eta2"], row["eta3"]]) == eta]
        category_counts = Counter(str(row["category"]) for row in members)
        split_counts = Counter(str(row["split"]) for row in members)
        origin_counts = Counter(str(row["dataset_origin"] or "<MISSING>") for row in members)
        source_type_counts = Counter(str(row["source_type"] or "<MISSING>") for row in members)
        frequency_rows.append({
            "frequency_rank": rank,
            "eta1": eta[0],
            "eta2": eta[1],
            "eta3": eta[2],
            "eta_norm": math.sqrt(sum(value * value for value in eta)),
            "state_count": count,
            "state_fraction": count / EXPECTED_STATES,
            "is_zero_eta": bool_text(eta == (0.0, 0.0, 0.0)),
            "is_exactly_repeated": bool_text(count > 1),
            "train_count": split_counts["train"],
            "validation_count": split_counts["validation"],
            "test_count": split_counts["test"],
            "normal_count": category_counts["NORMAL"],
            "startup_count": category_counts["STARTUP"],
            "recovery_count": category_counts["RECOVERY"],
            "pre_deadlock_count": category_counts["PRE_DEADLOCK"],
            "strict_deadlock_count": category_counts["STRICT_DEADLOCK"],
            "dataset_origin_counts_json": json.dumps(dict(sorted(origin_counts.items())), sort_keys=True),
            "source_type_counts_json": json.dumps(dict(sorted(source_type_counts.items())), sort_keys=True),
        })

    arrays = np.asarray([[row["eta1"], row["eta2"], row["eta3"]] for row in state_records], dtype=np.float64)
    norms = np.linalg.norm(arrays, axis=1)
    zero_mask = np.all(arrays == 0.0, axis=1)
    active_mask = ~zero_mask
    normalized_arrays = (arrays - ETA_LOW) / (ETA_HIGH - ETA_LOW)
    normalized_norms = np.linalg.norm(normalized_arrays, axis=1)

    exact_boundary = np.asarray([row["any_exact_union_envelope_boundary"] == "true" for row in state_records])
    near_boundary = np.asarray([
        row["any_near_or_on_union_envelope_boundary_1pct"] == "true" for row in state_records
    ])
    multiple_robust = np.asarray([
        row["has_multiple_stored_robust_candidates"] == "true" for row in state_records
    ])
    near_available = np.asarray([
        row["near_equivalent_metadata_available"] == "true" for row in state_records
    ])
    multiple_near = np.asarray([
        row["multiple_near_equivalent_candidates"] == "true" for row in state_records
    ])

    coordinate_summary: dict[str, Any] = {}
    for index, name in enumerate(("eta1", "eta2", "eta3")):
        coordinate_summary[name] = {
            "all_states": distribution(arrays[:, index]),
            "active_states": distribution(arrays[active_mask, index]),
        }
    norm_summary = {
        "physical_eta_l2_all_states": distribution(norms),
        "physical_eta_l2_active_states": distribution(norms[active_mask]),
        "coordinate_normalized_eta_l2_all_states": distribution(normalized_norms),
        "coordinate_normalized_eta_l2_active_states": distribution(normalized_norms[active_mask]),
    }

    category_slices: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in state_records}):
        indices = np.asarray([str(row["category"]) == category for row in state_records])
        category_slices[category] = {
            "states": int(np.sum(indices)),
            "zero": int(np.sum(zero_mask & indices)),
            "active": int(np.sum(active_mask & indices)),
            "distinct_canonical_eta_at_1e-12": int(len({
                canonical[str(row["state_id"])] for row, selected in zip(state_records, indices) if selected
            })),
            "multiple_stored_robust_candidates": int(np.sum(multiple_robust & indices)),
            "multiple_near_equivalent_candidates_confirmed": int(np.sum(multiple_near & indices)),
        }

    summary = {
        "schema": "direct_eta_target_geometry_stage_a_v1",
        "state_counting_rule": "each of the 424 unique augmented state_id values counts once; Flow variants never reweight statistics",
        "unique_states": EXPECTED_STATES,
        "feature_samples_cross_checked": EXPECTED_SAMPLES,
        "flow_variants_per_state": EXPECTED_VARIANTS,
        "zero_eta_definition": "exact canonical eta vector == (0,0,0); no norm threshold",
        "zero_eta_states": int(np.sum(zero_mask)),
        "active_eta_states": int(np.sum(active_mask)),
        "distinct_canonical_eta_vectors_at_1e-12": len(frequencies),
        "exactly_repeated_eta_vectors": sum(count > 1 for count in frequencies.values()),
        "singleton_eta_vectors": sum(count == 1 for count in frequencies.values()),
        "states_in_exactly_repeated_eta_vectors": sum(count for count in frequencies.values() if count > 1),
        "eta_domain_union_envelope": {
            "low": ETA_LOW.tolist(),
            "high": ETA_HIGH.tolist(),
            "source": "frozen eta_targets normalization envelope; union of V1/V3 refined candidates and strict-deadlock domain, enlarged only to include eta=0",
        },
        "boundary_rule": {
            "exact_atol": DISTINCT_ATOL,
            "near_or_on_coordinate_normalized_distance": NEAR_BOUNDARY_NORMALIZED_TOL,
        },
        "states_on_any_exact_union_envelope_boundary": int(np.sum(exact_boundary)),
        "fraction_on_any_exact_union_envelope_boundary": float(np.mean(exact_boundary)),
        "active_states_on_any_exact_union_envelope_boundary": int(np.sum(exact_boundary & active_mask)),
        "active_fraction_on_any_exact_union_envelope_boundary": float(np.mean(exact_boundary[active_mask])),
        "states_near_or_on_any_union_envelope_boundary_1pct": int(np.sum(near_boundary)),
        "fraction_near_or_on_any_union_envelope_boundary_1pct": float(np.mean(near_boundary)),
        "active_states_near_or_on_any_union_envelope_boundary_1pct": int(np.sum(near_boundary & active_mask)),
        "active_fraction_near_or_on_any_union_envelope_boundary_1pct": float(np.mean(near_boundary[active_mask])),
        "coordinate_exact_boundary_counts": {
            f"{name}_{side}": int(sum(row[f"{name}_exact_{side}_boundary"] == "true" for row in state_records))
            for name in ("eta1", "eta2", "eta3") for side in ("low", "high")
        },
        "states_with_multiple_stored_robust_candidates": int(np.sum(multiple_robust)),
        "fraction_with_multiple_stored_robust_candidates": float(np.mean(multiple_robust)),
        "near_equivalent_candidate_metadata_available_states": int(np.sum(near_available)),
        "near_equivalent_candidate_metadata_unavailable_states": int(np.sum(~near_available)),
        "states_with_multiple_near_equivalent_candidates_confirmed": int(np.sum(multiple_near)),
        "fraction_with_multiple_near_equivalent_candidates_conservative_all_424": float(np.mean(multiple_near)),
        "fraction_with_multiple_near_equivalent_candidates_among_available_413": float(np.mean(multiple_near[near_available])),
        "coordinate_distributions": coordinate_summary,
        "eta_norm_distributions": norm_summary,
        "category_slices": category_slices,
        "qualitative_shape": "large exact zero spike plus a small set of repeatedly selected active lattice/refinement modes; not one smooth continuous cloud",
        "clustering": "not run: exact canonical frequencies are directly observable and no single clustering method is used to assert multimodality",
        "scope_warning": "Repeated canonical values and multiple stored B63 candidates describe oracle targets/candidate support, not connected components of the underlying success set.",
    }

    asset_inventory = {
        "schema": "direct_eta_stage_a_asset_inventory_v1",
        "assets": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "schema": schema_for(path),
            }
            for name, path in ASSETS.items()
        },
        "resolved_lineage": {
            "dataset": str(DATASET),
            "canonical_eta_source": str(DATASET / "sample_metadata.jsonl") + "::eta_best_metadata_only",
            "v3_state_oracle_metadata": str(V3 / "oracle_search_results.jsonl"),
            "startup_state_oracle_metadata": str(STARTUP / "startup_oracle_search_results.jsonl"),
            "strict_deadlock_oracle_metadata": str(STRICT / "robust_basin_validation.csv"),
        },
    }

    dataset_424_manifest = {
        "schema": "direct_eta_dataset_424_manifest_v1",
        "authoritative_dataset": str(DATASET),
        "samples": str(DATASET / "samples.npz"),
        "samples_sha256": asset_inventory["assets"]["dataset_samples"]["sha256"],
        "sample_metadata": str(DATASET / "sample_metadata.jsonl"),
        "sample_metadata_sha256": asset_inventory["assets"]["dataset_sample_metadata"]["sha256"],
        "state_manifest": str(DATASET / "state_manifest.jsonl"),
        "state_manifest_sha256": asset_inventory["assets"]["dataset_state_manifest"]["sha256"],
        "canonical_eta_source_field": "eta_best_metadata_only",
        "unique_states": EXPECTED_STATES,
        "feature_samples": EXPECTED_SAMPLES,
        "flow_variants_per_state": EXPECTED_VARIANTS,
        "feature_dimension": 214,
        "state_counting_rule": "one row per unique state_id; never Flow-variant weighted",
        "states": [{
            "state_index": row["state_index"],
            "state_id": row["state_id"],
            "split": row["split"],
            "category": row["category"],
            "dataset_origin": row["dataset_origin"],
            "source_type": row["source_type"],
            "source_trajectory": row["source_trajectory"],
            "leakage_group": row["leakage_group"],
            "flow_variant_count": row["flow_variant_count"],
            "canonical_eta": [row["eta1"], row["eta2"], row["eta3"]],
        } for row in state_records],
    }
    frozen_asset_hashes = {
        "schema": "direct_eta_stage_a_frozen_asset_hashes_v1",
        "assets": {
            name: {
                "path": value["path"],
                "bytes": value["bytes"],
                "sha256": value["sha256"],
            }
            for name, value in asset_inventory["assets"].items()
        },
    }

    write_csv(OUT / "target_geometry_424.csv", state_records)
    write_csv(OUT / "canonical_eta_frequency.csv", frequency_rows)
    write_json(OUT / "stage_a_summary.json", summary)
    write_json(OUT / "asset_inventory.json", asset_inventory)
    write_json(OUT / "dataset_424_manifest.json", dataset_424_manifest)
    write_json(OUT / "frozen_asset_hashes.json", frozen_asset_hashes)

    def fmt_distribution(value: dict[str, Any]) -> str:
        return " | ".join(
            f"{value[key]:.9g}" if isinstance(value[key], float) else str(value[key])
            for key in ("min", "q05", "q25", "median", "mean", "q75", "q95", "max", "std_population")
        )

    frequency_table = "\n".join(
        f"| {row['frequency_rank']} | ({row['eta1']:.9g}, {row['eta2']:.9g}, {row['eta3']:.9g}) | {row['state_count']} | {100 * row['state_fraction']:.2f}% |"
        for row in frequency_rows
    )
    coordinate_table = "\n".join(
        f"| {name} all | {fmt_distribution(coordinate_summary[name]['all_states'])} |\n"
        f"| {name} active | {fmt_distribution(coordinate_summary[name]['active_states'])} |"
        for name in ("eta1", "eta2", "eta3")
    )
    norm_table = "\n".join(
        f"| {name} | {fmt_distribution(values)} |" for name, values in norm_summary.items()
    )
    category_table = "\n".join(
        f"| {category} | {values['states']} | {values['zero']} | {values['active']} | {values['distinct_canonical_eta_at_1e-12']} | {values['multiple_stored_robust_candidates']} | {values['multiple_near_equivalent_candidates_confirmed']} |"
        for category, values in category_slices.items()
    )

    report = f"""# Stage A — 424-state canonical eta target geometry

## Counting and provenance

Every unique augmented `state_id` counts exactly once. The 27,136 deployment-feature rows were used only to verify the invariant 64 Flow variants per state and one shared canonical eta; they do not reweight any statistic. Canonical eta is read directly from `eta_best_metadata_only` and cross-checked against the frozen `eta_targets.npz`; eta is never inferred from `g`.

The exact checkpoint SHA256 is `{asset_inventory['assets']['predictor_checkpoint']['sha256']}`. The exact samples SHA256 is `{asset_inventory['assets']['dataset_samples']['sha256']}`.

## Main result

- ZERO: **{summary['zero_eta_states']} / 424 ({100 * summary['zero_eta_states'] / 424:.2f}%)**.
- ACTIVE: **{summary['active_eta_states']} / 424 ({100 * summary['active_eta_states'] / 424:.2f}%)**.
- Distinct canonical eta vectors at absolute tolerance 1e-12: **{summary['distinct_canonical_eta_vectors_at_1e-12']}**.
- Exactly repeated vectors: **{summary['exactly_repeated_eta_vectors']}**; singleton vectors: **{summary['singleton_eta_vectors']}**; **{summary['states_in_exactly_repeated_eta_vectors']} / 424** states belong to a repeated vector.
- Geometry description: **large exact zero spike plus a small set of repeatedly selected active lattice/refinement modes**. This is not one smooth continuous cloud. This statement is about canonical target B, not success-set topology A.

## Exact canonical eta frequencies

| Rank | eta | States | Fraction |
|---:|---|---:|---:|
{frequency_table}

## Coordinate distributions

Columns after population are min, q05, q25, median, mean, q75, q95, max, population SD.

| Population | min | q05 | q25 | median | mean | q75 | q95 | max | SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{coordinate_table}

## Eta norm distributions

| Quantity | min | q05 | q25 | median | mean | q75 | q95 | max | SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{norm_table}

## Frozen union-envelope boundary saturation

The frozen normalization envelope is low `{ETA_LOW.tolist()}`, high `{ETA_HIGH.tolist()}`. It is the union envelope documented by the fixed-D target preparation, enlarged only to include eta=0. Exact uses absolute tolerance 1e-12; near/on means coordinate-normalized distance <=1% from any face.

- Any exact face: **{summary['states_on_any_exact_union_envelope_boundary']} / 424 ({100 * summary['fraction_on_any_exact_union_envelope_boundary']:.2f}%)**.
- Among ACTIVE only: **{summary['active_states_on_any_exact_union_envelope_boundary']} / 182 ({100 * summary['active_fraction_on_any_exact_union_envelope_boundary']:.2f}%)**.
- Near/on any face at 1%: **{summary['states_near_or_on_any_union_envelope_boundary_1pct']} / 424 ({100 * summary['fraction_near_or_on_any_union_envelope_boundary_1pct']:.2f}%)**; ACTIVE only **{summary['active_states_near_or_on_any_union_envelope_boundary_1pct']} / 182 ({100 * summary['active_fraction_near_or_on_any_union_envelope_boundary_1pct']:.2f}%)**.
- Exact coordinate-face counts: `{json.dumps(summary['coordinate_exact_boundary_counts'], sort_keys=True)}`.

The overall boundary fraction is dominated by the 242 exact-zero targets lying on the eta1 lower envelope. The ACTIVE-only fraction is the relevant saturation diagnostic.

## Stored robust candidates and near-equivalent selection

- Multiple stored robust B63 eta values: **{summary['states_with_multiple_stored_robust_candidates']} / 424 ({100 * summary['fraction_with_multiple_stored_robust_candidates']:.2f}%)**.
- Multiple statistically near-equivalent `E_near` candidates confirmed: **{summary['states_with_multiple_near_equivalent_candidates_confirmed']} states**.
- `E_near` metadata is present for **{summary['near_equivalent_candidate_metadata_available_states']}** V3/startup states and unavailable for the **{summary['near_equivalent_candidate_metadata_unavailable_states']}** strict-deadlock augmentations. Thus the conservative all-state frequency is **{100 * summary['fraction_with_multiple_near_equivalent_candidates_conservative_all_424']:.2f}%**, or **{100 * summary['fraction_with_multiple_near_equivalent_candidates_among_available_413']:.2f}%** among states with that field.

Accordingly, frozen canonical selection chose one target from a stored multi-`E_near` near-equivalent set in **73 confirmed states**. These are not necessarily exact floating-point J_def ties: `E_near` denotes paired-statistical indistinguishability under the frozen oracle procedure.

Multiple stored robust candidates do not establish disconnected basins. Likewise, repeated canonical values reflect a finite frozen candidate/refinement lattice and frozen minimum-J_def selection; they are not by themselves evidence for a multimodal conditional success set.

## Existing category slices (post-hoc only)

| Existing category | States | ZERO | ACTIVE | Distinct eta | Multiple stored robust | Multiple near-equivalent confirmed |
|---|---:|---:|---:|---:|---:|---:|
{category_table}

No semantic phase was resampled or reweighted. No clustering was used to declare multimodality.

## Files

- `target_geometry_424.csv`: one row per unique state, with source/split/category, canonical eta, norm, candidate metadata, and boundary flags.
- `canonical_eta_frequency.csv`: one row per distinct canonical eta vector.
- `stage_a_summary.json`: machine-readable statistics and frozen rules.
- `asset_inventory.json`: exact authoritative paths, hashes, and schemas.
- `frozen_asset_hashes.json`: compact path/hash table for parent-manifest assembly.
- `dataset_424_manifest.json`: one authoritative manifest entry per unique state.
- `stage_audit.py`: deterministic reproduction script.
"""
    (OUT / "target_geometry_report.md").write_text(report)
    print(json.dumps({
        "zero": summary["zero_eta_states"],
        "active": summary["active_eta_states"],
        "distinct_eta": summary["distinct_canonical_eta_vectors_at_1e-12"],
        "multiple_robust": summary["states_with_multiple_stored_robust_candidates"],
        "multiple_near_equivalent_confirmed": summary["states_with_multiple_near_equivalent_candidates_confirmed"],
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
