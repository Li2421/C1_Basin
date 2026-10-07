"""Freeze assets and outcome-blind uniform subsets before new basin rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYS = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"
ETA = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
DATA = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
FRESH = ROOT / "diagnostics/gphi_structured_eta_fresh_wide_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def record(path: Path, expected: str | None = None) -> dict[str, Any]:
    observed = sha256(path)
    if expected is not None and observed != expected:
        raise RuntimeError((path, expected, observed))
    return {"path": str(path.resolve()), "sha256": observed, "bytes": path.stat().st_size}


def main() -> None:
    outputs = [HERE / "frozen_asset_hashes.json", HERE / "dataset_424_manifest.json",
               HERE / "basin_subset_manifest.json", HERE / "fresh32_manifest.json"]
    if any(path.exists() for path in outputs):
        raise RuntimeError("refusing to overwrite frozen audit manifests")

    meta_rows = [json.loads(line) for line in (DATA / "sample_metadata.jsonl").read_text().splitlines()]
    state_rows = [json.loads(line) for line in (DATA / "state_manifest.jsonl").read_text().splitlines()]
    by_state: dict[str, dict[str, Any]] = {}
    for row in meta_rows:
        state_id = row["state_id"]
        canonical = [float(x) for x in row["eta_best_metadata_only"]]
        if state_id in by_state:
            if by_state[state_id]["canonical_eta"] != canonical:
                raise RuntimeError((state_id, "Flow-variant eta conflict"))
            by_state[state_id]["sample_count"] += 1
            continue
        by_state[state_id] = {
            "state_id": state_id, "state_index": int(row["state_index"]),
            "split": row["split"], "category": row["category"],
            "source_trajectory": row["source_trajectory"], "leakage_group": row["leakage_group"],
            "canonical_eta": canonical, "zero_eta": canonical == [0.0, 0.0, 0.0],
            "E_near_metadata_only": row.get("E_near_metadata_only", []),
            "oracle_J_min_metadata_only": row.get("oracle_J_min_metadata_only"), "sample_count": 1,
        }
    state_manifest = {row["state_id"]: row for row in state_rows}
    if len(by_state) != 424 or len(state_manifest) != 424:
        raise RuntimeError((len(by_state), len(state_manifest)))
    states = []
    for state_id, item in sorted(by_state.items(), key=lambda pair: pair[1]["state_index"]):
        source = state_manifest[state_id]
        # The immutable merged manifest keeps relative state names; base-state
        # payloads remain in the startup-complete lineage, while the 11 newly
        # added strict-deadlock payloads live beside the merged dataset.
        state_root = DATA if item["category"] == "STRICT_DEADLOCK" else ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
        state_path = state_root / source["state_file"]
        if sha256(state_path) != source["state_sha256"]:
            raise RuntimeError((state_id, "state hash mismatch"))
        item.update({"state_file": str(state_path.resolve()), "state_sha256": source["state_sha256"],
                     "absolute_step": int(source.get("absolute_step", source.get("physical_step", source.get("step", 0)))),
                     "rng_namespace": None if source.get("rng_namespace") is None else int(source["rng_namespace"])})
        states.append(item)
    if sum(x["sample_count"] for x in states) != 27136 or any(x["sample_count"] != 64 for x in states):
        raise RuntimeError("unexpected Flow variants per state")

    dataset = {
        "schema": "direct_eta_dataset_424_manifest_v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "unique_states": 424, "feature_samples": 27136, "feature_dimension": 214,
        "flow_variants_per_state": 64, "canonical_eta_semantics": "eta_best_metadata_only; never inferred from g",
        "eta_low": [0.0, -0.53125, -0.125], "eta_high": [1.25, 0.5, 0.75], "states": states,
    }
    dataset["content_sha256"] = canonical_hash(dataset)
    atomic_json(HERE / "dataset_424_manifest.json", dataset)

    basin_seed = 2026092603
    permutation = np.random.default_rng(basin_seed).permutation(424).tolist()
    selected = [states[index] for index in permutation[:32]]
    basin = {
        "schema": "direct_eta_basin_subset_v1", "status": "FROZEN_BEFORE_NEW_BASIN_RESULTS",
        "permutation_seed": basin_seed, "permutation_algorithm": "numpy.default_rng(seed).permutation(424)",
        "selection_rule": "first 32 unique states; no outcome/category conditioning",
        "full_permutation_state_indices": permutation, "selected_states": selected,
        "naturally_occurring_composition": {
            "zero": sum(x["zero_eta"] for x in selected), "active": sum(not x["zero_eta"] for x in selected),
            "categories": {name: sum(x["category"] == name for x in selected) for name in sorted({x["category"] for x in states})},
        },
    }
    basin["content_sha256"] = canonical_hash(basin)
    atomic_json(HERE / "basin_subset_manifest.json", basin)

    source_fresh_path = FRESH / "fresh_wide_manifest.json"
    source_fresh = json.loads(source_fresh_path.read_text())
    if not source_fresh.get("frozen_before_rollout") or len(source_fresh["episodes"]) != 200:
        raise RuntimeError("fresh WIDE manifest is not authoritative/frozen")
    fresh_seed = 2026092604
    fresh_permutation = np.random.default_rng(fresh_seed).permutation(200).tolist()
    fresh_selected = [source_fresh["episodes"][index] for index in fresh_permutation[:32]]
    fresh = {
        "schema": "direct_eta_fresh32_step0_manifest_v1", "status": "FROZEN_BEFORE_NEW_ORACLE_RESULTS",
        "source_manifest": str(source_fresh_path), "source_manifest_sha256": sha256(source_fresh_path),
        "source_manifest_content_sha256": source_fresh["content_sha256"],
        "permutation_seed": fresh_seed, "permutation_algorithm": "numpy.default_rng(seed).permutation(200)",
        "selection_rule": "first 32 episodes; no Safety/eta outcome conditioning",
        "full_permutation_episode_indices": fresh_permutation, "selected_episodes": fresh_selected,
        "step": 0, "horizon": int(source_fresh["environment"]["max_steps"]),
        "dt": float(source_fresh["environment"]["dt"]),
    }
    fresh["content_sha256"] = canonical_hash(fresh)
    atomic_json(HERE / "fresh32_manifest.json", fresh)

    assets = {
        "schema": "direct_eta_basin_geometry_frozen_assets_v1", "status": "LOCKED_BEFORE_NEW_ROLLOUTS",
        "structured_eta_checkpoint": record(ETA / "best_fixed_d_eta_checkpoint.npz",
            "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095"),
        "eta_loader": record(ETA / "eta_model.py"), "eta_targets": record(ETA / "eta_targets.npz"),
        "samples": record(DATA / "samples.npz", "79d7da0492d9b414c03ce53f9ee826c54ac7dce3509b2cd3d086fc1cf852deb9"),
        "sample_metadata": record(DATA / "sample_metadata.jsonl", "03714a831f9f98a091ae6e7be8aa67c9b8243e3c81f0ff8098f07b835f691213"),
        "state_manifest": record(DATA / "state_manifest.jsonl", "fda79c67cb009b11a5738e64e4b2470bae52699f2b188700fcd0b8c5fc0c12ef"),
        "feature_builder": record(DATA / "startup_feature_builder.py", "c15d351a5ac89daa8e51eaf6965498ff58d16bdf4d6e9ef271dfc179afd8bb84"),
        "flowbc": record(SYS / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl",
            "8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32"),
        "environment": record(SYS / "single_integrator/environment.py", "427b1c0db1d68698e095a95e50a4404bae6c806a0f87351bfac431f6eeefd49b"),
        "cbf": record(SYS / "single_integrator/cbf.py", "841a2dbb74676599d8c4187de9cf29920a6eda02c4372e29060ce6ca451ade48"),
        "projection": record(ROOT / "diagnostics/success_basin_multimodality/exact_projector.py",
            "e29d510dc1752f138bdfcc491f8f5008dbcd215282301396852bdbfd754ac544"),
        "eta_basis": record(ROOT / "diagnostics/cl_fhcb/closed_loop.py",
            "aeca60cc1968733ca2dde4535c0de11a5431e2adcbac01027005a2ffc6694fb0"),
        "cached_oracle_v3": record(ROOT / "diagnostics/gphi_training_dataset_v3/oracle_search_results.jsonl"),
        "cached_oracle_v4": record(ROOT / "diagnostics/gphi_training_dataset_v4/new_oracle_search_results.jsonl"),
        "cached_oracle_startup": record(ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1/startup_oracle_search_results.jsonl"),
        "strict_deadlock_capacity": record(ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1/robust_basin_validation.csv"),
        "source_fresh_wide_manifest": record(source_fresh_path),
        "audit_protocol": record(HERE / "protocol.md"), "experiment_budget": record(HERE / "experiment_budget.json"),
        "dataset_424_manifest": record(HERE / "dataset_424_manifest.json"),
        "basin_subset_manifest": record(HERE / "basin_subset_manifest.json"),
        "fresh32_manifest": record(HERE / "fresh32_manifest.json"),
    }
    assets["content_sha256"] = canonical_hash(assets)
    atomic_json(HERE / "frozen_asset_hashes.json", assets)
    print(json.dumps({"status": "FROZEN", "dataset": dataset["content_sha256"],
                      "basin_subset": basin["content_sha256"], "fresh32": fresh["content_sha256"],
                      "basin_composition": basin["naturally_occurring_composition"]}, indent=2))


if __name__ == "__main__":
    main()
