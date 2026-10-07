"""Freeze migration assets, exact search config, and outcome-blind subset."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/orthoflow3_representation_migration_v1"
ARCHIVE = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA_DESIGN = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json"
DATASET = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1/dataset_424_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def artifact(path: Path, expected: str | None = None) -> dict:
    digest = sha(path)
    if expected is not None and digest != expected:
        raise RuntimeError((str(path), digest, expected))
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def state_seeds(category: str) -> list[int]:
    if category == "STARTUP":
        base = 95710001
    elif category == "STRICT_DEADLOCK":
        base = 95310001
    else:
        base = 95210001
    return list(range(base, base + 64))


def main() -> None:
    frozen_paths = [
        HERE / "p0_implementation_manifest.json",
        HERE / "orthoflow3_authoritative_manifest.json",
        HERE / "basis_hashes.json",
        HERE / "migration_subset_manifest.json",
        HERE / "orthoflow3_search_config.json",
    ]
    if any(path.exists() for path in frozen_paths):
        raise RuntimeError("refusing to overwrite frozen migration manifests")

    authoritative = {
        "schema": "orthoflow3_authoritative_manifest_v1",
        "status": "UNIQUE_IMPLEMENTATION_RESOLVED",
        "validated_name": "P1-OrthoFlow3",
        "implementation": artifact(
            ARCHIVE / "tools/bases.py",
            "51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38",
        ),
        "implementation_symbols": ["raw_ortho_flow", "basis_terms", "correction"],
        "scale": artifact(
            ARCHIVE / "P1_SCALE.json",
            "ecdca9e63e5d345ec45ae264bcdbc1006f9281cc5a32c3fdfec901e55609a133",
        ),
        "preregistration": artifact(
            ARCHIVE / "PREREGISTRATION.json",
            "6ab6350dd8ae52af988549543a35437bc335854838cae820a7ad9112ad5e1d2a",
        ),
        "eta_design": artifact(
            ETA_DESIGN,
            "6a0732c5ad35bc2000e61be6c2825034058b30aad09458cf2cf5975cc501f258",
        ),
        "validated_report": artifact(ARCHIVE / "REPORT.md"),
        "validated_summary": artifact(
            ARCHIVE / "FINAL_SUMMARY.json",
            "8303e41110bd8a4549406baec04149cabe38b34e4b0527ece9181b5d7561c7b8",
        ),
        "full_global_summary": artifact(
            ARCHIVE / "full_global_summary.json",
            "c917da7ab4bd86bb36f626afecc6c6bb7e53eec6ddebbb5a9fc535402df0882b",
        ),
        "archived_implementation_checks": artifact(ARCHIVE / "implementation_checks.json"),
        "basis_audit": artifact(ARCHIVE / "basis_audit.json"),
        "state_anchor_manifest": artifact(ARCHIVE / "state_anchor_manifest.json"),
        "validated_results": {
            "pilot_p0_basin_exists": 10,
            "pilot_orthoflow3_basin_exists": 12,
            "full_orthoflow3_basin_exists": 61,
            "full_orthoflow3_population": 61,
            "selected_center_seed16_success": "61/61 centers achieved 16/16",
            "valid_scientific_rollouts": 32255,
            "collisions": 0,
        },
    }
    authoritative["content_sha256"] = canonical_hash(authoritative)
    atomic_json(HERE / "orthoflow3_authoritative_manifest.json", authoritative)

    p0 = {
        "schema": "p0_implementation_manifest_v1",
        "preservation_rule": "historical sources remain byte-unchanged; explicit selector only in new code",
        "giveway_historical_corrector": artifact(ROOT / "diagnostics/cl_fhcb/closed_loop.py"),
        "dimension_generic_historical_corrector": artifact(
            ROOT / "shared_control/diagnostic_corrector.py",
            "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40",
        ),
        "historical_name": "P0-3D",
        "future_selector": "p0",
    }
    p0["content_sha256"] = canonical_hash(p0)
    atomic_json(HERE / "p0_implementation_manifest.json", p0)

    basis_hashes = {
        "schema": "basis_family_hashes_v1",
        "new_common_interface": artifact(ROOT / "shared_control/basis_families.py"),
        "new_interface_tests": artifact(ROOT / "tests/test_basis_families.py"),
        "authoritative_orthoflow3": authoritative["implementation"],
        "authoritative_scale": authoritative["scale"],
        "legacy_p0": p0["dimension_generic_historical_corrector"],
    }
    basis_hashes["content_sha256"] = canonical_hash(basis_hashes)
    atomic_json(HERE / "basis_hashes.json", basis_hashes)

    dataset = json.loads(DATASET.read_text())
    if dataset["unique_states"] != 424 or len(dataset["states"]) != 424:
        raise RuntimeError("authoritative 424-state manifest mismatch")
    permutation_seed = 2026092801
    permutation = np.random.default_rng(permutation_seed).permutation(424).tolist()
    selected = []
    for rank, index in enumerate(permutation[:32]):
        state = dict(dataset["states"][index])
        state.update(
            {
                "selection_rank": rank,
                "matched_flow_seeds": state_seeds(state["category"]),
                "exact_screen_seed": state_seeds(state["category"])[0],
            }
        )
        selected.append(state)
    subset = {
        "schema": "orthoflow3_migration_subset_v1",
        "status": "FROZEN_BEFORE_NEW_OUTCOMES",
        "source_manifest": artifact(DATASET),
        "permutation_seed": permutation_seed,
        "permutation_algorithm": "numpy.random.default_rng(seed).permutation(424)",
        "selection_rule": "first 32 unique states without inspecting labels/outcomes",
        "full_permutation_state_indices": permutation,
        "selected_states": selected,
        "post_selection_descriptive_composition": {
            "categories": {
                name: sum(row["category"] == name for row in selected)
                for name in sorted({row["category"] for row in selected})
            },
            "historical_p0_zero": sum(row["zero_eta"] for row in selected),
            "historical_p0_active": sum(not row["zero_eta"] for row in selected),
        },
    }
    subset["content_sha256"] = canonical_hash(subset)
    atomic_json(HERE / "migration_subset_manifest.json", subset)

    design = json.loads(ETA_DESIGN.read_text())
    prereg = json.loads((ARCHIVE / "PREREGISTRATION.json").read_text())
    scale = json.loads((ARCHIVE / "P1_SCALE.json").read_text())
    search = {
        "schema": "orthoflow3_giveway_migration_search_v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "basis_family": "orthoflow3",
        "basis_version": "p1_orthoflow3_v1",
        "basis_dimension": 3,
        "eta_domain": design["domain"],
        "common_candidate_points": len(design["points"]),
        "candidate_design": artifact(ETA_DESIGN),
        "global_scale": scale,
        "archived_candidate_selection": prereg["search"]["candidate_selection"],
        "archived_candidate_seed_count": len(prereg["search"]["candidate_seeds"]),
        "migration_flow_seed_bridge": (
            "same 16/64 state-specific matched Flow streams as P0 dataset, preserving paired capacity comparison"
        ),
        "exact_screen": "all 256 candidates on first matched Flow stream",
        "candidate_screen": "selected <=8 candidates on first 16 matched streams",
        "robust_promotion": ">=63/64 B63; priority success16 desc, mean successful J_def asc, eta index asc",
        "canonical_rule": "B63 first; minimum mean successful J_def; eta index tie break",
        "j_def": "dt * sum ||u_exec-u_safe||^2",
        "interpolation_alphas": [0.25, 0.5, 0.75],
        "interpolation_screen_seeds": 8,
        "new_rollout_cap": 15000,
        "physical_step_cap": 8000000,
        "full_rebuild_wall_cap_minutes": 60,
    }
    search["content_sha256"] = canonical_hash(search)
    atomic_json(HERE / "orthoflow3_search_config.json", search)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "subset_sha256": subset["content_sha256"],
                "composition": subset["post_selection_descriptive_composition"],
                "basis_interface_sha256": basis_hashes["new_common_interface"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
