#!/usr/bin/env python3
"""Pre-register and materialize the full common-256 P0-3D experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
CAPACITY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
PRIOR = ROOT / "diagnostics/double_bottleneck_eta3_basin"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(paths):
    rows = []
    for path in sorted(paths):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main() -> int:
    for name in ("jobs", "raw", "logs", "local", "stochastic", "figures", "representatives"):
        (STUDY / name).mkdir(parents=True, exist_ok=True)
    designs = json.loads((CAPACITY / "parameter_designs.json").read_text())
    samples = designs["P0-3D"]["stage_a"]
    if len(samples) != 256 or len({row["parameter_id"] for row in samples}) != 256:
        raise RuntimeError("expected the frozen common 256-point P0 design")
    catalog_source = json.loads((PRIOR / "episode_catalog.json").read_text())
    episodes = []
    for population, source in (("safe_timeout_target", catalog_source["targets"]), ("baseline_success_control", catalog_source["controls"])):
        for row in source:
            episodes.append({**row, "population": population})
    if len(episodes) != 85 or sum(row["population"] == "safe_timeout_target" for row in episodes) != 61:
        raise RuntimeError("frozen episode population mismatch")

    eta_artifact = {
        "schema": "double_bottleneck_eta3_common_256_v1",
        "source_sha256": sha(CAPACITY / "parameter_designs.json"),
        "domain": {"eta1_goal": [0.5, 1.25], "eta2_safe": [-0.5, 0.5], "eta3_relation": [0.0, 0.75]},
        "points": samples,
    }
    (STUDY / "eta_points.json").write_text(json.dumps(eta_artifact, indent=2, sort_keys=True) + "\n")
    (STUDY / "episode_catalog.json").write_text(json.dumps({"schema": "double_bottleneck_eta3_full_population_v1", "episodes": episodes}, indent=2, sort_keys=True) + "\n")

    jobs = []
    for episode in episodes:
        for sample in samples:
            jobs.append({
                **episode,
                "pilot_stratum": "full_timeout" if episode["population"] == "safe_timeout_target" else "full_control",
                "stage": "global_256",
                "representation": "P0-3D",
                **sample,
                "job_id": f"global_256|P0-3D|{episode['episode_id']}|{sample['parameter_id']}",
            })
    all_manifest = {"schema": "double_bottleneck_eta3_full_sobol_jobs_v1", "stage": "global_256", "jobs": jobs}
    (STUDY / "jobs/global_all.json").write_text(json.dumps(all_manifest, indent=2, sort_keys=True) + "\n")

    catalog_ids = {row["episode_id"] for row in episodes}
    design_by_id = {row["parameter_id"]: row for row in samples}
    cached = {}
    pilot_paths = list((CAPACITY / "raw").glob("P0_3D_stage_a*_shard*.jsonl"))
    for row in load_jsonl(pilot_paths):
        if row["parameter_id"] == "ZERO" or row["episode_id"] not in catalog_ids:
            continue
        key = (row["episode_id"], row["parameter_id"])
        if key in cached:
            raise RuntimeError(f"duplicate pilot cache row {key}")
        expected = design_by_id[row["parameter_id"]]
        if not np.allclose(row["theta"], expected["theta"], atol=1e-14, rtol=0):
            raise RuntimeError(f"pilot theta mismatch {key}")
        cached[key] = {
            **row,
            "job_id": f"global_256|P0-3D|{row['episode_id']}|{row['parameter_id']}",
            "stage": "global_256",
            "representation": "P0-3D",
            "sample_type": expected["sample_type"],
            "cache_source": "capacity_dense_p0_pilot",
        }

    inherited_paths = list((CAPACITY / "raw").glob("P1_Agent6_full_p0_inheritance_shard*.jsonl"))
    for row in load_jsonl(inherited_paths):
        source_id = row["parameter_id"].removeprefix("INHERIT_")
        key = (row["episode_id"], source_id)
        if key in cached:
            continue
        if source_id not in design_by_id:
            raise RuntimeError(f"unknown inherited P0 id {source_id}")
        theta6 = np.asarray(row["theta"], dtype=np.float64)
        theta3 = np.asarray((theta6[0], theta6[4], theta6[5]))
        expected = design_by_id[source_id]
        if not np.allclose(theta3, expected["theta"], atol=1e-14, rtol=0):
            raise RuntimeError(f"inheritance theta mismatch {key}")
        cached[key] = {
            **row,
            "job_id": f"global_256|P0-3D|{row['episode_id']}|{source_id}",
            "stage": "global_256",
            "representation": "P0-3D",
            "parameter_id": source_id,
            "sample_type": expected["sample_type"],
            "theta": theta3.tolist(),
            "cache_source": "exact_agent6_embedding_of_p0",
        }
    if len(cached) != 5010:
        raise RuntimeError(f"unexpected cache size {len(cached)} != 5010")
    cached_path = STUDY / "raw/global_cached.jsonl"
    cached_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for _, row in sorted(cached.items())))
    pending = [job for job in jobs if (job["episode_id"], job["parameter_id"]) not in cached]
    if len(pending) != 16750:
        raise RuntimeError(f"unexpected pending size {len(pending)} != 16750")
    (STUDY / "jobs/global_pending.json").write_text(json.dumps({"schema": "double_bottleneck_eta3_full_sobol_jobs_v1", "stage": "global_256", "jobs": pending}, indent=2, sort_keys=True) + "\n")
    cache_manifest = {
        "schema": "double_bottleneck_eta3_full_sobol_cache_v1",
        "all_jobs": len(jobs),
        "cached_jobs": len(cached),
        "pending_jobs": len(pending),
        "cached_sources": {
            "capacity_dense_p0_pilot": sum(row["cache_source"] == "capacity_dense_p0_pilot" for row in cached.values()),
            "exact_agent6_embedding_of_p0": sum(row["cache_source"] == "exact_agent6_embedding_of_p0" for row in cached.values()),
        },
        "source_hashes": {
            "capacity_design": sha(CAPACITY / "parameter_designs.json"),
            "capacity_preregistration": sha(CAPACITY / "PREREGISTRATION.json"),
            "capacity_inheritance_amendment": sha(CAPACITY / "H0_INHERITANCE_AMENDMENT.json"),
            "prior_episode_catalog": sha(PRIOR / "episode_catalog.json"),
        },
    }
    (STUDY / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n")

    prereg = {
        "schema": "double_bottleneck_eta3_full_sobol_preregistration_v1",
        "registered_at": "2026-09-26T11:36:39+08:00",
        "frozen": {
            "checkpoint": "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd",
            "dataset_manifest": "771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56",
            "macflow_source": "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8",
            "environment": "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc",
            "hard_projection": "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79",
            "canonical_p0": "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40",
            "horizon_steps": 850,
            "success": "all four agents reach their respective 0.08 m goals before frozen runtime termination",
        },
        "p0_equation": {
            "goal": "B_goal,i = radial_bound_0.5(goal_i - position_i)",
            "relation": "B_rel,i = (1/(N-1)) sum_(j != i) radial_bound_0.5(position_i-position_j)",
            "correction": "g_i = eta1 B_goal,i + eta2 u_safe,i + eta3 B_rel,i",
            "execution": "u_exec = Pi_U(x)(u_safe + g_eta)",
        },
        "global": {"eta_points": 256, "same_points_all_episodes": True, "eta_points_sha256": sha(STUDY / "eta_points.json"), "timeouts": 61, "controls": 24, "domain_expansion": False},
        "local": {
            "representatives_per_positive_episode": ["minimum Euclidean eta norm", "maximum successful Sobol support within normalized Euclidean radius 0.20", "maximum global timeout coverage then control preservation"],
            "deduplicate_representatives": True,
            "normalized_radius": 0.03125,
            "axis_offsets": 6,
            "scrambled_sobol_offsets": 32,
            "sobol_seed": 920031,
            "clip_to_global_domain": True,
            "classification": {"broad": "local success fraction >= 0.75", "narrow": "0.10 <= fraction < 0.75", "isolated": "fraction < 0.10"},
            "episode_locally_robust": "maximum representative local fraction >= 0.50",
        },
        "stochastic": {
            "new_macflow_seeds": [1103, 1201, 1301, 1409, 1511, 1601, 1709, 1801],
            "pairs": "same deduplicated representatives used for local robustness",
            "classification": {"seed_robust": "success fraction >= 0.75", "seed_sensitive": "0.25 <= fraction < 0.75", "lucky": "fraction < 0.25"},
        },
        "geometry_representatives": {"high": "maximum positive rho_B", "median": "lower median positive rho_B after episode-id tie break", "low": "minimum positive rho_B", "none": "lexicographically first empty-basin episode"},
        "overlap": {"metric": "Jaccard on common 256-bit basin membership", "cluster_edge_threshold": 0.25},
        "behavior": {"episodes": "high, median-positive, and low-positive geometry representatives", "eta": "the selected maximum-global-coverage representative for the episode", "comparison": "eta=0 with identical initial state and original MACFlow seed"},
        "decision": {"pass": "meaningful existence with many locally and seed-robust representatives, not dominated by isolated luck", "revise": "meaningful existence but sparse/fragile local, seed, or overlap structure", "reject": "most episodes remain empty or nearly all successes are isolated/nonreproducible"},
        "prohibitions": ["representation expansion", "G_phi training", "domain expansion", "per-episode adaptive eta points", "MACFlow change", "safety change", "horizon change", "basis change"],
    }
    (STUDY / "PREREGISTRATION.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_jobs": len(jobs), "cached": len(cached), "pending": len(pending), "eta_points": len(samples), "episodes": len(episodes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
