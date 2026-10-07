#!/usr/bin/env python3
"""Freeze the non-adaptive P0 seed-robustness protocol and candidate jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
LONG = SOURCE / "long_run_v2"
OUT = ROOT / "diagnostics/double_bottleneck_eta3_seed_robustness"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=float)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=float)
WIDTH = HIGH - LOW

EXPECTED_FILES = {
    "checkpoint": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl", "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd"),
    "dataset_manifest": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/data/manifest.json", "771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56"),
    "macflow": (ROOT / "double_bottleneck/flowbc_4a_agent.py", "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8"),
    "environment": (ROOT / "double_bottleneck/environment.py", "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc"),
    "hard_projection": (ROOT / "shared_control/hard_projection.py", "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79"),
    "canonical_p0": (ROOT / "shared_control/diagnostic_corrector.py", "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40"),
    "eta_points": (SOURCE / "eta_points.json", "6a0732c5ad35bc2000e61be6c2825034058b30aad09458cf2cf5975cc501f258"),
    "episode_catalog": (SOURCE / "episode_catalog.json", "ddb1e32646fd771e84fd4cce024e7bcf416f26a3a08c75557f6bb94c8b7cfe5c"),
    "frozen_timeout_basin": (LONG / "timeout_basin_matrix.json", "0951e015531505bdbaa6d604c56879784862fa295b608814c1b3824cc1d2c654"),
    "frozen_global_manifest": (LONG / "global_rollouts_manifest.json", "5a1f0bf4fe00f6ec3302dae212f2da0a84c48f9d1a774d13d020a7699f3a8cbb"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        digest.update(str(item.relative_to(path)).encode() + b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def main() -> int:
    hashes = {}
    for name, (path, expected) in EXPECTED_FILES.items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"frozen mismatch {name}: {actual} != {expected}")
        hashes[name] = actual
    hashes["toy_giveway_source_tree"] = tree_sha(ROOT / "toy_giveway")
    expected_toy = "10c8ac15a724ef0c80d8c0e4c0de63d0b28fd89a7956c030734e8a43bd8b4a26"
    if hashes["toy_giveway_source_tree"] != expected_toy:
        raise RuntimeError("Toy Give-Way source tree differs from frozen reference")

    eta_artifact = json.loads((SOURCE / "eta_points.json").read_text())
    points = eta_artifact["points"]
    basin = json.loads((LONG / "timeout_basin_matrix.json").read_text())
    episodes = basin["episodes"]
    positives = [episode for episode in episodes if episode["basin_exists"]]
    if len(points) != 256 or len(episodes) != 61 or len(positives) != 38:
        raise RuntimeError("frozen P0 population/design mismatch")
    if any(episode["successful_eta_count"] != len(episode["successful_eta_indices"]) for episode in positives):
        raise RuntimeError("frozen basin count mismatch")

    theta = np.asarray([point["theta"] for point in points], dtype=float)
    unit = (theta - LOW) / WIDTH
    selection = []
    jobs = []
    for episode in positives:
        successful = list(map(int, episode["successful_eta_indices"]))
        ranked = []
        for eta_index in successful:
            distances = np.linalg.norm(unit - unit[eta_index], axis=1)
            distances[eta_index] = np.inf
            nearest = np.argsort(distances, kind="stable")[:8]
            neighbor_count = sum(int(index) in successful for index in nearest)
            zero_distance = float(np.linalg.norm(theta[eta_index] / WIDTH))
            ranked.append({
                "eta_index": eta_index,
                "parameter_id": points[eta_index]["parameter_id"],
                "theta": theta[eta_index].tolist(),
                "successful_neighbors_among_8": int(neighbor_count),
                "normalized_distance_to_eta_zero": zero_distance,
                "nearest_eta_indices": nearest.astype(int).tolist(),
            })
        ranked.sort(key=lambda row: (-row["successful_neighbors_among_8"], row["normalized_distance_to_eta_zero"], row["eta_index"]))
        selected = ranked if len(ranked) <= 8 else ranked[:8]
        selection.append({
            "episode_id": episode["episode_id"],
            "successful_eta_count": len(successful),
            "candidate_rule": "all successful P0 eta because |B_i| <= 8" if len(successful) <= 8 else "top 8 by frozen deterministic ranking",
            "candidates": selected,
        })
        base = {key: episode[key] for key in ("episode_id", "population", "set", "family_id", "regime", "rollout_id", "baseline_outcome")}
        for candidate in selected:
            for seed in range(2001, 2017):
                jobs.append({
                    **base,
                    "stage": "candidate_seed16",
                    "representation": "P0-3D",
                    "eta_index": candidate["eta_index"],
                    "parameter_id": candidate["parameter_id"],
                    "sample_type": "existing_successful_global_candidate",
                    "theta": candidate["theta"],
                    "seed": seed,
                    "job_id": f"candidate_seed16|{episode['episode_id']}|{candidate['eta_index']:03d}|{seed}",
                })

    if len(selection) != 38 or sum(len(row["candidates"]) for row in selection) != 58 or len(jobs) != 928:
        raise RuntimeError("candidate-selection cardinality mismatch")

    protocol = {
        "schema": "double_bottleneck_eta3_seed_robustness_protocol_v1",
        "registered_at": "2026-09-26T18:20:00+08:00",
        "frozen_hashes": hashes,
        "source": {
            "positive_episodes": 38,
            "total_timeout_episodes": 61,
            "candidate_eta_pairs": 58,
            "maximum_B_i_size": max(row["successful_eta_count"] for row in positives),
            "new_global_search": False,
        },
        "candidate_seed_evaluation": {"seeds": list(range(2001, 2017)), "rollouts": 928},
        "eta_robust_selection": {
            "primary": "maximum Q_seed over 16 seeds",
            "tie_1": "larger successful-neighbor count among 8 nearest global Sobol points",
            "tie_2": "smaller normalized L2 distance to eta=0 after division by full domain widths",
            "tie_3": "smaller eta index",
        },
        "categories": {
            "strongly_robust": "Q_max >= 0.75",
            "moderately_robust": "0.50 <= Q_max < 0.75",
            "weakly_robust": "0.25 <= Q_max < 0.50",
            "seed_fragile": "Q_max < 0.25",
        },
        "local_confirmation": {
            "eligibility": "Q_max >= 0.50",
            "eta_points": 16,
            "seeds_per_eta": list(range(3001, 3009)),
            "normalized_coordinate_radius_per_axis": 0.05,
            "sobol_seed": 20260926,
            "clip_to_original_domain": True,
        },
        "control_preservation": {"controls": 24, "seeds_per_control": [4001, 4002, 4003, 4004], "deduplicate_identical_eta": True},
        "cross_state_reuse": {
            "timeout_states": 61,
            "screen_seed": 5001,
            "deduplicate_identical_eta": True,
            "top5_tie_break": ["smaller normalized distance to eta=0", "smaller eta index"],
            "confirmation_seeds": list(range(6001, 6009)),
            "confirmation_population": "states rescued by screen seed",
        },
        "decision": {
            "PASS": "at least 19/38 have Q_max >= 0.50 and local robustness is nontrivial for a meaningful subset",
            "REVISE": "some have Q_max >= 0.50 but fewer than 19/38, or local robust neighborhoods are very sparse",
            "REJECT": "almost all candidate eta remain below Q_max=0.25",
        },
        "prohibitions": ["new global eta", "representation expansion", "training", "canonical changes", "adaptive search"],
    }
    (OUT / "PREREGISTRATION.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (OUT / "candidate_selection.json").write_text(json.dumps({"episodes": selection}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/candidate_seed16.json").write_text(json.dumps({"jobs": jobs}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "double_bottleneck_eta3_seed_robustness_manifest_v1",
        "protocol": "PREREGISTRATION.json",
        "candidate_jobs": len(jobs),
        "completed_candidate_jobs": 0,
        "state": "candidate_jobs_registered",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"positive_episodes": len(positives), "candidate_pairs": 58, "candidate_jobs": len(jobs), "max_B_i": max(row["successful_eta_count"] for row in positives), "hashes_verified": len(hashes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
