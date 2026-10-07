#!/usr/bin/env python3
"""Freeze the unattended, non-adaptive common-256 P0 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
LONG = STUDY / "long_run_v2"
EXPECTED = {
    "checkpoint": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl", "6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd"),
    "dataset_manifest": (ROOT / "diagnostics/double_bottleneck_recovery_density_final/data/manifest.json", "771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56"),
    "macflow": (ROOT / "double_bottleneck/flowbc_4a_agent.py", "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8"),
    "environment": (ROOT / "double_bottleneck/environment.py", "3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc"),
    "hard_projection": (ROOT / "shared_control/hard_projection.py", "847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79"),
    "canonical_p0": (ROOT / "shared_control/diagnostic_corrector.py", "48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40"),
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
    for name in ("jobs", "raw", "logs", "figures", "representatives"):
        (LONG / name).mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, (path, expected) in EXPECTED.items():
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"frozen mismatch {name}: {actual} != {expected}")
        hashes[name] = actual
    hashes["toy_giveway_source_tree"] = tree_sha(ROOT / "toy_giveway")
    eta_artifact = json.loads((STUDY / "eta_points.json").read_text())
    episodes = json.loads((STUDY / "episode_catalog.json").read_text())["episodes"]
    points = eta_artifact["points"]
    if len(points) != 256 or len(episodes) != 85:
        raise RuntimeError("frozen population/design mismatch")
    hashes["eta_points"] = sha(STUDY / "eta_points.json")
    jobs = []
    for episode in episodes:
        for eta_index, point in enumerate(points):
            jobs.append({
                **episode,
                "stage": "long_global_256",
                "representation": "P0-3D",
                "eta_index": eta_index,
                **point,
                "job_id": f"long_global_256|{episode['episode_id']}|{eta_index:03d}",
            })
    (LONG / "jobs/global.json").write_text(json.dumps({"schema": "eta3_long_global_jobs_v1", "jobs": jobs}, indent=2, sort_keys=True) + "\n")
    protocol = {
        "schema": "double_bottleneck_eta3_unattended_protocol_v1",
        "registered_at": "2026-09-26T12:30:59+08:00",
        "frozen_hashes": hashes,
        "global": {"episodes": 85, "eta_points": 256, "rollouts": 21760, "same_points_for_every_episode": True},
        "domain": eta_artifact["domain"],
        "success": "all four agents reach their 0.08 m goal regions before or at frozen runtime termination within 850 steps",
        "representative": {"nearest_neighbors": 8, "primary": "maximum successful-neighbor count", "tie_1": "minimum L2 distance to eta=0 after division by full domain widths", "tie_2": "smallest eta_index"},
        "local": {"samples": 32, "normalized_coordinate_radius_per_axis": 0.05, "sobol_seed": 930051, "clip": True},
        "stochastic": {"seeds": list(range(1001, 1009))},
        "categories": {"low": "q < 0.25", "medium": "0.25 <= q < 0.75", "high": "q >= 0.75"},
        "status": {"PASS": "N_exist>=31 and median_Q_local>=0.25 and median_Q_seed>=0.50", "REVISE": "N_exist>=16 and PASS false", "REJECT": "N_exist<=15"},
        "prohibitions": ["adaptive eta", "representation expansion", "training", "canonical changes", "early stop"],
    }
    (LONG / "PREREGISTRATION.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (LONG / "run_manifest.json").write_text(json.dumps({"schema": "eta3_long_run_manifest_v1", "protocol": "PREREGISTRATION.json", "global_jobs": len(jobs), "completed_global": 0, "state": "registered"}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"jobs": len(jobs), "hashes_verified": len(hashes), "toy_hash": hashes["toy_giveway_source_tree"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
