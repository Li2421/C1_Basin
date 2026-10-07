#!/usr/bin/env python3
"""Materialize cached P0 pilot rows and frozen P1 pilot jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA3 = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    pilot = json.loads((OUT / "pilot_catalog.json").read_text())["episodes"]
    pilot_ids = {row["episode_id"] for row in pilot}
    points = json.loads((ETA3 / "eta_points.json").read_text())["points"]
    scale = json.loads((OUT / "P1_SCALE.json").read_text())["scale"]
    p0_rows = []
    for path in sorted((ETA3 / "long_run_v2/raw").glob("global_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["episode_id"] in pilot_ids:
                p0_rows.append(row)
    if len(p0_rows) != 24 * 256 or len({(row["episode_id"], int(row["eta_index"])) for row in p0_rows}) != 24 * 256:
        raise RuntimeError("cached P0 pilot matrix incomplete")
    p0_rows.sort(key=lambda row: (row["episode_id"], int(row["eta_index"])))
    cache = OUT / "raw/P0_pilot_global_cached.jsonl"
    cache.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in p0_rows))

    jobs = []
    for episode in pilot:
        for eta_index, point in enumerate(points):
            jobs.append({
                **episode,
                "stage": "pilot_global_256",
                "representation": "P1-OrthoFlow3",
                "eta_index": eta_index,
                **point,
                "ortho_scale": scale,
                "job_id": f"pilot_global_256|P1-OrthoFlow3|{episode['episode_id']}|{eta_index:03d}",
            })
    manifest = {
        "schema": "eta_basis_redesign_jobs_v1",
        "representation": "P1-OrthoFlow3",
        "ortho_scale": scale,
        "ortho_scale_sha256": sha(OUT / "P1_SCALE.json"),
        "eta_points_sha256": sha(ETA3 / "eta_points.json"),
        "jobs": jobs,
    }
    path = OUT / "jobs/P1_pilot_global_256.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    run = json.loads((OUT / "run_manifest.json").read_text())
    run.update({
        "state": "pilot_global_registered",
        "p0_cached_pilot_rows": len(p0_rows),
        "p0_cached_sha256": sha(cache),
        "p1_pilot_global_jobs": len(jobs),
        "p1_pilot_jobs_sha256": sha(path),
    })
    (OUT / "run_manifest.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"p0_cached": len(p0_rows), "p1_jobs": len(jobs), "scale": scale}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
